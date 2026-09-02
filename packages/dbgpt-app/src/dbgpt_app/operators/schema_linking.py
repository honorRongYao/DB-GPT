"""语义层（Schema Linking）检索算子。

替代默认的 HODatasourceRetrieverOperator，解决"给大模型的表结构参考不准"问题。
官方默认的数据源检索算子走的是向量摘要检索（top_k 截断字段、无表间关联关系、
无语义维护），导致 LLM 生成 SQL 时拿到的表结构不完整、join 关系靠猜。

本算子采用业界主流的两阶段 Schema Linking 思路：
  1. 全量轻目录：列出数据库中所有表（表名 + 表注释，注释中可维护业务语义与表间关联关系），
     目录很小（每张表一行），全部喂给 LLM 也不会有 token 压力；
  2. LLM 选表 + 关联关系：把目录和用户问题交给 LLM，LLM 判断要查询哪些表，
     以及表与表之间的关联关系（通过哪个字段 join、一对多/多对一）；
  3. 全字段加载：对 LLM 选中的表调用 get_columns() 加载全部字段（不截断，
     含类型/主键/字段注释），并过滤掉 LLM 编造的不存在表名。

输出类型与 HODatasourceRetrieverOperator 一致（HOContextBody），画布上可直接替换接线。
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from dbgpt.agent import AgentGenerateContext
from dbgpt.agent.resource.database import DBResource
from dbgpt.core import (
    LLMClient,
    ModelMessage,
    ModelMessageRoleType,
    ModelOutput,
    ModelRequest,
)
from dbgpt.core.awel import MapOperator
from dbgpt.core.awel.flow import (
    TAGS_ORDER_HIGH,
    IOField,
    OperatorCategory,
    Parameter,
    ViewMetadata,
    ui,
)
from dbgpt.model.operators import MixinLLMOperator
from dbgpt.util.i18n_utils import _

from .category_recall import DEFAULT_MAX_DISTANCE, TARGET_TABLE, recall_many
from .llm import HOContextBody

logger = logging.getLogger(__name__)

_DEFAULT_SELECT_TABLE_PROMPT = """你是一个数据库专家，负责为用户的查询问题选出正确的数据表，并判断表与表之间的关联关系。

下面是数据库 {db_name} 中所有可用表的目录（格式：表名 -- 表注释；表注释中可能包含该表的业务含义、字段说明以及与其他表的关联关系，请仔细阅读）：

{table_catalog}

请根据用户问题，从目录中选出生成 SQL 所需的相关表（可能是一张或多张），并判断表与表之间的关联关系（例如通过哪个字段 join、是一对多还是多对一）。

参考示例：
示例1：
用户问题：查询每个用户的订单总金额
输出：{{"tables": [{{"table": "orders", "relation": "通过 user_id 关联 user 表（多对一）"}}, {{"table": "user", "relation": "与 orders 通过 user_id 关联（一对多）"}}]}}

要求：
1. 表名必须与目录中的名字完全一致，原样输出，禁止编造、禁止改写大小写、禁止添加库名前缀。
2. 只能从目录中选择表，禁止编造不存在的表名。
3. 如果问题需要多张表，必须把建立关联所需的所有表都选出来；如果单表即可回答，只选一张，relation 留空字符串。
4. 选表数量尽量精简，一般不要超过 {max_selected_tables} 张。
5. 当用户问题中的过滤概念（品类、品牌、状态、类型、名称等业务维度）在事实表中只有外键（如 category_id）而没有对应的名称字段时，必须把该维度表一并选出用于 WHERE 过滤；宁可多选一张表，也不能让过滤条件无法准确表达。漏选维度表是常见错误，会导致 SQL 过滤条件错误、结果不准。
6. 多表关联时，每张选中的表都必须填写 relation，描述该表与其他表的关联关系，不能留空：
   - 如果该表通过某字段关联其他表（事实表），写"通过 XX 字段关联 YY 表（多对一）"，如：通过 user_id 关联 user 表（多对一）；
   - 如果该表是被其他表引用的维度表，写"被 YY 表通过 XX 字段引用（一对多）"，如：被 orders 表通过 user_id 引用（一对多）。
7. 输出严格 JSON，不要包含任何其他解释文字，格式如下：
{{
  "tables": [
    {{"table": "表名", "relation": "该表与其他表的关联关系说明，如：通过 user_id 关联 user 表（多对一）"}}
  ]
}}
"""

_DEFAULT_SELECT_VALUE_FIELDS_PROMPT = """你是一个数据库专家。下面是语义层根据用户问题选出的数据表及其完整字段定义（格式：字段名 类型 [主键] -- 注释）：

{schemas}

请判断：为了正确生成 SQL 的 WHERE 过滤条件，哪些字段的真实取值可能有参考价值？
只要该字段的取值在 SQL 中可能被用到（例如可能用于 WHERE 过滤、分组、精确匹配、模糊匹配，或取值属于枚举型/有限集合/大小写敏感需要原样书写的字段：品类、状态、日期、品牌、类型、ID 等），都应该选出来。

宁可多选也不要漏选：只要觉得可能有参考价值的字段都选出来，LLM 写 SQL 时参考真实取值可以避免猜错过滤值（取值猜错会导致过滤条件错误、查不到结果）。

输出每张表最多 5 个这样的字段（优先选最可能有参考价值的），格式为严格 JSON（只输出 JSON，不要任何解释）：
{{"value_fields": {{"表名1": ["字段a", "字段b"], "表名2": ["字段c"]}}}}

如果某张表确实没有任何需要参考取值的字段，该表就不出现在结果中。
"""

_DEFAULT_VERIFY_VALUE_FIELDS_PROMPT = """你是数据库专家。下面是已选表中需要参考真实取值的字段，以及这些字段查到的真实数据（每字段最多3个取值）：

{field_values}

用户问题：{question}

请判断：这些字段的真实取值，是否足以帮你正确写出 WHERE 过滤条件（品类、品牌、日期、状态等业务维度的取值是否准确、可用的过滤值是否齐全）？

重点检查：
1. 用户问题中出现的业务过滤概念（品类、品牌、商品类型等），是否已有字段能看到对应的真实取值？若取值里看不到问题提到的概念（如问题说"电动风扇"，但取值里只有"风扇/塔式风扇"），说明需要补充品类/标题类字段来确认真实叫法。
2. 时间过滤字段（如 dt、date）是否有取值参考？
3. 已选字段查出的取值是否异常（全空、类型不符、字段名错误）？

若已足够，输出：{{"sufficient": true, "add_fields": {}}}
若不足，从已选表的完整字段中补充字段，输出：
{{"sufficient": false, "add_fields": {{"表名": ["补充字段1", "补充字段2"]}}}}

要求：
1. 只能补充已选表真实存在的字段，禁止编造字段名。
2. 每张表补充字段不超过 5 个。
3. 输出严格 JSON，不要包含任何其他解释文字。
"""

_DEFAULT_VERIFY_TABLES_PROMPT = """你是一个数据库专家。下面是初步选出的数据表（含字段列表与表注释），以及数据库中全部可用表的目录。

当前已选表：
{selected_tables}

全表目录（补充表只能从这里选）：
{table_catalog}

请校验：当前已选表的字段，是否足以完整回答用户问题（包括 WHERE 过滤、聚合、分组、排序所需的全部维度）？重点检查：
1. 用户问题中出现的业务过滤概念（品类、品牌、状态、类型、名称等），已选表中是否有对应的名称字段可直接表达？
   如果只有外键（如 category_id）而没有名称字段（如 category_1、category_2），则缺少对应的维度表，必须补充。
2. 统计口径所需字段（销量、金额、时间等）是否齐全。
3. 时间等过滤字段是否存在，字段类型是否与问题口径匹配（如日期是 date、年月是 varchar 格式 202607）。

若已选表已足够，输出：{{"sufficient": true, "missing_tables": []}}
若不足，从全表目录中选出缺失的表（可多张），输出：
{{"sufficient": false, "missing_tables": [{{"table": "表名", "reason": "补充原因"}}]}}

要求：
1. 表名必须与全表目录完全一致，禁止编造。
2. 只补充真正缺失的表，不要重复已选中的表。
3. 输出严格 JSON，不要包含任何其他解释文字。
"""

_DEFAULT_REWRITE_QUESTION_PROMPT = """你是一个对话理解助手。下面是当前用户问题出现之前的最近几轮对话记录：

{history}

请判断：当前用户问题是否引用了前文对话中的内容？
判断依据（满足任意一条即可认为引用了前文）：
1. 出现指代词：那、这个、它、上面、刚才、环比、同比、继续、再、还、结果、数据等；
2. 缺少主语或实体：没有明确提到品类/品牌/表/统计对象；
3. 缺少时间基准或统计口径：如"环比是多少"没有说明对比的是哪一期、哪个范围的数据；
4. 与前一问属于同一话题的追问。

- 若引用了前文，把当前问题改写为一个自包含的完整问题：把前文中的实体（品类/品牌/商品等）、
时间范围（如 Q1、202601-202603）、过滤条件、统计口径明确写进问题，
使改写后的问题脱离前文也能独立理解。
- 若没有引用前文，保持原问题不变，原样输出。

只输出严格 JSON，不要包含任何其他解释文字：
{{"related": true 或 false, "rewritten_question": "改写后的完整问题（related 为 false 时与原问题相同）"}}
"""

_DEFAULT_EXTRACT_KEYWORDS_PROMPT = """你是一个数据分析助手。请从用户问题中提取"关键实体词"，用于后续语义检索（向量匹配）和选表参考。

提取规则：
1. 只提取需要"去库里匹配真实取值"的实体词：品类名（咖啡机、空调、风扇）、品牌名（苏泊尔、美的）、评论打标词（异味、噪音大、好评）、地区（广东省、华东地区）、商品/标题实体（咖啡机pro版）等。
2. 不提取：指标（销量、金额、评论数）、时间（2026年、Q1、上半年）、动作词（分析、对比、列出）、排序/占比修饰（前3、环比、占比）、虚词助词。
3. 用户问题包含多个独立诉求时（如"分析评论，同时对比销量"），每个诉求里的关键实体词都要提取，不要遗漏。
4. 同一个实体只保留一次，去重。
5. 品牌名与品类名连用时（如"德龙咖啡机""美的空调"），拆成品牌名和品类名两个词分别提取，不要保留组合词。
6. 不提取型号/版本后缀（如 pro、plus、max、2026款）；型号主体（如 AC-555）作为整体提取。
7. 提取结果若存在包含关系（如"德龙咖啡机"包含"咖啡机"），只保留最小粒度的词。

输出严格 JSON（只输出 JSON，不要任何解释文字）：
{"关键词列表": ["咖啡机", "异味"]}

示例1 输入：帮我分析2026年上半年咖啡机的异味问题
输出：{"关键词列表": ["咖啡机", "异味"]}

示例2 输入：分别统计广东省和华东地区电动风扇的销量
输出：{"关键词列表": ["广东省", "华东地区", "电动风扇"]}
"""

# 追问补全相关常量：历史只取最近几轮（每轮只留头尾）、单条内容截断，避免上下文撑爆模型输入
_REWRITE_HISTORY_MAX_ROUNDS = 5
_REWRITE_HISTORY_MAX_CHARS = 200
# 标准 DB-GPT 对话中 UserProxyAgent 的 role，用于按真实 conv_id 模糊检索整段会话历史
_REWRITE_HISTORY_ROLE = "Human"
_SCHEMA_LINKING_INJECT_MARKER = "以下数据表已由语义层根据你的问题选好"
_SCHEMA_LINKING_QUESTION_MARKER = "用户问题:"

_PARAMETER_DATASOURCE = Parameter.build_from(
    _("Datasource"),
    "datasource",
    type=DBResource,
    description=_(
        "数据源（必填）：选择要查询的数据库资源。"
        "语义层会从这个数据源读取全部表名、表注释和字段信息，"
        "请务必选择业务数据所在的库，选错会导致目录和表结构取不到。"
    ),
)

_PARAMETER_PROMPT_TEMPLATE = Parameter.build_from(
    _("Select Table Prompt Template"),
    "prompt_template",
    type=str,
    optional=True,
    default=_DEFAULT_SELECT_TABLE_PROMPT,
    description=_(
        "选表提示词（可选，默认已内置中文提示词）："
        "这是交给 LLM 让它从全量表目录中挑选相关表并判断表间关联关系的指令模板（system 消息），"
        "模板中可使用的占位符：{db_name} 数据库名、{table_catalog} 全量表目录、"
        "{max_selected_tables} 最大可选表数。"
        "用户问题不写在模板里，代码会自动作为 user 消息追加在提示词后面。"
        "当发现选表不准（选错表/漏选关联表）时，可以在这里调整指令；"
        "注意模板中的 JSON 示例花括号需要写成 {{ }} 双花括号（代码渲染时会还原为单花括号）。"
    ),
    ui=ui.DefaultUITextArea(),
)

_PARAMETER_MODEL = Parameter.build_from(
    _("Model Name"),
    "model",
    type=str,
    optional=True,
    default=None,
    description=_(
        "选表模型名称（可选）：指定用于“LLM 选表”环节的模型名称。"
        "留空时自动使用 DB-GPT 默认部署模型的第一个模型。"
        "一般无需填写，保持默认即可；仅在需要与生成 SQL 的模型区分开时才填写。"
    ),
)

_PARAMETER_LLM_CLIENT = Parameter.build_from(
    _("LLM Client"),
    "llm_client",
    type=LLMClient,
    optional=True,
    default=None,
    description=_(
        "LLM 客户端（可选）：指定连接哪个 LLM 模型服务来执行选表。"
        "留空时使用 DB-GPT 默认部署的客户端。"
        "一般无需填写，保持默认即可。"
    ),
)

_PARAMETER_MAX_SELECTED_TABLES = Parameter.build_from(
    _("Max Selected Tables"),
    "max_selected_tables",
    type=int,
    optional=True,
    default=5,
    description=_(
        "最大可选表数（可选，默认 5）：LLM 选出的表最多加载前几张的完整字段。"
        "控制喂给生成 SQL 的 LLM 的表结构 token 量。"
        "单表查询保持默认即可；涉及多表 join（如订单+用户+商品）时建议调到 8-10。"
    ),
)

_PARAMETER_CONTEXT_KEY = Parameter.build_from(
    _("Context Key"),
    "context_key",
    type=str,
    optional=True,
    default="context",
    alias=["context"],
    description=_(
        "上下文键（可选，默认 context，一般不要改）："
        "本算子输出的表结构文本在“键值对”里的键名，"
        "下游“大语言模型算子”渲染提示词时，提示词里的 {context} 占位符会按这个键名"
        "找到表结构并填入。只有当你把下游 LLM 提示词中的表结构占位符改名"
        "（例如改成 {table_info}）时，这里才需要改成同名，否则两边对不上，"
        "表结构不会进入提示词。"
    ),
)

_PARAMETER_ENUM_ENABLED = Parameter.build_from(
    _("Value Enumeration"),
    "enum_enabled",
    type=bool,
    optional=True,
    default=True,
    description=_(
        "值域枚举开关（可选，默认开启）："
        "开启后，语义层会对选表时 LLM 指定的 value_fields 字段查询真实取值（含频次），"
        "附在表结构后面，帮助 LLM 写对 WHERE 条件值"
        "（例如知道品类值实际是“咖啡机（配件）”而不是“咖啡机”）。设为 false 关闭。"
    ),
)

_PARAMETER_ENUM_LIMIT = Parameter.build_from(
    _("Value Enum Limit"),
    "enum_limit",
    type=int,
    optional=True,
    default=3,
    description=_(
        "值域枚举上限（可选，默认 3）："
        "每个字段按频次降序列出最常见的真实取值，只列前 N 个，"
        "不展示频次，避免撑大提示词。"
    ),
)

_INPUTS_QUESTION = IOField.build_from(
    _("User question"),
    "query",
    str,
    description=_(
        "用户问题输入：用户在对话页输入的自然语言查询，例如“每个用户的订单金额排行”。"
        "语义层会用它（结合全量表目录）判断要查询哪些表，并加载这些表的完整字段。"
        "在画布上由“通用大语言模型 HTTP 触发器”的“Request String Messages”输出口接入。"
    ),
)

_OUTPUTS_CONTEXT = IOField.build_from(
    _("Retrieved context"),
    "context",
    HOContextBody,
    description=_(
        "检索结果输出：包含数据库名/方言、LLM 根据用户问题选中的表及表间关联关系、"
        "以及选中表的完整表结构（字段/类型/主键/注释，不截断）的上下文对象。"
        "在画布上接到“大语言模型算子”的“extra_context”输入口，"
        "作为 LLM 生成 SQL 时的表结构参考。"
    ),
)


class HOSchemaLinkingRetrieverOperator(MixinLLMOperator, MapOperator[str, HOContextBody]):
    """语义层检索算子（Datasource Schema Linking Operator）。

    核心价值：替代官方"数据源检索算子"（向量摘要 top_k 截断、无关联关系），
    让 LLM 生成 SQL 前先经过"两阶段 Schema Linking"：
      阶段1：全量轻目录 -> LLM 选表 + 判断表间关联关系；
      阶段2：按选中表加载全部字段（不截断），过滤编造表名。

    典型接线（画布）：
      通用大语言模型 HTTP 触发器.out.1 (Request String Messages)
        -> 本算子.in.0 (User question)
      数据源资源 -> 本算子.参数.datasource
      本算子.out.0 (Retrieved context)
        -> 大语言模型算子.in.1 (extra_context)
    """

    metadata = ViewMetadata(
        label=_("Datasource Schema Linking Operator"),
        name="higher_order_datasource_schema_linking_operator",
        description=_(
            "语义层算子：先列出数据库全部表目录（表名+注释，注释可维护业务语义与表间关联关系），"
            "再让 LLM 根据用户问题判断要查询哪些表以及表与表之间的关联关系，"
            "最后对选中的表加载完整字段（不截断，含类型/主键/注释）。"
            "输出与官方“数据源检索算子”类型一致（HOContextBody），可在画布上直接替换接线。"
        ),
        category=OperatorCategory.DATABASE,
        parameters=[
            _PARAMETER_DATASOURCE.new(),
            _PARAMETER_PROMPT_TEMPLATE.new(),
            _PARAMETER_MODEL.new(),
            _PARAMETER_LLM_CLIENT.new(),
            _PARAMETER_MAX_SELECTED_TABLES.new(),
            _PARAMETER_CONTEXT_KEY.new(),
            _PARAMETER_ENUM_ENABLED.new(),
            _PARAMETER_ENUM_LIMIT.new(),
        ],
        inputs=[_INPUTS_QUESTION.new()],
        outputs=[_OUTPUTS_CONTEXT.new()],
        tags={"order": TAGS_ORDER_HIGH},
    )

    def __init__(
        self,
        datasource: DBResource,
        prompt_template: str = _DEFAULT_SELECT_TABLE_PROMPT,
        model: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        max_selected_tables: int = 5,
        context_key: Optional[str] = "context",
        enum_enabled: bool = True,
        enum_limit: int = 3,
        **kwargs,
    ):
        """初始化语义层检索算子。

        参数说明：
            datasource: 数据源资源（必填），用于读取表名/注释/字段
            prompt_template: 选表提示词模板，占位符支持 {db_name}/{table_catalog}/{max_selected_tables}，用户问题自动作为 user 消息传入
            model: 选表使用的模型名称，留空用默认模型
            llm_client: LLM 客户端，留空用系统默认
            max_selected_tables: 最多加载几张选中表的全字段
            context_key: 输出上下文对象的键名，默认 "context"，与下游提示词 {context} 对应
            enum_enabled: 是否对 LLM 指定的 value_fields 字段枚举真实取值（默认开启）
            enum_limit: 每个字段最多枚举多少个取值（默认 3，可调）
        """
        MapOperator.__init__(self, **kwargs)
        MixinLLMOperator.__init__(self, llm_client, save_model_output=False)
        self._datasource = datasource
        self._prompt_template = prompt_template
        self._model = model
        self._max_selected_tables = max_selected_tables
        self._context_key = context_key
        self._enum_enabled = enum_enabled
        self._enum_limit = enum_limit

    # ---------------- 1. 全量轻目录 ----------------
    async def _build_table_catalog(self) -> List[str]:
        """列出全部可用表（表名 + 表注释）。

        表注释是语义维护的载体：建议在注释中写明业务含义以及该表与其他表的关联关系，
        语义层每次全量读取，保证"目录 + 关系"完整无截断。
        """
        connector = self._datasource.connector
        table_names = await self.blocking_func_to_async(connector.get_table_names)
        table_names = list(table_names)
        catalog = []
        for table_name in table_names:
            comment = ""
            try:
                # 不同方言 get_table_comment 返回类型不一致，兼容 dict 和 str 两种
                comment_raw = await self.blocking_func_to_async(
                    connector.get_table_comment, table_name
                )
                if isinstance(comment_raw, dict):
                    comment = (comment_raw or {}).get("text") or ""
                else:
                    comment = str(comment_raw or "")
            except Exception as e:
                logger.warning(f"Get comment of table {table_name} failed: {e}")
            comment = str(comment).strip()
            if not comment:
                # Doris/StarRocks 等引擎 information_schema 不填充注释，改用 SHOW CREATE TABLE 解析
                comment = await self._get_comment_by_show_create(
                    connector, table_name
                )
            catalog.append(f"{table_name} -- {comment}" if comment else table_name)
        return catalog

    async def _get_comment_by_show_create(self, connector, table_name: str) -> str:
        """用 SHOW CREATE TABLE 解析表级 COMMENT，作为注释兜底。

        information_schema 读不到注释时（如 Doris/StarRocks），SHOW CREATE TABLE 会带出
        建表语句中的表级 COMMENT。列注释也是 COMMENT '...'，表级注释在语句末尾，
        因此取最后一个匹配项。
        """
        try:
            rows = await self.blocking_func_to_async(
                connector.run, f"SHOW CREATE TABLE {table_name}"
            )
            if rows and len(rows) > 1 and len(rows[1]) > 1:
                create_sql = str(rows[1][1])
                matches = re.findall(
                    r"COMMENT\s*(?:=\s*)?'((?:[^'\\\\]|\\\\.)*)'", create_sql
                )
                if matches:
                    return matches[-1].replace("''", "'").strip()
        except Exception as e:
            logger.warning(
                f"Get comment of table {table_name} by SHOW CREATE TABLE failed: {e}"
            )
        return ""

    async def _get_columns_with_fallback(
        self, connector, table_name: str
    ) -> List[dict]:
        """加载表字段；get_columns 失败/为空时用 SHOW CREATE TABLE 解析兜底。

        Doris 等引擎的 information_schema 元数据查询慢，偶发连接超时
        （Lost connection to MySQL server）或事务未回滚（Can't reconnect），
        导致 get_columns 返回空或抛异常，表结构字段为空、LLM 只能猜字段名。
        SHOW CREATE TABLE 走普通 SQL 执行（session_scope 自动提交/回滚），
        对 Doris 支持可靠，作为字段加载的兜底。
        """
        try:
            columns = await self.blocking_func_to_async(
                connector.get_columns, table_name
            )
            if columns:
                return columns
        except Exception as e:
            logger.warning(
                f"get_columns of table {table_name} failed, "
                f"try SHOW CREATE TABLE fallback: {e}"
            )
        try:
            rows = await self.blocking_func_to_async(
                connector.run, f"SHOW CREATE TABLE {table_name}"
            )
            if rows and len(rows) > 1 and len(rows[1]) > 1:
                create_sql = str(rows[1][1])
                return self._parse_columns_from_create_sql(create_sql)
        except Exception as e:
            logger.warning(
                f"Parse columns of table {table_name} by SHOW CREATE TABLE failed: {e}"
            )
        return []

    @staticmethod
    def _parse_columns_from_create_sql(create_sql: str) -> List[dict]:
        """从 SHOW CREATE TABLE 语句解析列定义（列名/类型/注释）。

        与 connector.get_columns 返回结构兼容：name/type/comment/is_in_primary_key。
        """
        start = create_sql.find("(")
        end = create_sql.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return []
        body = create_sql[start + 1 : end]
        columns = []
        for line in body.splitlines():
            line = line.strip().rstrip(",").strip()
            if not line:
                continue
            m = re.match(r"`([^`]+)`\s+", line)
            if not m:
                continue
            name = m.group(1)
            type_part = line[m.end() :].strip()
            # 去掉 NULL / NOT NULL / DEFAULT / AUTO_INCREMENT / COMMENT 等修饰子句
            type_part = re.split(
                r"\s+(?:NULL|COMMENT|DEFAULT|AUTO_INCREMENT)\b|;\s*$",
                type_part,
                maxsplit=1,
            )[0].strip()
            comment = ""
            cm = re.search(
                r"COMMENT\s+[\"']([^\"']*)[\"']", line, re.IGNORECASE
            )
            if cm:
                comment = cm.group(1)
            columns.append(
                {
                    "name": name,
                    "type": type_part,
                    "default_expression": "",
                    "is_in_primary_key": False,
                    "comment": comment,
                }
            )
        return columns

    # ---------------- 2. LLM 选表 + 关联 ----------------
    async def _select_tables(
        self, question: str, catalog: List[str], keywords_text: str = ""
    ) -> List[dict]:
        """调用 LLM 从目录中选择相关表并判断关联关系。

        keywords_text: 从问题提取的关键实体词文本，附加到 user 消息辅助选表。
        """
        catalog_str = "\n".join(
            f"{i + 1}. {item}" for i, item in enumerate(catalog)
        )
        try:
            prompt = self._prompt_template.format(
                db_name=self._datasource._db_name,
                table_catalog=catalog_str,
                max_selected_tables=self._max_selected_tables,
            )
        except Exception as e:
            logger.warning(
                f"Select table prompt template format failed ({e}), "
                f"fallback to all tables"
            )
            return self._fallback_all_tables(catalog)
        try:
            # system 放角色/目录/要求，user 放用户问题（DB-GPT 枚举用 HUMAN 表示用户消息）
            user_content = f"用户问题：{question}"
            if keywords_text:
                user_content += (
                    f"\n\n用户问题中的关键实体词（选表请覆盖这些实体所在的表）：\n{keywords_text}"
                )
            output = await self._llm_complete(prompt, user_content)
            return self._parse_table_selection(output)
        except Exception as e:
            logger.warning(f"LLM select tables failed, fallback to all tables: {e}")
            return self._fallback_all_tables(catalog)

    @staticmethod
    def _fallback_all_tables(catalog: List[str]) -> List[dict]:
        """兜底：返回全部表（只取纯表名，不带注释串），保证不中断流程。"""
        return [
            {"table": t.split(" -- ", 1)[0], "relation": ""} for t in catalog
        ]

    async def _build_model_request(self, messages: List[ModelMessage]) -> ModelRequest:
        models = await self.llm_client.models()
        if not models:
            raise ValueError("No models available.")
        model = self._model or models[0].model
        return ModelRequest.build_request(model, messages=messages)

    async def _llm_complete(self, prompt: str, user_content: str) -> str:
        """执行一次 LLM 补全（system=提示词, user=问题），返回输出文本。"""
        messages = [
            ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt),
            ModelMessage(role=ModelMessageRoleType.HUMAN, content=user_content),
        ]
        model_request = await self._build_model_request(messages)
        model_output: ModelOutput = await self.llm_client.generate(model_request)
        return model_output.text

    def _parse_table_selection(self, text: str) -> List[dict]:
        """解析 LLM 输出的 JSON，提取选中的表及关联关系。"""
        data = self._parse_json_strict(text)
        tables = data.get("tables") or []
        result = []
        for item in tables:
            if isinstance(item, dict) and item.get("table"):
                result.append(
                    {
                        "table": str(item["table"]),
                        "relation": str(item.get("relation") or ""),
                    }
                )
        return result

    @staticmethod
    def _parse_json_strict(text: str) -> dict:
        """容错解析 LLM 输出的 JSON：优先整体解析，失败则截取首个 { 到最后一个 } 再解析。

        兼容 LLM 常见输出问题：中文引号“”、中文冒号：等。
        """
        text = text.strip()
        text = (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("：", ":")
        )
        # 兼容 LLM 把提示词示例的双花括号照抄进输出（{{ } } 成对出现时才还原）
        if "{{" in text:
            text = text.replace("{{", "{").replace("}}", "}")
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Can not parse LLM output: {text}")
            return json.loads(text[start : end + 1])

    @staticmethod
    def _build_catalog_comments(catalog: List[str]) -> dict:
        """从目录项（"表名 -- 注释"）构建 {表名: 注释} 映射。"""
        result = {}
        for item in catalog:
            parts = item.split(" -- ", 1)
            result[parts[0]] = parts[1] if len(parts) > 1 else ""
        return result

    # ---------------- 3. 按选中表加载全字段 ----------------
    @staticmethod
    def _extract_business_comment(comment: str) -> str:
        """从表注释中提取业务含义部分，关联关系部分不展示。

        约定表注释格式："业务含义；关联关系"。业务含义（第一个分号之前）展示在
        表结构中；关联关系（分号之后）仅供选表 LLM 在目录中读取，不在这里重复出现。
        """
        for sep in ("；", ";"):
            if sep in comment:
                return comment.split(sep, 1)[0].strip()
        return comment

    @staticmethod
    def _format_columns(columns: List[dict], with_pk: bool = False) -> List[str]:
        """把字段列表格式化为 "    字段 类型 [PK] -- 注释" 行。"""
        lines = []
        for col in columns:
            name = col.get("name", "")
            col_type = col.get("type", "")
            pk = " [PK]" if with_pk and col.get("is_in_primary_key") else ""
            comment = str(col.get("comment") or "").strip()
            suffix = f" -- {comment}" if comment else ""
            lines.append(f"    {name} {col_type}{pk}{suffix}")
        return lines

    async def _build_schemas_text(
        self,
        table_names: List[str],
        catalog_comments: dict,
        columns_by_table: dict,
        value_fields_map: dict,
    ) -> str:
        """按选中表拼装完整表结构文本（字段/类型/主键/注释 + 值域枚举）。

        表注释只展示业务含义（第一个分号前），分号后的关联关系不重复展示，
        关联关系已由"根据用户问题选择的表及表间关联关系"节给出。
        """
        connector = self._datasource.connector
        schemas = []
        for table_name in table_names:
            columns = columns_by_table.get(table_name, [])
            col_lines = self._format_columns(columns, with_pk=True)
            table_comment = catalog_comments.get(table_name, "")
            display_comment = self._extract_business_comment(table_comment)
            if display_comment:
                schema = (
                    f"{table_name} -- {display_comment}\n("
                    + "\n".join(col_lines)
                    + "\n)"
                )
            else:
                schema = f"{table_name}(\n" + "\n".join(col_lines) + "\n)"
            # 值域枚举：对 LLM 确认的 value_fields 字段一次查询多字段，
            # 取最常见的 N 行真实取值组合（每行各值横向对应字段名顺序）；
            # 帮助 LLM 写对 WHERE 条件值（如品类值实际是"咖啡机（配件）"而非"咖啡机"）
            if self._enum_enabled:
                vf_list = value_fields_map.get(table_name, [])
                if vf_list:
                    rows = await self._enumerate_field_values(
                        connector, table_name, vf_list, self._enum_limit
                    )
                    if rows:
                        header = (
                            "    "
                            + "，".join(vf_list)
                            + f" 取值（真实数据，前{len(rows)}行）:"
                        )
                        value_lines = [
                            "    " + ", ".join(r) for r in rows
                        ]
                        schema += (
                            "\n\n 字段取值参考（真实数据）:\n"
                            + header
                            + "\n"
                            + "\n".join(value_lines)
                        )
            schemas.append(schema)
        return "\n\n".join(schemas)

    async def _select_value_fields(
        self, question: str, columns_by_table: dict
    ) -> dict:
        """字段确认步骤：LLM 看到选中表完整字段后，指定需要参考真实取值的字段。

        返回 {表名: [字段, ...]}；未开启值域枚举或 LLM 调用失败时返回空 dict。
        """
        if not self._enum_enabled:
            return {}
        schema_lines = []
        for table_name, columns in columns_by_table.items():
            if not columns:
                continue
            col_lines = self._format_columns(columns)
            schema_lines.append(f"{table_name}(\n" + "\n".join(col_lines) + "\n)")
        if not schema_lines:
            return {}
        try:
            prompt = _DEFAULT_SELECT_VALUE_FIELDS_PROMPT.format(
                schemas="\n\n".join(schema_lines)
            )
            output = await self._llm_complete(prompt, f"用户问题：{question}")
            return self._parse_fields_map(output, "value_fields")
        except Exception as e:
            logger.warning(f"LLM select value fields failed, skip: {e}")
            return {}

    @staticmethod
    def _parse_fields_map(text: str, key: str) -> dict:
        """解析 {key: {表名: [字段]}} 形式的 JSON，过滤非列表/空字段，每表最多 5 个。"""
        data = HOSchemaLinkingRetrieverOperator._parse_json_strict(text)
        result = {}
        for table, fields in (data.get(key) or {}).items():
            if not isinstance(fields, list):
                continue
            cleaned = [f for f in fields if isinstance(f, str) and f.strip()][:5]
            if cleaned:
                result[str(table)] = cleaned
        return result

    @staticmethod
    def _filter_value_fields(value_fields_map: dict, columns_by_table: dict) -> dict:
        """过滤不存在的字段名，防止 LLM 编造字段污染 SELECT（如 *、拼错的列名）。

        只保留该表真实存在的列；过滤后某表没有合法字段则整表移除。
        """
        result = {}
        for table_name, fields in value_fields_map.items():
            real_names = {c.get("name") for c in columns_by_table.get(table_name, [])}
            cleaned = [f for f in fields if f in real_names]
            if cleaned:
                result[table_name] = cleaned
        return result

    # ---------------- 4.5 二次兜底：选值域字段校验 + 补充字段 ----------------
    async def _verify_and_fix_value_fields(
        self,
        question: str,
        columns_by_table: dict,
        value_fields_map: dict,
    ) -> dict:
        """二次兜底：查已选值域字段的真实取值，LLM 判断是否足够，不足则补字段。

        对标 _verify_and_fix_tables：先对已选字段查真实取值（limit 3），连同
        用户问题交给 LLM，判断取值是否足以写对 WHERE（如问题说"电动风扇"但取值
        里只有"风扇/塔式风扇"，就需要补品类/标题类字段确认真实叫法）。
        需要时从该表真实字段中补充，去重后合并进 value_fields_map。
        LLM 调用或解析失败时原样返回，不阻塞主流程。
        """
        if not self._enum_enabled or not value_fields_map:
            return value_fields_map
        connector = self._datasource.connector
        field_lines = []
        for table_name, fields in value_fields_map.items():
            if not fields or not columns_by_table.get(table_name):
                continue
            rows = await self._enumerate_field_values(connector, table_name, fields, 3)
            if not rows:
                value_desc = "（未查到取值）"
            else:
                value_desc = "\n".join("  " + ", ".join(r) for r in rows)
            field_lines.append(
                f"{table_name} 字段 [{', '.join(fields)}] 真实取值:\n{value_desc}"
            )
        if not field_lines:
            return value_fields_map
        try:
            prompt = _DEFAULT_VERIFY_VALUE_FIELDS_PROMPT.format(
                field_values="\n\n".join(field_lines),
                question=question,
            )
            output = await self._llm_complete(prompt, f"用户问题：{question}")
            add_map = self._parse_fields_map(output, "add_fields")
        except Exception as e:
            logger.warning(f"Verify value fields failed, keep original fields: {e}")
            return value_fields_map
        if not add_map:
            return value_fields_map
        # 补充字段只接受真实存在的列名；与已选合并去重，每表最多 5 个
        result = {k: list(v) for k, v in value_fields_map.items()}
        for table_name, new_fields in add_map.items():
            real_names = {
                c.get("name") for c in columns_by_table.get(table_name, [])
            }
            merged = list(result.get(table_name, []))
            for f in new_fields:
                if f and f in real_names and f not in merged:
                    merged.append(f)
            result[table_name] = merged[:5]
        return result

    # ---------------- 3.5 二次兜底：选表校验 + 补充缺失表 ----------------
    def _parse_verify_result(self, text: str) -> Tuple[bool, List[str]]:
        """解析校验结果 JSON，返回 (是否足够, 缺失表名列表)。"""
        data = self._parse_json_strict(text)
        sufficient = bool(data.get("sufficient", True))
        missing = []
        for item in data.get("missing_tables") or []:
            if isinstance(item, dict) and item.get("table"):
                missing.append(str(item["table"]))
        return sufficient, missing

    async def _verify_and_fix_tables(
        self,
        question: str,
        catalog: List[str],
        table_names: List[str],
        columns_by_table: dict,
    ) -> List[str]:
        """二次兜底：校验已选表是否足以回答用户问题，不足则从目录补充缺失表。

        针对"漏选维度表"这类问题：例如用户问"咖啡机Q1总销量"，销售表只有
        category_id 外键、没有品类名称字段，校验会让 LLM 补充品类维度表。
        只做表级补充，不修改已选表的字段；LLM 校验失败或解析失败时
        静默保留原选表结果，不中断主流程。
        """
        if not table_names:
            return table_names
        catalog_comments = self._build_catalog_comments(catalog)
        schema_lines = []
        for t in table_names:
            cols = columns_by_table.get(t, [])
            col_names = ", ".join(c.get("name", "") for c in cols)
            comment = catalog_comments.get(t, "")
            schema_lines.append(f"{t} -- {comment}（字段：{col_names}）")
        try:
            prompt = _DEFAULT_VERIFY_TABLES_PROMPT.format(
                selected_tables="\n".join(schema_lines),
                table_catalog="\n".join(f"- {c}" for c in catalog),
            )
            output = await self._llm_complete(prompt, f"用户问题：{question}")
            sufficient, missing = self._parse_verify_result(output)
        except Exception as e:
            logger.warning(f"LLM verify tables failed, keep original selection: {e}")
            return table_names
        if sufficient or not missing:
            return table_names
        catalog_tables = {item.split(" -- ")[0] for item in catalog}
        result = list(table_names)
        for name in missing:
            if name not in catalog_tables or name in result:
                continue
            result.append(name)
            if len(result) >= self._max_selected_tables:
                break
        return result

    # ---------------- 追问补全：结合历史对话，把指代问题改写为自包含完整问题 ----------------

    @staticmethod
    def _strip_schema_linking_text(content: str) -> str:
        """去掉语义层注入到消息里的表结构文本，只保留真正的用户问题/回答内容。"""
        if not content:
            return ""
        idx = content.find(_SCHEMA_LINKING_INJECT_MARKER)
        if idx < 0:
            return content
        q_idx = content.find(_SCHEMA_LINKING_QUESTION_MARKER, idx)
        if q_idx < 0:
            return ""
        return content[q_idx + len(_SCHEMA_LINKING_QUESTION_MARKER) :].strip()

    @staticmethod
    def _extract_conclusion_text(content: str) -> str:
        """从回复内容提取最终结论文本。

        DataScientist 的回复 content 可能是 chart_action 的 action_report JSON
        （含 display_type/sql/thought），此时只保留 thought 摘要：sql 会污染
        下一轮的改写判断（可能照抄旧 SQL）且浪费 token，display_type 对理解
        口径无帮助。非 JSON 的普通文本（如校验失败反馈）原样返回。
        """
        if not content:
            return ""
        text = content.strip()
        try:
            obj = json.loads(text)
        except Exception:
            return text
        if isinstance(obj, dict):
            thought = obj.get("thought")
            if isinstance(thought, str) and thought.strip():
                return thought.strip()
            # 纯 action_report 结构（无 thought 摘要）不保留，避免带出 sql 噪音
            return ""
        return text

    @staticmethod
    def _format_history_item(role_label: str, content: str) -> str:
        """格式化一条历史消息：去掉注入文本、压缩空白、超长截断。"""
        content = HOSchemaLinkingRetrieverOperator._strip_schema_linking_text(content)
        content = " ".join(content.split())
        if not content:
            return ""
        if len(content) > _REWRITE_HISTORY_MAX_CHARS:
            content = content[:_REWRITE_HISTORY_MAX_CHARS] + "..."
        return f"{role_label}: {content}"

    async def _build_history_text_from_db(
        self, input_value: AgentGenerateContext
    ) -> str:
        """从数据库 gpts_message 表读取同一会话的最近几轮（与 Agent 记忆同源）。

        每轮只取头尾，中间全部丢弃：
          - 头 = 该轮第一条 Human 发送的消息（用户问题，必留）
          - 尾 = 该轮最后一条非 Human 发送的消息（最终回复，按自增 id 定位）
        is_success 判成败：尾消息 is_success=1 才附结论；is_success=0 只留用户问题。
        重试过程中的纠错反馈、未过校验的中间回复等一律不进历史。

        分轮方式：每一轮对话的 agent_conv_id 带递增后缀（如 abc_1、abc_2），
        语义层执行时当前轮消息尚未入库，所以用真实 conv_id 做 LIKE 匹配
        （get_by_agent）捞取整段会话，再按完整 conv_id 分组即得每一轮；
        DAO 已按自增 id 升序返回（id 是真实写入顺序，rounds 每轮从 0 重计且
        重试会回跳，不能用于排序）。查询失败时静默返回空，由上层回退原问题。
        """
        try:
            agent_context = getattr(input_value, "agent_context", None)
            conv_id = getattr(agent_context, "conv_id", None)
            if not conv_id:
                return ""
            from dbgpt.agent.util.conv_utils import parse_conv_id
            from dbgpt_serve.agent.db.gpts_messages_db import GptsMessagesDao

            real_conv_id, _ = parse_conv_id(conv_id)
            # 标准 DB-GPT 对话中，UserProxyAgent 的 role 为 "Human"，
            # 会话内每条消息的 sender 或 receiver 必有一方是 "Human"，一次查询即可覆盖整段对话
            message_dao = GptsMessagesDao()
            results = await self.blocking_func_to_async(
                message_dao.get_by_agent, real_conv_id, _REWRITE_HISTORY_ROLE
            )
            if not results:
                return ""
            # 按完整 conv_id（含后缀）分轮
            groups: dict = {}
            for m in results:
                groups.setdefault(m.conv_id, []).append(m)
            # 按每轮最大 id 升序排列轮次（id 是真实写入顺序），取最近几轮
            round_list = sorted(groups.values(), key=lambda ms: max(x.id for x in ms))
            history = []
            for ms in round_list[-_REWRITE_HISTORY_MAX_ROUNDS:]:
                # 头：该轮第一条 Human 发送的消息（用户问题，必留）
                head = next(
                    (x for x in ms if (x.sender or "") == _REWRITE_HISTORY_ROLE),
                    None,
                )
                if head is None:
                    continue
                head_text = self._format_history_item("用户", head.content)
                if head_text:
                    history.append(head_text)
                # 尾：该轮 id 最大的非 Human 消息（最终回复）
                tail = next(
                    (
                        x
                        for x in reversed(ms)
                        if (x.sender or "") != _REWRITE_HISTORY_ROLE
                    ),
                    None,
                )
                # 判定轮次成败：尾消息 is_success=1 才附最终结论，否则只留用户问题
                if tail is not None and getattr(tail, "is_success", False):
                    # content 若是 action_report JSON，只提取 thought 摘要，避免带出 sql
                    tail_content = self._extract_conclusion_text(tail.content)
                    tail_text = self._format_history_item("助手", tail_content)
                    if tail_text:
                        history.append(tail_text)
            return "\n".join(history)
        except Exception as e:
            logger.warning(f"Build history from db failed, skip: {e}")
            return ""

    def _parse_rewrite_result(self, text: str) -> Tuple[bool, str]:
        """解析改写结果 JSON，返回 (是否引用了前文, 改写后的问题)。"""
        data = self._parse_json_strict(text)
        related = bool(data.get("related", False))
        rewritten = str(data.get("rewritten_question") or "").strip()
        return related, rewritten

    @staticmethod
    def _parse_keywords_result(text: str) -> List[str]:
        """容错解析关键词提取结果，返回去重后的实体词列表。"""
        try:
            data = HOSchemaLinkingRetrieverOperator._parse_json_strict(text)
            kws = data.get("关键词列表")
        except Exception:
            kws = None
        if not isinstance(kws, list):
            return []
        seen, out = set(), []
        for k in kws:
            s = str(k or "").strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    async def _extract_keywords(self, question: str) -> List[str]:
        """关键词提取：从改写后的完整问题中提取需要语义检索/选表参考的实体词。

        只输出实体词列表（不判断目标字段）；后续可把结果交给向量检索方
        （返回 {向量子查询, 目标字段} 拼进分析 SQL）。失败或为空时返回
        空列表，不阻塞主流程。
        """
        try:
            output = await self._llm_complete(
                _DEFAULT_EXTRACT_KEYWORDS_PROMPT, f"用户问题：{question}"
            )
            return self._parse_keywords_result(output)
        except Exception as e:
            logger.warning(f"Extract keywords failed, use empty list: {e}")
            return []

    async def _rewrite_followup_question(
        self, question: str, input_value: AgentGenerateContext
    ) -> str:
        """追问补全：判断当前问题是否指代前文，若是则改写为自包含的完整问题。

        历史来源：数据库 gpts_message 表（通过 agent_context.conv_id 查询，与
        DataScientist 的会话记忆同源）。不读 rely_messages：AWEL Agent 流程中
        语义层位于 Agent Trigger 之后、AWEL Agent Operator 之前，起始 context
        未携带 rely_messages（实测恒为空），且 gpts_message 表是 rely 的严格超集。
        历史不可用、或 LLM 判定/改写失败时，静默返回原问题，不阻塞主流程。
        """
        current = self._strip_schema_linking_text(question).strip()
        history = await self._build_history_text_from_db(input_value)
        if not history:
            return question
        try:
            prompt = _DEFAULT_REWRITE_QUESTION_PROMPT.format(history=history)
            output = await self._llm_complete(
                prompt, f"当前用户问题：{question}"
            )
            related, rewritten = self._parse_rewrite_result(output)
        except Exception as e:
            logger.warning(
                f"Rewrite followup question failed, keep original question: {e}"
            )
            return question
        if (
            not related
            or not rewritten
            or rewritten == current
            or rewritten == question
        ):
            return question
        return rewritten

    async def _enumerate_field_values(
        self, connector, table_name: str, fields: List[str], limit: int
    ) -> List[List[str]]:
        """查询多个字段的前 N 行真实数据，供 LLM 写 WHERE 条件时参考。

        直接取表的前 limit 行原始记录（不做聚合、不按频次排序），
        返回行内各值顺序与 fields 一致，形如 [["其他配件", "生活家居配件", ...], ...]。
        空值显示为空字符串；查询失败（字段不存在、表不可读等）时静默返回空列表，不影响主流程。
        """
        fields_sql = ", ".join(fields)
        try:
            rows = await self.blocking_func_to_async(
                connector.run,
                f"SELECT {fields_sql} FROM {table_name} LIMIT {limit}",
            )
        except Exception as e:
            logger.warning(
                f"Enumerate values of {table_name}.{fields} failed: {e}"
            )
            return []
        result = []
        for row in rows:
            if isinstance(row, dict):
                row_values = [str(row.get(f) or "") for f in fields]
            else:
                # list/tuple 或 SQLAlchemy Row 对象（均支持索引访问）
                try:
                    first = row[0]
                except (IndexError, TypeError, KeyError):
                    continue
                # RDBMSConnector._query 会把列名作为首行插入（即 fields 列名），跳过该表头行
                if first in fields:
                    continue
                row_values = []
                for i in range(len(fields)):
                    try:
                        v = row[i]
                    except (IndexError, TypeError, KeyError):
                        v = None
                    if v is None:
                        row_values.append("")
                        continue
                    text = str(v)
                    if len(text) > 20:
                        text = text[:20] + "..."
                    row_values.append(text)
            result.append(row_values)
        return result

    # ---------------- 主流程 ----------------
    async def _run_schema_linking(
        self,
        question: str,
        keywords_text: str = "",
        recalled_table_names: Optional[List[str]] = None,
    ) -> dict:
        """执行语义层流程（目录 -> 选表+关联 -> 加载字段 -> 字段确认 -> 全字段+值域枚举）。

        供普通检索算子（map）与 Agent 版算子（map）共用。
        keywords_text: 从问题提取的关键实体词文本，附加给选表 LLM 参考。
        recalled_table_names: 从召回 SQL 注释中识别的来源表，加入允许表集合。
        """
        db_name = self._datasource._db_name
        dialect = self._datasource.dialect

        # 1. 全量轻目录（表名 + 注释，注释含业务语义与关联关系）
        catalog = await self._build_table_catalog()

        # 2. LLM 选表 + 关联关系，并合并向量召回 SQL 注释中的来源表。
        selected = await self._select_tables(question, catalog, keywords_text)
        catalog_tables = {item.split(" -- ")[0] for item in catalog}
        selected_names = {item["table"] for item in selected}
        for table_name in recalled_table_names or []:
            if table_name in catalog_tables and table_name not in selected_names:
                relation = (
                    "向量实体定位表；通过 table_name、column_name 和 raw_data 标识并关联来源业务表"
                    if table_name == TARGET_TABLE
                    else "向量召回命中的来源业务表"
                )
                selected.append({"table": table_name, "relation": relation})
                selected_names.add(table_name)
        if selected:
            selected_text = "\n".join(
                f"- {s['table']}"
                + (f" -- 关联: {s['relation']}" if s["relation"] else "")
                for s in selected
            )
        else:
            selected_text = "- (LLM 未返回有效选表结果)"

        # 3. 过滤选中表（防止 LLM 编造表名）并加载完整字段
        connector = self._datasource.connector
        seen = set()
        table_names = []
        for item in selected:
            name = item["table"]
            if name in seen or name not in catalog_tables:
                continue
            seen.add(name)
            table_names.append(name)
        if not table_names:
            logger.warning("No valid selected tables, fallback to all tables")
            table_names = list(catalog_tables)[: self._max_selected_tables]

        columns_by_table = {}
        for t in table_names:
            columns_by_table[t] = await self._get_columns_with_fallback(connector, t)

        # 3.5 二次兜底：校验已选表是否足以回答用户问题，不足则从目录补充缺失表
        # （例如"咖啡机Q1总销量"：销售表只有 category_id 外键、无品类名称字段，
        #  校验会让 LLM 补充品类维度表，避免漏选维度表导致过滤条件错误）
        fixed_table_names = await self._verify_and_fix_tables(
            question, catalog, table_names, columns_by_table
        )
        added_tables = [t for t in fixed_table_names if t not in table_names]
        if added_tables:
            table_names = fixed_table_names
            # 补充的表也加入选中列表，保证"选中的表及关联关系"与表结构一致
            selected.extend(
                {"table": t, "relation": "二次校验补充"} for t in added_tables
            )
            for t in added_tables:
                columns_by_table[t] = await self._get_columns_with_fallback(
                    connector, t
                )

        # 4. 字段确认：LLM 看到选中表完整字段后，指定需要参考真实取值的字段
        value_fields_map = await self._select_value_fields(question, columns_by_table)
        # 只保留真实存在的列名，防止 LLM 编造字段污染值域枚举（如 SELECT *、拼错的列名）
        value_fields_map = self._filter_value_fields(value_fields_map, columns_by_table)

        # 4.5 二次兜底：查已选值域字段的真实取值，LLM 判断是否足够，不足则补字段
        value_fields_map = await self._verify_and_fix_value_fields(
            question, columns_by_table, value_fields_map
        )

        # 5. 拼装表结构（含字段取值参考）
        catalog_comments = self._build_catalog_comments(catalog)
        schemas_text = await self._build_schemas_text(
            table_names, catalog_comments, columns_by_table, value_fields_map
        )
        return {
            "db_name": db_name,
            "dialect": dialect,
            "selected_text": selected_text,
            "schemas_text": schemas_text,
            "table_names": table_names,
            "columns_by_table": columns_by_table,
            "value_fields_map": value_fields_map,
        }

    async def map(self, question: str) -> HOContextBody:
        """执行语义层检索。"""
        result = await self._run_schema_linking(question)
        context = (
            f"数据库名: {result['db_name']}\n"
            f"方言: {result['dialect']}\n\n"
            f"根据用户问题选择的表及表间关联关系:\n{result['selected_text']}\n\n"
            f"选中表的完整表结构（字段、类型、主键、注释）:\n{result['schemas_text']}\n\n"
            f"用户问题:\n{question}"
        )
        return HOContextBody(context_key=self._context_key, context=context)


class HOSchemaLinkingAgentOperator(HOSchemaLinkingRetrieverOperator):
    """Agent 语义层算子（Agent Schema Linking Operator）。

    用于在 AWEL Agent 流程中串联语义层：
      1. 接收上游 Agent 触发器输出的 AgentGenerateContext，取出其中的用户问题；
      2. 追问补全：结合会话历史（数据库 gpts_message 表，与 Agent 记忆同源），
         若当前问题指代前文（如"那环比是多少了"），
         改写为自包含的完整问题，避免漏选维度表、口径错乱；
      3. 执行两阶段语义层（全表目录 -> LLM 选表+关联 -> 按选中表加载全字段）；
      4. 把选好的表结构与用户问题一并写入消息内容（作为 user 消息），
         让下游 Agent（如 DataScientist）严格按语义层选出的表生成 SQL；
      5. 原样透传 AgentGenerateContext 给下游 AWEL Agent Operator。

    典型接线（画布）：
      Agent Trigger.out.0 (Agent Operator Context)
        -> 本算子.in.0 (Agent Operator Context)
      数据源资源 -> 本算子.参数.datasource
      本算子.out.0 (Agent Operator Context)
        -> AWEL Agent Operator.in.0（Agent 选择 DataScientist）
    """

    metadata = ViewMetadata(
        label=_("Agent Schema Linking Operator"),
        name="agent_schema_linking_operator",
        description=_(
            "在 Agent 流程中执行语义层：接收上游 Agent 触发器输出的 AgentGenerateContext，"
            "取其中的用户问题，执行“全表目录 -> LLM 选表+关联 -> 加载选中表全字段”，"
            "把选好的表结构与用户问题写入消息内容（user 消息），"
            "让下游 Agent 算子（如 DataScientist）严格按语义层选出的表生成 SQL，"
            "然后原样透传 AgentGenerateContext。"
        ),
        category=OperatorCategory.AGENT,
        parameters=[
            _PARAMETER_DATASOURCE.new(),
            _PARAMETER_PROMPT_TEMPLATE.new(),
            _PARAMETER_MODEL.new(),
            _PARAMETER_LLM_CLIENT.new(),
            _PARAMETER_MAX_SELECTED_TABLES.new(),
            _PARAMETER_ENUM_ENABLED.new(),
            _PARAMETER_ENUM_LIMIT.new(),
        ],
        inputs=[
            IOField.build_from(
                _("Agent Operator Context"),
                "agent_operator_context",
                AgentGenerateContext,
                description=_(
                    "上游 Agent 触发器输出的 AgentGenerateContext，"
                    "本算子读取其中的用户问题并执行语义层选表，之后原样透传。"
                ),
            )
        ],
        outputs=[
            IOField.build_from(
                _("Agent Operator Context"),
                "agent_operator_context",
                AgentGenerateContext,
                description=_(
                    "透传的 AgentGenerateContext（消息内容已注入语义层选好的表结构与关联关系），"
                    "接到 AWEL Agent Operator（DataScientist）的输入。"
                ),
            )
        ],
        tags={"order": TAGS_ORDER_HIGH},
    )

    async def _split_keywords_for_vector_recall(
        self,
        keywords: List[str],
        table_names: List[str],
        columns_by_table: dict,
        value_fields_map: dict,
    ) -> Tuple[Dict[str, List[dict]], List[str]]:
        """在选中的原始业务表中模糊匹配关键词，并返回命中的表、字段和值。"""
        connector = self._datasource.connector
        searchable_fields = {}
        for table_name in table_names:
            if table_name == "dim_product_category":
                real_fields = {
                    column.get("name")
                    for column in columns_by_table.get(table_name, [])
                }
                fields = [
                    field
                    for field in (
                        "category_1",
                        "category_2",
                        "category_3",
                        "category_4",
                    )
                    if field in real_fields
                ]
            else:
                fields = value_fields_map.get(table_name) or [
                    column.get("name")
                    for column in columns_by_table.get(table_name, [])
                    if any(
                        marker in str(column.get("name") or "").lower()
                        for marker in (
                            "category",
                            "product",
                            "brand",
                            "type",
                            "name",
                            "model",
                        )
                    )
                ]
            searchable_fields[table_name] = [field for field in fields if field]

        matched_evidence: Dict[str, List[dict]] = {}
        unmatched = []
        for keyword in keywords:
            escaped_keyword = (
                keyword.replace("\\", "\\\\")
                .replace("'", "''")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            keyword_evidence = []
            for table_name, fields in searchable_fields.items():
                for field in fields:
                    escaped_table = table_name.replace("`", "``")
                    escaped_field = field.replace("`", "``")
                    sql = (
                        f"SELECT CAST(`{escaped_field}` AS STRING) AS matched_value "
                        f"FROM `{escaped_table}` "
                        f"WHERE CAST(`{escaped_field}` AS STRING) "
                        f"LIKE '%{escaped_keyword}%' ESCAPE '\\\\' LIMIT 1"
                    )
                    try:
                        rows = await self.blocking_func_to_async(connector.run, sql)
                        # 第 1 行是列名，第 2 行才是首条实际命中值。
                        if rows and len(rows) > 1:
                            row = rows[1]
                            matched_value = row[0] if row else ""
                            keyword_evidence.append(
                                {
                                    "table_name": table_name,
                                    "column_name": field,
                                    "matched_value": str(matched_value or ""),
                                }
                            )
                    except Exception as e:
                        logger.warning(
                            "LIKE check failed for %s.%s: %s", table_name, field, e
                        )
            if keyword_evidence:
                matched_evidence[keyword] = keyword_evidence
            else:
                unmatched.append(keyword)
        return matched_evidence, unmatched

    async def map(self, input_value: AgentGenerateContext) -> AgentGenerateContext:
        """执行语义层并把选表结果注入用户消息，然后原样透传。"""
        if not input_value.message:
            raise ValueError("The message is empty.")
        question = input_value.message.content or ""
        # 追问补全：结合历史对话，把"那环比是多少了"这类指代问题改写为
        # "咖啡机Q1销量环比上季度"的完整问题，避免漏选品类表、口径错乱
        question = await self._rewrite_followup_question(question, input_value)
        # 关键词提取后执行向量召回；召回模块只返回带来源注释的 SQL。
        keywords = await self._extract_keywords(question)
        keywords_text = "、".join(keywords) if keywords else ""
        logger.info(f"schema_linking extracted keywords: {keywords}")

        # 先保持原 Schema Linking 流程完成选表，再查询这些原始业务表的真实字段。
        result = await self._run_schema_linking(
            question,
            keywords_text=keywords_text,
        )

        recall_sql_list = []
        matched_evidence: Dict[str, List[dict]] = {}
        recall_keywords = []
        if keywords:
            try:
                matched_evidence, recall_keywords = (
                    await self._split_keywords_for_vector_recall(
                        keywords,
                        result["table_names"],
                        result["columns_by_table"],
                        result["value_fields_map"],
                    )
                )
                logger.info(
                    "schema_linking source-table LIKE matched evidence: %s, "
                    "LIKE unmatched keywords: %s; all keywords use vector recall",
                    matched_evidence,
                    recall_keywords,
                )
                recall_results = await self.blocking_func_to_async(
                    recall_many, keywords, DEFAULT_MAX_DISTANCE
                )
                for recall_result in recall_results:
                    recall_sql_list.extend(recall_result.get("sql", []))
            except Exception as e:
                logger.warning(f"Keyword LIKE check or vector recall failed, skip: {e}")

        # SQL 首行注释格式为“# 关键词:... 来自表 ...，主键 ...”，从中提取来源表。
        recalled_tables = []
        for recall_sql in recall_sql_list:
            match = re.search(r"来自表\s+([^，\s]+)", recall_sql)
            if match and match.group(1) not in recalled_tables:
                recalled_tables.append(match.group(1))

        # 向量 SQL 会直接引用向量表；有召回 SQL 时，将向量表及命中的来源业务表
        # 一并加入允许表集合并加载完整结构。
        if recall_sql_list and TARGET_TABLE not in recalled_tables:
            recalled_tables.append(TARGET_TABLE)
        if recalled_tables:
            result = await self._run_schema_linking(
                question,
                keywords_text=keywords_text,
                recalled_table_names=recalled_tables,
            )
        keyword_filter_context = ""
        if matched_evidence:
            evidence_lines = []
            for keyword, evidence_items in matched_evidence.items():
                evidence_lines.append(f"- 关键词：{keyword}")
                for item in evidence_items:
                    evidence_lines.append(
                        "  - 表：{table_name}；字段：{column_name}；命中值：{matched_value}".format(
                            **item
                        )
                    )
            keyword_filter_context += (
                "\n\n原始业务表中的普通模糊匹配证据（LIKE '%关键词%'）：\n"
                + "\n".join(evidence_lines)
            )
        if recall_keywords:
            keyword_filter_context += (
                "\n\n原始业务表中未直接模糊匹配的关键词：\n"
                + "、".join(recall_keywords)
            )
        if recall_sql_list:
            keyword_filter_context += (
                "\n\n所有关键词对应的完整向量召回 SQL：\n"
                + "\n\n".join(recall_sql_list)
            )

        # 允许表 = Schema Linking 选中的表 + 召回 SQL 注释中的来源表。
        input_value.message.content = (
            f"允许使用的表及表间关联关系:\n{result['selected_text']}\n\n"
            f"允许使用表的完整表结构（字段、类型、主键、注释）:\n{result['schemas_text']}"
            f"{keyword_filter_context}\n\n"
            f"用户问题:\n{question}"
        )
        return input_value