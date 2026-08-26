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
from typing import List, Optional, Tuple

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

# 追问补全相关常量：历史只取最近几轮、单条内容截断，避免上下文撑爆模型输入
_REWRITE_HISTORY_MAX_MESSAGES = 6
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
        enum_limit: int = 30,
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
            enum_limit: 每个字段最多枚举多少个取值（默认 30）
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

    # ---------------- 2. LLM 选表 + 关联 ----------------
    async def _select_tables(self, question: str, catalog: List[str]) -> List[dict]:
        """调用 LLM 从目录中选择相关表并判断关联关系。"""
        catalog_str = "\n".join(
            f"{i + 1}. {item}" for i, item in enumerate(catalog)
        )
        try:
            prompt = self._prompt_template.format(
                db_name=self._datasource._db_name,
                table_catalog=catalog_str,
                max_selected_tables=self._max_selected_tables,
            )
            # logger.info(f"Select table prompt:\n{prompt}")
            # system 放角色/目录/要求，user 放用户问题（DB-GPT 枚举用 HUMAN 表示用户消息）
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt),
                ModelMessage(
                    role=ModelMessageRoleType.HUMAN, content=f"用户问题：{question}"
                ),
            ]
            model_request = await self._build_model_request(messages)
            model_output: ModelOutput = await self.llm_client.generate(model_request)
            return self._parse_table_selection(model_output.text)
        except Exception as e:
            logger.warning(f"LLM select tables failed, fallback to all tables: {e}")
            # 兜底：返回全部表（只取纯表名，不带注释串），保证不中断流程
            return [
                {"table": t.split(" -- ", 1)[0], "relation": ""} for t in catalog
            ]

    async def _build_model_request(self, messages: List[ModelMessage]) -> ModelRequest:
        models = await self.llm_client.models()
        if not models:
            raise ValueError("No models available.")
        model = self._model or models[0].model
        return ModelRequest.build_request(model, messages=messages)

    def _parse_table_selection(self, text: str) -> List[dict]:
        """解析 LLM 输出的 JSON，提取选中的表及关联关系。"""
        text = text.strip()
        try:
            data = json.loads(text)
        except Exception:
            # 去掉 ```json ... ``` 代码块包裹后重试
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Can not parse LLM output: {text}")
            data = json.loads(text[start : end + 1])
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
            col_lines = []
            for col in columns:
                name = col.get("name", "")
                col_type = col.get("type", "")
                pk = " [PK]" if col.get("is_in_primary_key") else ""
                comment = str(col.get("comment") or "").strip()
                if comment:
                    col_lines.append(f"    {name} {col_type}{pk} -- {comment}")
                else:
                    col_lines.append(f"    {name} {col_type}{pk}")
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
                    # 行数上限：画布参数可调，但硬性封顶 3 行，避免撑大提示词
                    enum_limit = min(self._enum_limit, 3)
                    rows = await self._enumerate_field_values(
                        connector, table_name, vf_list, enum_limit
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
            col_lines = []
            for col in columns:
                name = col.get("name", "")
                col_type = col.get("type", "")
                comment = str(col.get("comment") or "").strip()
                if comment:
                    col_lines.append(f"    {name} {col_type} -- {comment}")
                else:
                    col_lines.append(f"    {name} {col_type}")
            schema_lines.append(f"{table_name}(\n" + "\n".join(col_lines) + "\n)")
        if not schema_lines:
            return {}
        try:
            prompt = _DEFAULT_SELECT_VALUE_FIELDS_PROMPT.format(
                schemas="\n\n".join(schema_lines)
            )
            # logger.info(f"Select value fields prompt:\n{prompt}")
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt),
                ModelMessage(
                    role=ModelMessageRoleType.HUMAN, content=f"用户问题：{question}"
                ),
            ]
            model_request = await self._build_model_request(messages)
            model_output: ModelOutput = await self.llm_client.generate(model_request)
            return self._parse_value_fields(model_output.text)
        except Exception as e:
            logger.warning(f"LLM select value fields failed, skip: {e}")
            return {}

    def _parse_value_fields(self, text: str) -> dict:
        """解析 LLM 输出的 value_fields JSON，返回 {表名: [字段, ...]}。"""
        text = text.strip()
        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Can not parse LLM output: {text}")
            data = json.loads(text[start : end + 1])
        vf_map = data.get("value_fields") or {}
        result = {}
        for table, fields in vf_map.items():
            if not isinstance(fields, list):
                continue
            cleaned = [f for f in fields if isinstance(f, str) and f.strip()][:5]
            if cleaned:
                result[str(table)] = cleaned
        return result

    # ---------------- 3.5 二次兜底：选表校验 + 补充缺失表 ----------------
    def _parse_verify_result(self, text: str) -> Tuple[bool, List[str]]:
        """解析校验结果 JSON，返回 (是否足够, 缺失表名列表)。"""
        text = text.strip()
        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Can not parse LLM output: {text}")
            data = json.loads(text[start : end + 1])
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
        catalog_comments = {}
        for item in catalog:
            parts = item.split(" -- ", 1)
            catalog_comments[parts[0]] = parts[1] if len(parts) > 1 else ""
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
            # logger.info(f"Verify tables prompt:\n{prompt}")
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt),
                ModelMessage(
                    role=ModelMessageRoleType.HUMAN, content=f"用户问题：{question}"
                ),
            ]
            model_request = await self._build_model_request(messages)
            model_output: ModelOutput = await self.llm_client.generate(model_request)
            sufficient, missing = self._parse_verify_result(model_output.text)
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
        added = [t for t in result if t not in table_names]
        # if added:
        #     logger.info(f"Verify tables: supplement missing tables {added}")
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
        """从数据库 gpts_message 表读取同一会话的最近消息（与 Agent 记忆同源）。

        注意：每一轮对话的 agent_conv_id 会带递增后缀（如 abc_1、abc_2），
        而语义层执行时当前轮消息尚未入库，所以必须像 Agent 历史恢复那样
        用真实 conv_id 做 LIKE 匹配（get_by_agent），而不是 get_by_conv_id 精确匹配。
        取最近几轮拼成摘要；查询失败时静默返回空，由上层回退到原问题。
        """
        try:
            agent_context = getattr(input_value, "agent_context", None)
            conv_id = getattr(agent_context, "conv_id", None)
            if not conv_id:
                return ""
            from dbgpt.agent.util.conv_utils import parse_conv_id
            from dbgpt_serve.agent.agents.db_gpts_memory import (
                MetaDbGptsMessageMemory,
            )

            real_conv_id, _ = parse_conv_id(conv_id)
            # 标准 DB-GPT 对话中，UserProxyAgent 的 role 为 "Human"，
            # 会话内每条消息的 sender 或 receiver 必有一方是 "Human"，故一次查询即可覆盖整段对话
            message_memory = MetaDbGptsMessageMemory()
            messages = await self.blocking_func_to_async(
                message_memory.get_by_agent, real_conv_id, _REWRITE_HISTORY_ROLE
            )
            if not messages:
                return ""
            # DAO get_by_agent 已按 (conv_id 后缀, rounds) 排好序（即会话顺序），
            # 不要用 rounds 重排：rounds 在每一轮（每个 conv_id）内都会从 0 重新计数，
            # 按 (rounds, created_at) 重排会把多轮消息的顺序打乱（如 q1,q2,a1,a2）
            history = []
            for m in messages[-_REWRITE_HISTORY_MAX_MESSAGES:]:
                content = (getattr(m, "content", None) or "").strip()
                if not content:
                    continue
                role = (getattr(m, "role", None) or "").lower()
                label = "用户" if role in ("human", "user") else "助手"
                line = HOSchemaLinkingRetrieverOperator._format_history_item(
                    label, content
                )
                if line:
                    history.append(line)
            return "\n".join(history)
        except Exception as e:
            logger.warning(f"Build history from db failed, skip: {e}")
            return ""

    def _parse_rewrite_result(self, text: str) -> Tuple[bool, str]:
        """解析改写结果 JSON，返回 (是否引用了前文, 改写后的问题)。"""
        text = text.strip()
        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Can not parse LLM output: {text}")
            data = json.loads(text[start : end + 1])
        related = bool(data.get("related", False))
        rewritten = str(data.get("rewritten_question") or "").strip()
        return related, rewritten

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
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt),
                ModelMessage(
                    role=ModelMessageRoleType.HUMAN,
                    content=f"当前用户问题：{question}",
                ),
            ]
            model_request = await self._build_model_request(messages)
            model_output: ModelOutput = await self.llm_client.generate(model_request)
            related, rewritten = self._parse_rewrite_result(model_output.text)
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
        # logger.info(f"Rewrite followup question: {question} -> {rewritten}")
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
    async def _run_schema_linking(self, question: str) -> dict:
        """执行语义层流程（目录 -> 选表+关联 -> 加载字段 -> 字段确认 -> 全字段+值域枚举）。

        供普通检索算子（map）与 Agent 版算子（map）共用。
        """
        db_name = self._datasource._db_name
        dialect = self._datasource.dialect

        # 1. 全量轻目录（表名 + 注释，注释含业务语义与关联关系）
        catalog = await self._build_table_catalog()

        # 2. LLM 选表 + 关联关系
        selected = await self._select_tables(question, catalog)
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
        catalog_tables = {item.split(" -- ")[0] for item in catalog}
        seen = set()
        table_names = []
        for item in selected:
            name = item["table"]
            if name in seen or name not in catalog_tables:
                continue
            seen.add(name)
            table_names.append(name)
            if len(table_names) >= self._max_selected_tables:
                break
        if not table_names:
            logger.warning("No valid selected tables, fallback to all tables")
            table_names = list(catalog_tables)[: self._max_selected_tables]

        columns_by_table = {}
        for t in table_names:
            try:
                columns_by_table[t] = await self.blocking_func_to_async(
                    connector.get_columns, t
                )
            except Exception as e:
                logger.warning(f"Get columns of table {t} failed: {e}")
                columns_by_table[t] = []

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
                try:
                    columns_by_table[t] = await self.blocking_func_to_async(
                        connector.get_columns, t
                    )
                except Exception as e:
                    logger.warning(f"Get columns of table {t} failed: {e}")
                    columns_by_table[t] = []

        # 4. 字段确认：LLM 看到选中表完整字段后，指定需要参考真实取值的字段
        value_fields_map = await self._select_value_fields(question, columns_by_table)

        # 5. 拼装表结构（含字段取值参考）
        catalog_comments = {}
        for item in catalog:
            parts = item.split(" -- ", 1)
            catalog_comments[parts[0]] = parts[1] if len(parts) > 1 else ""
        schemas_text = await self._build_schemas_text(
            table_names, catalog_comments, columns_by_table, value_fields_map
        )
        return {
            "db_name": db_name,
            "dialect": dialect,
            "selected_text": selected_text,
            "schemas_text": schemas_text,
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
        # logger.info(f"Schema linking context:\n{context}")
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

    async def map(self, input_value: AgentGenerateContext) -> AgentGenerateContext:
        """执行语义层并把选表结果注入用户消息，然后原样透传。"""
        if not input_value.message:
            raise ValueError("The message is empty.")
        question = input_value.message.content or ""
        # 追问补全：结合历史对话，把"那环比是多少了"这类指代问题改写为
        # "咖啡机Q1销量环比上季度"的完整问题，避免漏选品类表、口径错乱
        question = await self._rewrite_followup_question(question, input_value)
        result = await self._run_schema_linking(question)
        # 把语义层选好的表结构与用户问题一起作为 user 消息，约束 Agent 只使用这些表
        input_value.message.content = (
            "以下数据表已由语义层根据你的问题选好（含表间关联关系与完整表结构），"
            "请严格使用这些表生成 SQL 进行数据分析，禁止使用除此之外的任何表，"
            "禁止编造字段名，禁止添加库名前缀。"
            "必须完整实现用户问题的所有量化要求：例如 TOP3/前N/排名必须用窗口函数"
            "或 LIMIT 取前N，枚举类过滤值不确定时用 LIKE 模糊匹配，聚合、分组、排序"
            "与问题口径一致；生成 SQL 后逐项自查是否满足。\n\n"
            f"根据用户问题选择的表及表间关联关系:\n{result['selected_text']}\n\n"
            f"选中表的完整表结构（字段、类型、主键、注释）:\n{result['schemas_text']}\n\n"
            f"用户问题:\n{question}"
        )
        # logger.info(
        #     f"Agent schema linking injected for db {result['db_name']}:\n"
        #     f"{input_value.message.content}"
        # )
        return input_value