"""语义层（Schema Linking）检索算子。

替代默认 HODatasourceRetrieverOperator，解决"给大模型的表结构参考不准"问题：
官方算子走向量摘要检索（top_k 截断字段、无关联关系、无语义维护），导致 LLM 写 SQL 时
表结构不全、join 靠猜。本算子采用两阶段 Schema Linking：

1. 全量轻目录：所有表（表名+注释，注释维护业务语义与关联关系），每表一行，全量给 LLM 无 token 压力；
2. LLM 选表 + 关联：判断用哪些表、表间靠什么字段关联（一对多/多对一）；
3. 全字段加载：对选中表加载全部字段（类型/主键/注释，不截断），过滤 LLM 编造的表名。

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

from .category_recall import DEFAULT_MAX_DISTANCE, TARGET_TABLE, recall_many
from .llm import HOContextBody

logger = logging.getLogger(__name__)

# 关键词提取后最多保留的实体词个数，防止一次问题提取过多词导致串行召回次数暴涨
MAX_EXTRACT_KEYWORDS = 5

_DEFAULT_SELECT_TABLE_PROMPT = """你是一个数据库专家，请为用户问题选出所需数据表，并说明表间关联关系。

数据库 {db_name} 全部可用表目录（格式：表名 -- 表注释；注释含业务含义、字段说明及与其他表的关联关系，请仔细阅读）：
{table_catalog}

参考示例：
用户问题：查询每个用户的订单总金额
输出：{{"tables": [{{"table": "orders", "relation": "通过 user_id 关联 user 表（多对一）"}}, {{"table": "user", "relation": "与 orders 通过 user_id 关联（一对多）"}}]}}

要求：
1. 表名必须与目录完全一致原样输出，禁止编造、改大小写、加库名前缀；只能从目录选。
2. 多表才能回答的问题，关联所需表必须选全；单表可回答则只选一张，relation 留空。
3. 选表精简，一般不超过 {max_selected_tables} 张。
4. 问题中的过滤概念（品类/品牌/状态/类型/名称等）在事实表中只有外键（如 category_id）而无名称字段时，必须补选对应维度表用于 WHERE；宁可多选也不能让过滤条件无法表达，漏选维度表是常见错误。
5. 多表关联时每张表都要写 relation：
   - 事实表（引用他表）：写"通过 XX 字段关联 YY 表（多对一）"，如：通过 user_id 关联 user 表（多对一）；
   - 被引用的维度表：写"被 YY 表通过 XX 字段引用（一对多）"，如：被 orders 表通过 user_id 引用（一对多）。
6. 只输出严格 JSON，不要任何解释文字：
{{
  "tables": [
    {{"table": "表名", "relation": "关联关系说明，如：通过 user_id 关联 user 表（多对一）"}}
  ]
}}
"""

_DEFAULT_SELECT_VALUE_FIELDS_PROMPT = """你是数据库专家。下面是根据用户问题选出的数据表及其完整字段（格式：字段名 类型 [主键] -- 注释）：
{schemas}

请挑选"真实取值对正确写 WHERE 有参考价值"的字段：只要取值可能用于过滤/分组/精确或模糊匹配/大小写敏感需原样书写的字段（品类、状态、日期、品牌、类型、ID 等枚举/有限集合型）都选上。
宁可多选不漏选：LLM 参考真实取值可避免猜错过滤值导致查不到结果。

每张表最多输出 5 个，只输出严格 JSON：
{{"value_fields": {{"表名1": ["字段a", "字段b"], "表名2": ["字段c"]}}}}
某张表确实无需参考取值的，不要出现在结果中。
"""

_DEFAULT_VERIFY_VALUE_FIELDS_PROMPT = """你是数据库专家。下面是已选"需参考真实取值"的字段及其真实数据（每字段最多3个取值）：
{field_values}

用户问题：{question}

判断：这些真实取值是否足以正确写出 WHERE 过滤条件（品类/品牌/日期/状态等维度取值是否准确、可用过滤值是否齐全）？重点检查：
1. 问题里的业务过滤概念是否已有字段能看到真实取值？取值里看不到问题提到的概念（如问题说"电动风扇"，取值只有"风扇/塔式风扇"）时，需补充品类/标题类字段确认真实叫法；
2. 时间过滤字段（dt/date 等）有无取值参考；
3. 已选字段取值是否异常（全空/类型不符/字段名错误）。

足够，输出：{{"sufficient": true, "add_fields": {{}}}}
不足，从已选表完整字段中补充（每表不超 5 个、禁止编造字段名）：
{{"sufficient": false, "add_fields": {{"表名": ["补充字段1", "补充字段2"]}}}}
只输出严格 JSON。
"""

_DEFAULT_VERIFY_TABLES_PROMPT = """你是数据库专家。校验"已选表字段是否足以完整回答用户问题"（WHERE 过滤、聚合、分组、排序所需维度），不足则补表。

当前已选表（含字段与注释）：
{selected_tables}

全表目录（补充表只能从这里选）：
{table_catalog}

重点检查：
1. 问题里的过滤概念（品类/品牌/状态/类型/名称等）是否有名称字段可直接表达？若只有外键（category_id）而无名称字段（category_1/2…），则缺对应维度表，必须补充；
2. 统计口径字段（销量/金额/时间）是否齐全；
3. 时间过滤字段类型是否与问题口径匹配（如年月是 varchar 202607）。

足够，输出：{{"sufficient": true, "missing_tables": []}}
不足，从全表目录选缺失表（可多张）：
{{"sufficient": false, "missing_tables": [{{"table": "表名", "reason": "补充原因"}}]}}
要求：表名与目录完全一致、禁止编造；只补真正缺失的表；只输出严格 JSON。
"""

_DEFAULT_REWRITE_QUESTION_PROMPT = """你是对话理解助手。下面是当前问题出现前的最近几轮对话记录：
{history}

判断：当前用户问题是否引用了前文？满足任一即算引用：
1. 含指代词（那/这个/它/上面/刚才/环比/同比/继续/再/还/结果/数据）；
2. 缺主语或实体（没提品类/品牌/统计对象）；
3. 缺时间基准或口径（如"环比是多少"没说对比哪期哪个范围）；
4. 与前一问属同一话题的追问。

若引用前文：改写为自包含的完整问题，把前文的实体（品类/品牌/商品）、时间范围（如 Q1、202601-202603）、过滤条件、统计口径写进问题，脱离前文也能独立理解。
若未引用：保持原问题原样输出。

只输出严格 JSON：
{{"related": true 或 false, "rewritten_question": "改写后的完整问题（related 为 false 时与原问题相同）"}}
"""

_DEFAULT_EXTRACT_KEYWORDS_PROMPT = """你是一个数据分析助手。请从用户问题中提取"关键实体词"，用于后续语义检索（向量匹配）和选表参考。

提取规则：
1. 只提取需要"去库里匹配真实取值"的实体词：品类名（咖啡机、空调、风扇）、品牌名（苏泊尔、美的）、评论打标词（异味、噪音大、好评）、地区（广东省、华东地区）等。
2. 词要尽量短、去修饰、去广告化，贴近数据库里真实存储的叫法（如短品类词"吹风机"，而不是"吹风机pro版超强风速大风量"这种口语长词），这样向量检索才能精确命中库里已有的取值。
3. 不提取：指标（销量、金额、评论数）、时间（2026年、Q1、上半年）、动作词（分析、对比、列出）、排序/占比修饰（前3、环比、占比）、数据表字段/统计单位名（如 ASIN、SKU、asin、parent_asin、comment_id、category_id）、型号/版本后缀（pro、plus、max、2026款，如"咖啡机pro版"只保留"咖啡机"）、虚词助词。
4. 用户问题包含多个独立诉求时（如"分析评论，同时对比销量"），每个诉求里的关键实体词都要提取，不要遗漏。
5. 同一个实体只保留一次，去重。
6. 品牌名与品类名连用时（如"德龙咖啡机""美的空调"），拆成品牌名和品类名两个词分别提取，不要保留组合词。
7. 提取结果若存在包含关系（如"德龙咖啡机"包含"咖啡机"），只保留最小粒度的词。

输出严格 JSON（只输出 JSON，不要任何解释文字）：
{"关键词列表": ["咖啡机", "异味"]}

示例1 输入：帮我分析2026年上半年咖啡机的异味问题
输出：{"关键词列表": ["咖啡机", "异味"]}

示例2 输入：分别统计广东省和华东地区电动风扇的销量
输出：{"关键词列表": ["广东省", "华东地区", "电动风扇"]}
"""

_DEFAULT_TABLE_VERIFY_PROMPT = """你是数据库专家。对查询方案做"表级校验"，一次输出两组动作：补缺失表（missing_tables）、剔无关表（drop_tables）。召回 SQL 的审核在下一步单独做，本步不要关心召回。

用户问题：{question}

一、当前候选表（格式：表名 -- 表注释（来源：LLM自选/向量召回带入/兜底；字段：字段1, 字段2））：
{selected_schemas}

二、全部可用表目录（补充表只能从这里选）：
{table_catalog}

校验原则：
1. 候选表字段是否足以回答用户问题（过滤/聚合/分组/排序维度齐备）？不足则从目录补缺失表入 missing_tables（附原因）。如问题按品类过滤、销售表只有 category_id 外键，必须补品类维度表。
2. 与问题统计口径无关的表（如问销量却带进来的评论/打标表）放入 drop_tables。LLM 自选表除非确实与问题无关否则保留——宁可多留，不误删必需表。

只输出严格 JSON：
{{"missing_tables": [{{"table": "表名", "reason": "补充原因"}}], "drop_tables": ["表名"]}}
要求：表名与全表目录完全一致、禁止编造；category_embedding 是内部向量表，永远不允许进入表集合；两个键都不能省略，无动作时输出空数组。
"""

_DEFAULT_RECALL_VERIFY_PROMPT = """你是数据库专家。这是"召回审核"（表级校验后的第二步）：逐条判断每条"关键词向量召回"对回答用户问题是否有意义，无意义的序号放入 drop_recalls。

用户问题：{question}

允许使用的业务表（召回来源表不在此名单即无意义；category_embedding 是内部向量表，永不直接使用）：
{allowed_tables}

向量召回清单（每条 = 序号. 关键词；来源表（表注释）；命中列）：
{recall_rows}

逐条判断原则：
1. 口径匹配：来源表是否属于回答该问题必需的统计对象/维度（查销量需销售表+品类维度表；查评论分析才需评论/打标表）。召回到与问题统计口径无关的表是噪音，剔除；
2. 命中列语义：命中列能否定位问题中的过滤实体（品类名/品牌名/标题等实体名词列）。观点词、评论片段、情感词等"非实体定位"列，在纯统计/销量类问题里无法用于过滤，剔除；仅当问题本身要求按这些词筛选（如分析某品类差评提到的问题）才保留；
3. 冗余性：同一关键词已有更合适召回（如品类表 category_x 列）时，旁路观点/明细类表的同词召回通常冗余，剔除。

示例：用户问题"2026年5月咖啡机的销售情况"
允许表：dwd_mkt_product_sales_data、dim_product_category
召回1. 咖啡机；来源表：dim_product_category（品类维度表）；命中列：category_2, category_3, category_4 → 可用于定位咖啡机品类做 WHERE 过滤 → 保留
召回2. 咖啡机；来源表：dwm_absa_comment_detail（评论打标表）；命中列：cb_big_word（观点）→ 对销量口径无用、来源表不在允许名单 → 放入 drop_recalls

只输出严格 JSON：
{{"drop_recalls": [无用召回的序号]}}
要求：drop_recalls 只列确定无用的；允许表名单之外的表产生的召回一律无用（其来源表已在表级校验中剔除）；无向量召回时输出空数组。
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
    """语义层检索算子（Datasource Schema Linking Operator）：先全量目录+LLM 选表+关联，
    再对选中表加载全字段（不截断、过滤编造表名），输出 HOContextBody 与官方算子兼容，可直接替换接线。"""

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
        """datasource: 数据源（必填）；prompt_template: 选表提示词模板（{db_name}/{table_catalog}/{max_selected_tables}）；
        model/llm_client: 选表模型，留空用默认；max_selected_tables: 最多加载几张表全字段；
        context_key: 输出键名（默认 context）；enum_enabled/enum_limit: 是否枚举 value_fields 真实取值及其条数。"""
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
        """列出全部可用表（表名 + 表注释，注释承载业务语义与关联关系，全量无截断）。"""
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
        """SHOW CREATE TABLE 解析表级 COMMENT 兜底（表级注释在语句末尾，取最后一个匹配）。"""
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
        """加载表字段；get_columns 失败/为空（Doris 元数据连接超时）时用 SHOW CREATE TABLE 解析兜底。"""
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
        """从 SHOW CREATE TABLE 解析列定义，返回结构与 get_columns 兼容（name/type/comment/is_in_primary_key）。"""
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
        失败策略：模板格式化失败 / LLM 调用失败 / 返回空结果 → 重试 1 次；
        重试仍失败才兜底返回全表，保证流程不中断。
        """
        catalog_str = "\n".join(
            f"{i + 1}. {item}" for i, item in enumerate(catalog)
        )
        # system 放角色/目录/要求，user 放用户问题（DB-GPT 枚举用 HUMAN 表示用户消息）
        user_content = f"用户问题：{question}"
        if keywords_text:
            user_content += (
                f"\n\n用户问题中的关键实体词（选表请覆盖这些实体所在的表）：\n{keywords_text}"
            )
        last_err = ""
        for attempt in range(1, 3):  # 首次尝试 + 重试 1 次
            try:
                prompt = self._prompt_template.format(
                    db_name=self._datasource._db_name,
                    table_catalog=catalog_str,
                    max_selected_tables=self._max_selected_tables,
                )
                output = await self._llm_complete(prompt, user_content)
                result = self._parse_table_selection(output)
                if result:
                    return result
                last_err = "LLM returned empty table selection"
            except Exception as e:
                last_err = str(e)
            logger.warning(
                f"Select tables attempt {attempt}/2 failed ({last_err}), retrying"
            )
        logger.warning(
            f"Select tables still failed after retry ({last_err}), "
            f"fallback to all tables"
        )
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
        """容错解析 LLM 输出的 JSON：兼容中文引号/冒号；整体失败则截取首个 { 到末个 } 再解析。"""
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
        """表结构里只展示业务含义（第一个分号前）；分号后的关联关系在目录/关联清单中已给，不重复。"""
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
        """按选中表拼装完整表结构（字段/类型/主键/注释 + 值域枚举真实取值参考）。"""
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
            # 值域枚举：对 value_fields 查前 N 行真实取值，帮 LLM 写对 WHERE 过滤值
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

        返回 {表名: [字段, ...]}；未开启值域枚举时返回空 dict（无字段参考属正常结论）。
        失败策略：LLM 调用失败 / 解析失败 → 重试 1 次；仍失败返回空 dict 跳过该优化。
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
        last_err = ""
        for attempt in range(1, 3):  # 首次尝试 + 重试 1 次
            try:
                prompt = _DEFAULT_SELECT_VALUE_FIELDS_PROMPT.format(
                    schemas="\n\n".join(schema_lines)
                )
                output = await self._llm_complete(prompt, f"用户问题：{question}")
                return self._parse_fields_map(output, "value_fields")
            except Exception as e:
                last_err = str(e)
            logger.warning(
                f"Select value fields attempt {attempt}/2 failed ({last_err}), retrying"
            )
        logger.warning(
            f"Select value fields still failed after retry ({last_err}), skip"
        )
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
        空 add_fields（判断足够）属正常结论不重试；
        失败策略：LLM 调用失败 / 解析失败 → 重试 1 次；仍失败原样返回，不阻塞主流程。
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
        last_err = ""
        add_map = {}
        for attempt in range(1, 3):  # 首次尝试 + 重试 1 次
            try:
                prompt = _DEFAULT_VERIFY_VALUE_FIELDS_PROMPT.format(
                    field_values="\n\n".join(field_lines),
                    question=question,
                )
                output = await self._llm_complete(prompt, f"用户问题：{question}")
                add_map = self._parse_fields_map(output, "add_fields")
                break  # 解析成功即算成功（含空 add_fields = 判断足够），不再重试
            except Exception as e:
                last_err = str(e)
            logger.warning(
                f"Verify value fields attempt {attempt}/2 failed ({last_err}), retrying"
            )
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
        只做表级补充，不修改已选表的字段；sufficient=true（无需补表）属正常结论不重试；
        失败策略：LLM 调用失败 / 解析失败 → 重试 1 次；仍失败保留原选表结果，不中断主流程。
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
        last_err = ""
        sufficient, missing = True, []
        for attempt in range(1, 3):  # 首次尝试 + 重试 1 次
            try:
                prompt = _DEFAULT_VERIFY_TABLES_PROMPT.format(
                    selected_tables="\n".join(schema_lines),
                    table_catalog="\n".join(f"- {c}" for c in catalog),
                )
                output = await self._llm_complete(prompt, f"用户问题：{question}")
                sufficient, missing = self._parse_verify_result(output)
                break  # 解析成功即算成功（sufficient=true = 无需补表），不再重试
            except Exception as e:
                last_err = str(e)
            logger.warning(
                f"Verify tables attempt {attempt}/2 failed ({last_err}), retrying"
            )
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
        """容错解析关键词提取结果，返回去重后的实体词列表。

        兼容多个候选 JSON 键（关键词列表/关键词/keywords）；
        结果去重后截断到 MAX_EXTRACT_KEYWORDS 个，防止召回次数暴涨；
        解析失败或无有效列表时返回空列表，并记录告警。
        """
        try:
            data = HOSchemaLinkingRetrieverOperator._parse_json_strict(text)
        except Exception:
            logger.warning("Extract keywords parse failed, use empty list")
            return []
        kws = None
        for key in ("关键词列表", "关键词", "keywords"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                kws = candidate
                break
        if not isinstance(kws, list):
            logger.warning("Extract keywords output has no valid keyword list, use empty list")
            return []
        seen, out = set(), []
        for k in kws:
            s = str(k or "").strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        if len(out) > MAX_EXTRACT_KEYWORDS:
            logger.info(
                "Extract keywords got %d words, keep first %d: %s",
                len(out),
                MAX_EXTRACT_KEYWORDS,
                out[:MAX_EXTRACT_KEYWORDS],
            )
            out = out[:MAX_EXTRACT_KEYWORDS]
        return out

    async def _extract_keywords(self, question: str) -> List[str]:
        """关键词提取：从改写后的完整问题中提取需要语义检索/选表参考的实体词。

        只输出实体词列表（不判断目标字段）；后续可把结果交给向量检索方
        （返回 {向量子查询, 目标字段} 拼进分析 SQL）。
        失败策略：LLM 调用失败 / 返回空结果 → 重试 1 次；仍失败返回空列表，
        不阻塞主流程。
        """
        last_err = ""
        for attempt in range(1, 3):  # 首次尝试 + 重试 1 次
            try:
                output = await self._llm_complete(
                    _DEFAULT_EXTRACT_KEYWORDS_PROMPT, f"用户问题：{question}"
                )
                result = self._parse_keywords_result(output)
                if result:
                    return result
                last_err = "LLM returned empty keywords"
            except Exception as e:
                last_err = str(e)
            logger.warning(
                f"Extract keywords attempt {attempt}/2 failed ({last_err}), retrying"
            )
        logger.warning(
            f"Extract keywords still failed after retry ({last_err}), use empty list"
        )
        return []

    async def _rewrite_followup_question(
        self, question: str, input_value: AgentGenerateContext
    ) -> str:
        """追问补全：判断当前问题是否指代前文，若是则改写为自包含的完整问题。

        历史来源：数据库 gpts_message 表（通过 agent_context.conv_id 查询，与
        DataScientist 的会话记忆同源）。不读 rely_messages：AWEL Agent 流程中
        语义层位于 Agent Trigger 之后、AWEL Agent Operator 之前，起始 context
        未携带 rely_messages（实测恒为空），且 gpts_message 表是 rely 的严格超集。
        历史不可用、或 LLM 判定非追问（related=false）时，直接返回原问题，不重试。
        失败策略：LLM 调用失败 / 解析失败 / 判定需改写却未产出有效改写 →
        重试 1 次；仍失败静默返回原问题，不阻塞主流程。
        """
        current = self._strip_schema_linking_text(question).strip()
        history = await self._build_history_text_from_db(input_value)
        if not history:
            return question
        last_err = ""
        for attempt in range(1, 3):  # 首次尝试 + 重试 1 次
            try:
                prompt = _DEFAULT_REWRITE_QUESTION_PROMPT.format(history=history)
                output = await self._llm_complete(
                    prompt, f"当前用户问题：{question}"
                )
                related, rewritten = self._parse_rewrite_result(output)
                if not related:
                    return question  # 判定为自包含问题，属正常结论，不重试
                if rewritten and rewritten != current and rewritten != question:
                    return rewritten
                last_err = "LLM returned no effective rewrite"
            except Exception as e:
                last_err = str(e)
            logger.warning(
                f"Rewrite followup question attempt {attempt}/2 failed ({last_err}), retrying"
            )
        logger.warning(
            f"Rewrite followup question still failed after retry ({last_err}), "
            f"keep original question"
        )
        return question

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
    ) -> dict:
        """执行语义层流程（目录 -> 选表+关联 -> 加载字段 -> 字段确认 -> 全字段+值域枚举）。

        供普通检索算子（map）使用；Agent 版算子（map）走独立的单轮流程
        （含向量召回与合并校验），不复用本方法。
        keywords_text: 从问题提取的关键实体词文本，附加给选表 LLM 参考。
        """
        db_name = self._datasource._db_name
        dialect = self._datasource.dialect

        # 1. 全量轻目录（表名 + 注释，注释含业务语义与关联关系）
        catalog = await self._build_table_catalog()

        # 2. LLM 选表 + 关联关系
        selected = await self._select_tables(question, catalog, keywords_text)
        catalog_tables = {item.split(" -- ")[0] for item in catalog}
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
      3. 执行单轮语义层：全表目录 -> LLM 选表+关联，同时做关键词向量召回，
         合并为候选集（LLM 自选 + 召回带入）后做一次"合并校验"（补缺失表 /
         剔无关表 / 剔无意义召回），再确认值域字段并枚举真实取值；
      4. 把最终允许表、完整表结构、保留的召回 SQL（含使用引导）与用户问题
         一并写入消息内容（作为 user 消息），
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

    # ---------------- V3.1 单轮语义层：向量召回 / 合并校验 辅助 ----------------
    @staticmethod
    def _parse_recall_items(recall_sql_list: List[str]) -> List[dict]:
        """解析召回 SQL 首行注释（# 关键词:... 来自表 ...，主键 ...，命中列 ...）。

        返回 [{sql, keyword, table, key_word, columns}]；注释格式无法解析的条目
        保留原样（table 为空串），后续按"无法归表、不可随表剔除"处理。
        """
        items = []
        for sql in recall_sql_list:
            first_line = (sql.splitlines() or [""])[0]
            match = re.match(
                r"^\s*(?:#|--)\s*关键词[:：]\s*(.*?)\s*来自表\s+(\S+)\s*，主键\s+(\S+)\s*，命中列\s*(.+)$",
                first_line,
            )
            if match:
                keyword, table, key_word, columns = match.groups()
                items.append(
                    {
                        "sql": sql,
                        "keyword": keyword.strip(),
                        "table": table.strip(),
                        "key_word": key_word.strip(),
                        "columns": columns.strip(),
                    }
                )
            else:
                items.append(
                    {
                        "sql": sql,
                        "keyword": "",
                        "table": "",
                        "key_word": "",
                        "columns": "",
                    }
                )
        return items

    async def _vector_recall(self, keywords: List[str]) -> List[dict]:
        """step4 向量召回：按关键词对 category_embedding 做语义召回，SQL 带 1 起固定序号。

        固定序号贯穿后续"合并校验（drop_recalls 点名剔除）"与"保留召回 SQL"两步，
        避免两步错位。召回模块异常或解析失败时返回空列表，不阻塞主流程。
        """
        if not keywords:
            return []
        try:
            recall_results = await self.blocking_func_to_async(
                recall_many, keywords, DEFAULT_MAX_DISTANCE
            )
        except Exception as e:
            logger.warning(f"Vector recall failed, skip: {e}")
            return []
        recall_sql_list = []
        for recall_result in recall_results:
            recall_sql_list.extend(recall_result.get("sql", []))
        items = self._parse_recall_items(recall_sql_list)
        for idx, item in enumerate(items, start=1):
            item["idx"] = idx
        return items

    @staticmethod
    def _build_recall_rows_text(
        recall_items: List[dict], catalog_comments: dict
    ) -> str:
        """把召回清单渲染成"序号. 关键词；来源表（表注释）；命中列"文本，供合并校验逐条审核。"""
        rows = []
        for item in recall_items:
            table = item.get("table") or ""
            comment = catalog_comments.get(table, "") if table else ""
            table_desc = table if not comment else f"{table}（{comment}）"
            rows.append(
                f"{item.get('idx', '?')}. 关键词：{item.get('keyword') or '未知'}；"
                f"来源表：{table_desc or '未知'}；命中列：{item.get('columns') or '未知'}"
            )
        return "\n".join(rows)

    @staticmethod
    def _parse_merge_verify_result(text: str) -> dict:
        """解析合并校验输出，返回 {"missing_tables": [...], "drop_tables": [...], "drop_recalls": set}。"""
        data = HOSchemaLinkingRetrieverOperator._parse_json_strict(text)
        missing_tables = []
        for item in data.get("missing_tables") or []:
            name = item if isinstance(item, str) else None
            if name is None and isinstance(item, dict):
                name = item.get("table")
            if name and str(name) not in missing_tables:
                missing_tables.append(str(name))
        drop_tables = []
        for item in data.get("drop_tables") or []:
            name = str(item or "").strip()
            if name and name not in drop_tables:
                drop_tables.append(name)
        drop_recalls = set()
        for item in data.get("drop_recalls") or []:
            if isinstance(item, (int, float)):
                drop_recalls.add(int(item))
        return {
            "missing_tables": missing_tables,
            "drop_tables": drop_tables,
            "drop_recalls": drop_recalls,
        }

    async def _run_table_verify(
        self,
        question: str,
        candidate_names: List[str],
        selected_names: List[str],
        recall_items: List[dict],
        catalog: List[str],
        catalog_comments: dict,
        columns_by_table: dict,
    ) -> dict:
        """step9a 表级校验（LLM）：补缺失表/剔无关表；失败自动重试 1 次，仍失败抛异常由调用方降级。"""
        selected_set = set(selected_names)
        recall_set = {item.get("table") for item in recall_items if item.get("table")}
        schema_lines = []
        for table in candidate_names:
            cols = columns_by_table.get(table, [])
            col_names = ", ".join(c.get("name", "") for c in cols)
            comment = catalog_comments.get(table, "")
            if table in selected_set:
                origin = "LLM自选"
            elif table in recall_set:
                origin = "向量召回带入"
            else:
                origin = "兜底"
            schema_lines.append(f"{table} -- {comment}（{origin}；字段：{col_names}）")
        prompt = _DEFAULT_TABLE_VERIFY_PROMPT.format(
            question=question,
            selected_schemas="\n".join(schema_lines),
            table_catalog="\n".join(f"- {item}" for item in catalog),
        )
        last_err = ""
        for attempt in range(1, 3):
            try:
                output = await self._llm_complete(prompt, f"用户问题：{question}")
                return self._parse_merge_verify_result(output)
            except Exception as e:
                last_err = str(e)
                logger.warning(f"Table verify attempt {attempt}/2 failed: {last_err}")
        raise RuntimeError(f"Table verify failed after retry: {last_err}")

    async def _run_recall_verify(
        self,
        question: str,
        allowed_tables: List[str],
        recall_items: List[dict],
        catalog_comments: dict,
    ) -> dict:
        """step9b 召回审核（LLM）：按序号剔无意义召回；失败自动重试 1 次，仍失败抛异常由调用方降级。"""
        prompt = _DEFAULT_RECALL_VERIFY_PROMPT.format(
            question=question,
            allowed_tables=(
                "\n".join(f"- {t}" for t in allowed_tables) or "（无允许表）"
            ),
            recall_rows=(
                self._build_recall_rows_text(recall_items, catalog_comments)
                or "（无向量召回）"
            ),
        )
        last_err = ""
        for attempt in range(1, 3):
            try:
                output = await self._llm_complete(prompt, f"用户问题：{question}")
                return self._parse_merge_verify_result(output)
            except Exception as e:
                last_err = str(e)
                logger.warning(f"Recall verify attempt {attempt}/2 failed: {last_err}")
        raise RuntimeError(f"Recall verify failed after retry: {last_err}")

    async def map(self, input_value: AgentGenerateContext) -> AgentGenerateContext:
        """单轮语义层：选表+校验后把最终表结构与保留的召回 SQL 注入用户消息原样透传。

        流程 step1-14：取问题→追问补全→提关键词→向量召回→全表目录→LLM 选表+关联→
        组候选→加载全字段→9a 表级校验(剔/补)→9b 召回审核(剔无用)→9.5 绑定规则+定表→
        10-12 值域字段确认/过滤/校验补→13 拼表结构+召回使用引导→14 组装 user 消息。
        """
        if not input_value.message:
            raise ValueError("The message is empty.")
        question = input_value.message.content or ""

        # step2 追问补全：把指代前文的半截话改写为自包含完整问题（有历史时才触发 LLM）
        question = await self._rewrite_followup_question(question, input_value)

        # step3 关键词提取：抽实体词（品类/品牌/观点词），供召回与选表参考
        keywords = await self._extract_keywords(question)
        keywords_text = "、".join(keywords) if keywords else ""
        logger.info(f"schema_linking extracted keywords: {keywords}")

        # step4 向量召回：每条带固定序号，后续校验按序号剔（drop_recalls）
        recall_items = await self._vector_recall(keywords)
        if recall_items:
            logger.info(
                "Vector recall got %d sqls: %s",
                len(recall_items),
                [f"{i['idx']}. {i['table']}/{i['columns']}" for i in recall_items],
            )

        # step5 全量轻目录（表名+注释，注释含业务语义与关联关系）
        catalog = await self._build_table_catalog()
        catalog_tables = {item.split(" -- ")[0] for item in catalog}
        catalog_comments = self._build_catalog_comments(catalog)

        # step6 LLM 选表+关联（关键实体词一并给 LLM 参考）
        selected = await self._select_tables(question, catalog, keywords_text)
        selected_names = []
        for item in selected:
            name = item["table"]
            if (
                name
                and name != TARGET_TABLE
                and name in catalog_tables
                and name not in selected_names
            ):
                selected_names.append(name)
        logger.info(f"schema_linking LLM selected tables: {selected_names}")

        # step7 组候选 = LLM 自选 + 召回来源表（recall_only）；category_embedding 永不进候选/允许表
        candidate_names = list(selected_names)
        recall_only_names = []
        for item in recall_items:
            table = item.get("table") or ""
            if not table or table == TARGET_TABLE or table not in catalog_tables:
                continue
            if table not in candidate_names:
                candidate_names.append(table)
                recall_only_names.append(table)
        if not candidate_names:
            logger.warning("No valid tables selected, fallback to catalog tables")
            candidate_names = [
                t for t in catalog_tables if t != TARGET_TABLE
            ][: self._max_selected_tables]

        # step8 加载候选表完整字段（类型/主键/注释）
        connector = self._datasource.connector
        columns_by_table = {}
        for table in candidate_names:
            columns_by_table[table] = await self._get_columns_with_fallback(
                connector, table
            )

        # step9a 表级校验：剔无关表/补缺失表（失败内部已重试，仍失败降级为只留 LLM 自选表）
        table_actions = {"missing_tables": [], "drop_tables": []}
        try:
            table_actions = await self._run_table_verify(
                question,
                candidate_names,
                selected_names,
                recall_items,
                catalog,
                catalog_comments,
                columns_by_table,
            )
        except Exception as e:
            logger.warning(
                f"Table verify failed, degrade to LLM-selected tables only: {e}"
            )
            # 保守降级：只信 step6 的 LLM 自选表，recall_only 表全部出局
            table_actions = {
                "missing_tables": [],
                "drop_tables": list(recall_only_names),
            }

        # 应用表级动作 → drop_names / missing_tables
        drop_names = {
            t
            for t in table_actions["drop_tables"]
            if t and t != TARGET_TABLE and t in candidate_names
        }
        missing_tables = [
            t
            for t in table_actions["missing_tables"]
            if t
            and t != TARGET_TABLE
            and t in catalog_tables
            and t not in candidate_names
        ]
        if drop_names:
            logger.info(f"Table verify dropped tables: {sorted(drop_names)}")
        # 中间允许表 = 候选 - 表级剔除 + 补充表，作为 step9b 召回审核的口径参照
        allowed_names = [t for t in candidate_names if t not in drop_names]
        allowed_names.extend(missing_tables)

        # step9b 召回审核：按序号剔无意义召回（失败内部已重试，仍失败只留 LLM 自选表的召回）
        recall_actions = {"drop_recalls": set()}
        try:
            recall_actions = await self._run_recall_verify(
                question,
                allowed_names,
                recall_items,
                catalog_comments,
            )
        except Exception as e:
            logger.warning(
                f"Recall verify failed, drop recalls of non-LLM-selected tables: {e}"
            )
            selected_set = set(selected_names)
            recall_actions = {
                "drop_recalls": {
                    item["idx"]
                    for item in recall_items
                    if item.get("table") and item["table"] not in selected_set
                }
            }
        if recall_actions["drop_recalls"]:
            logger.info(
                "Recall verify dropped recall ids: %s",
                sorted(recall_actions["drop_recalls"]),
            )

        # step9.5 绑定规则：recall_only 表召回全部被剔 → 出局
        for table in recall_only_names:
            recall_idx = [
                item["idx"] for item in recall_items if item.get("table") == table
            ]
            if recall_idx and all(
                i in recall_actions["drop_recalls"] for i in recall_idx
            ):
                drop_names.add(table)

        # step9.5 最终允许表 = 中间允许表 - 出局表 + 补充表；补充表此刻补加载字段
        final_names = [t for t in allowed_names if t not in drop_names]
        for table in final_names:
            if table not in columns_by_table:
                columns_by_table[table] = await self._get_columns_with_fallback(
                    connector, table
                )
        columns_by_table = {
            table: columns_by_table[table]
            for table in final_names
            if table in columns_by_table
        }

        # 保留的召回 SQL：未被 drop_recalls 点名 且 其来源表 ∈ final_tables（同生共死）
        kept_items = []
        for item in recall_items:
            if item.get("idx") in recall_actions["drop_recalls"]:
                continue
            table = item.get("table") or ""
            if table and table not in final_names:
                continue
            kept_items.append(item)
        if kept_items:
            logger.info(
                "Keep %d recall sqls: %s",
                len(kept_items),
                [f"{i['idx']}. {i['table']}/{i['columns']}" for i in kept_items],
            )

        # 按最终允许表重算 selected_text，避免被剔表残留在提示词里
        relation_by_table = {}
        for item in selected:
            name = item["table"]
            if name in final_names:
                relation_by_table[name] = item.get("relation") or ""
        for table in final_names:
            if table in relation_by_table:
                continue
            if table in recall_only_names:
                relation_by_table[table] = (
                    "向量召回命中的来源业务表：可按召回 SQL 输出的主键值做实体过滤/关联"
                )
            else:
                relation_by_table[table] = ""
        for table in missing_tables:
            relation_by_table[table] = "合并校验补充：用于表达问题中缺失的过滤维度"
        if final_names:
            selected_text = "\n".join(
                f"- {table}"
                + (
                    f" -- 关联: {relation_by_table[table]}"
                    if relation_by_table.get(table)
                    else ""
                )
                for table in final_names
            )
        else:
            selected_text = "- (无可用表)"

        # step10-12 值域字段：确认 -> 过滤编造列名 -> 查真实取值校验并补字段
        value_fields_map = await self._select_value_fields(question, columns_by_table)
        value_fields_map = self._filter_value_fields(value_fields_map, columns_by_table)
        value_fields_map = await self._verify_and_fix_value_fields(
            question, columns_by_table, value_fields_map
        )

        # step13 拼装完整表结构（含真实取值参考）+ 保留召回 SQL 使用引导
        schemas_text = await self._build_schemas_text(
            final_names, catalog_comments, columns_by_table, value_fields_map
        )
        recall_context = ""
        if kept_items:
            # 精简版"使用引导"：仅在有保留召回时出现；无召回/召回全剔时整段不进入提示词
            recall_context = (
                "\n\n关键词向量召回 SQL 使用说明：\n"
                "每段召回 SQL 会把问题中的实体词（品类/品牌/观点等）在内部向量表 "
                "category_embedding 中按语义相似命中来源业务表的行，并输出这些行的标识列"
                "（主键）真实取值，用于翻译成 WHERE/关联条件；不要改动其中的向量距离阈值与子查询。\n"
                "标识列为单列主键：结果可直接用于 IN 过滤或按该主键关联；"
                "为多列复合主键（逗号分隔）：必须逐列分别匹配"
                "（AND t.a = r.a AND t.b = r.b ...），禁止把整串当单个列名。\n"
                "category_embedding 是内部向量表，禁止在业务 SQL 中直接查询/关联。\n\n"
                "关键词向量召回来源 SQL：\n"
                + "\n\n".join(item["sql"] for item in kept_items)
            )

        # step14 组装 user 消息：允许表 + 完整表结构 + 召回来源 SQL + 用户问题
        input_value.message.content = (
            f"允许使用的表及表间关联关系:\n{selected_text}\n\n"
            f"允许使用表的完整表结构（字段、类型、主键、注释）:\n{schemas_text}"
            f"{recall_context}\n\n"
            f"用户问题:\n{question}"
        )
        return input_value