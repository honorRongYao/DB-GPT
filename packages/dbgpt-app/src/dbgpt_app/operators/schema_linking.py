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
from typing import List, Optional

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
5. 多表关联时，每张选中的表都必须填写 relation，描述该表与其他表的关联关系，不能留空：
   - 如果该表通过某字段关联其他表（事实表），写"通过 XX 字段关联 YY 表（多对一）"，如：通过 user_id 关联 user 表（多对一）；
   - 如果该表是被其他表引用的维度表，写"被 YY 表通过 XX 字段引用（一对多）"，如：被 orders 表通过 user_id 引用（一对多）。
6. 输出严格 JSON，不要包含任何其他解释文字，格式如下：
{{
  "tables": [
    {{"table": "表名", "relation": "该表与其他表的关联关系说明，如：通过 user_id 关联 user 表（多对一）"}}
  ]
}}
"""

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
        """
        MapOperator.__init__(self, **kwargs)
        MixinLLMOperator.__init__(self, llm_client, save_model_output=False)
        self._datasource = datasource
        self._prompt_template = prompt_template
        self._model = model
        self._max_selected_tables = max_selected_tables
        self._context_key = context_key

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
            logger.info(f"Select table prompt:\n{prompt}")
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

    async def _load_table_schemas(self, selected: List[dict], catalog: List[str]) -> str:
        """按选中的表加载完整字段（不截断，含类型/主键/注释）。

        表注释只展示业务含义（第一个分号前），分号后的关联关系不重复展示，
        关联关系已由"根据用户问题选择的表及表间关联关系"节给出。
        """
        connector = self._datasource.connector
        catalog_tables = {item.split(" -- ")[0] for item in catalog}
        # 表名 -> 表注释（从目录中获取）
        catalog_comments = {}
        for item in catalog:
            parts = item.split(" -- ", 1)
            catalog_comments[parts[0]] = parts[1] if len(parts) > 1 else ""
        # 去重并限制数量，过滤掉目录里不存在的表（防止 LLM 编造表名）
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

        schemas = []
        for table_name in table_names:
            try:
                columns = await self.blocking_func_to_async(
                    connector.get_columns, table_name
                )
            except Exception as e:
                logger.warning(f"Get columns of table {table_name} failed: {e}")
                columns = []
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
            schemas.append(schema)
        return "\n\n".join(schemas)

    # ---------------- 主流程 ----------------
    async def _run_schema_linking(self, question: str) -> dict:
        """执行语义层三步流程（目录 -> 选表+关联 -> 全字段），返回结果字典。

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

        # 3. 按选中表加载全字段
        schemas_text = await self._load_table_schemas(selected, catalog)
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
        logger.info(f"Schema linking context:\n{context}")
        return HOContextBody(context_key=self._context_key, context=context)


class HOSchemaLinkingAgentOperator(HOSchemaLinkingRetrieverOperator):
    """Agent 语义层算子（Agent Schema Linking Operator）。

    用于在 AWEL Agent 流程中串联语义层：
      1. 接收上游 Agent 触发器输出的 AgentGenerateContext，取出其中的用户问题；
      2. 执行两阶段语义层（全表目录 -> LLM 选表+关联 -> 按选中表加载全字段）；
      3. 把选好的表结构与用户问题一并写入消息内容（作为 user 消息），
         让下游 Agent（如 DataScientist）严格按语义层选出的表生成 SQL；
      4. 原样透传 AgentGenerateContext 给下游 AWEL Agent Operator。

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
        result = await self._run_schema_linking(question)
        # 把语义层选好的表结构与用户问题一起作为 user 消息，约束 Agent 只使用这些表
        input_value.message.content = (
            "以下数据表已由语义层根据你的问题选好（含表间关联关系与完整表结构），"
            "请严格使用这些表生成 SQL 进行数据分析，禁止使用除此之外的任何表，"
            "禁止编造字段名，禁止添加库名前缀：\n\n"
            f"根据用户问题选择的表及表间关联关系:\n{result['selected_text']}\n\n"
            f"选中表的完整表结构（字段、类型、主键、注释）:\n{result['schemas_text']}\n\n"
            f"用户问题:\n{question}"
        )
        logger.info(
            f"Agent schema linking injected for db {result['db_name']}:\n"
            f"{input_value.message.content}"
        )
        return input_value
