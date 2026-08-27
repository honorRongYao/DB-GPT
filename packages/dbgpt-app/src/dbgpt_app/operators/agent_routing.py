"""Agent workflow operators for database and web-search routing."""

import asyncio
import json
import logging
import re
from typing import Dict, Optional

from dbgpt.agent import AgentGenerateContext
from dbgpt.agent.core.plan.awel.agent_operator import AgentBranchOperator
from dbgpt.agent.resource.database import DBResource
from dbgpt.core import (
    LLMClient,
    ModelMessage,
    ModelMessageRoleType,
    ModelOutput,
    ModelRequest,
)
from dbgpt.core.awel import BranchFunc, BranchTaskType
from dbgpt.core.awel.flow import (
    TAGS_ORDER_HIGH,
    IOField,
    OperatorCategory,
    OperatorType,
    Parameter,
    ViewMetadata,
)
from dbgpt.model.operators import MixinLLMOperator
from dbgpt.util.i18n_utils import _

from .schema_linking import HOSchemaLinkingAgentOperator

logger = logging.getLogger(__name__)

_DATABASE_INTENT_KEY = "database_intent"
_DEFAULT_INTENT_PROMPT = """你是一个请求路由器。请判断用户问题是否需要查询【新宝公司】的业务数据库。

本数据库 {db_name} 存储的是【新宝公司】的业务数据，全部表目录（格式：表名 -- 表注释，下方缩进为该表的主要字段，请根据字段判断该表存了什么数据）：
{table_catalog}

本库中包含的品牌/产品：{brands}

判断规则：
1. 用户问题未指明公司时，默认按【新宝公司】处理。只要该问题属于【新宝公司】自身的业务数据范畴（例如按品牌/产品/客户/门店/时间统计销量、金额、数量、排名、汇总、明细筛选等，且该数据来自上述表中），database_intent 就为 true。
2. 问题中提到的**品牌或产品**：如果属于上面"本库中包含的品牌/产品"列表（或属于新宝旗下品牌），即使没提"新宝"，也属于新宝业务数据，应走数据库，不能误判成其他公司。
3. 以下情况必须返回 false（走网页搜索）：
   - 问题明确涉及**其他外部公司**（如美的、格力、海尔等不属于本库品牌的公司）的业务或信息；
   - 问题涉及行业格局、市场规模排行、产业报告、宏观政策、新闻时事、常识百科等外部信息；
   - 上述表的数据无法直接回答该问题（即数据库内查询不到，查不到就走百度）；
   - 问题问的是"哪家公司""最大的公司""行业排名""市场地位"等需要外部信息才能回答的表述。
4. 默认倾向 false：不确定时返回 false，交给网页搜索兜底。
5. 只判断是否应进入数据库分支，不要选表或生成 SQL。
6. 严格输出 JSON，不要附加解释：{{"database_intent": true}} 或 {{"database_intent": false}}。

参考示例：
- "新宝公司哪款吹叶机在2024年4月销量最好" -> {{"database_intent": true}}
- "哪款吹叶机在2024年4月销量最好" -> {{"database_intent": true}}
- "BEKO空气炸锅2024年5月销量" -> {{"database_intent": true}}
- "美的公司今年营收怎么样" -> {{"database_intent": false}}
- "2026年顺德最大的电器公司是哪家" -> {{"database_intent": false}}
- "查询新宝公司每个用户的订单总金额" -> {{"database_intent": true}}
- "今天北京的天气怎么样" -> {{"database_intent": false}}
"""


class AgentDatabaseBranchOperator(MixinLLMOperator, AgentBranchOperator):
    """基于内置智能体分支算子的数据库路由算子。

    在 branches() 内直接读取数据库全量表目录并让 LLM 判断用户问题是否需要查库：
    需要查库 -> 语义层（Schema Linking）分支；否则 -> 百度搜索分支。
    相比内置 AgentBranchOperator（只按 next_speakers 路由到 Agent 下游），
    本算子支持把请求路由到非 Agent 下游（如语义层算子）。
    """

    metadata = ViewMetadata(
        label=_("Agent Database Branch Operator"),
        name="agent_database_branch_operator",
        description=_(
            "读取数据库全量表目录，由 LLM 判断用户问题是否需要查库："
            "需要则进入语义层（Schema Linking）分支，否则进入百度搜索分支。"
        ),
        category=OperatorCategory.AGENT,
        operator_type=OperatorType.BRANCH,
        parameters=[
            Parameter.build_from(
                _("Datasource"),
                "datasource",
                type=DBResource,
                description=_("用于读取全部表名和表注释的数据库资源。"),
            ),
            Parameter.build_from(
                _("Intent Prompt Template"),
                "prompt_template",
                type=str,
                optional=True,
                default=_DEFAULT_INTENT_PROMPT,
                description=_(
                    "数据库意图判断提示词，可使用 {db_name}、{table_catalog} 和 {brands}。"
                ),
            ),
            Parameter.build_from(
                _("Database Brands"),
                "database_brands",
                type=str,
                optional=True,
                default=None,
                description=_(
                    "本库包含的品牌/产品列表（可选，逗号分隔），用于区分库内品牌与其他公司。"
                    "例如：新宝,Beko,Donlim。问题中提到的品牌若在此列表中则走数据库。"
                ),
            ),
            Parameter.build_from(
                _("Model Name"),
                "model",
                type=str,
                optional=True,
                default=None,
                description=_("意图判断模型名称，留空时使用第一个可用模型。"),
            ),
            Parameter.build_from(
                _("LLM Client"),
                "llm_client",
                type=LLMClient,
                optional=True,
                default=None,
                description=_("用于数据库意图判断的 LLM 客户端。"),
            ),
        ],
        inputs=[
            IOField.build_from(
                _("Agent Operator Context"),
                "agent_operator_context",
                AgentGenerateContext,
                description=_("上游 Agent 请求。"),
            )
        ],
        outputs=[
            IOField.build_from(
                _("Database Request"),
                "database_request",
                AgentGenerateContext,
                description=_("发送到数据库（语义层）分支。"),
            ),
            IOField.build_from(
                _("Web Search Request"),
                "web_search_request",
                AgentGenerateContext,
                description=_("发送到百度搜索分支。"),
            ),
        ],
        tags={"order": TAGS_ORDER_HIGH},
    )

    def __init__(
        self,
        datasource: DBResource,
        prompt_template: str = _DEFAULT_INTENT_PROMPT,
        database_brands: Optional[str] = None,
        model: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        database_task_name: Optional[str] = None,
        web_search_task_name: Optional[str] = None,
        **kwargs,
    ):
        """初始化数据库路由算子。

        参数说明：
            datasource: 数据库资源（必填），用于读取表目录判断能否查库
            prompt_template: 意图判断提示词模板，占位符 {db_name}/{table_catalog}/{brands}
            database_brands: 本库包含的品牌/产品列表（逗号分隔），用于区分库内品牌与其他公司
            model: 判断使用的模型名称，留空用默认模型
            llm_client: LLM 客户端，留空用系统默认
            database_task_name / web_search_task_name: 分支目标节点名（可选，
                留空时自动从直接下游识别：语义层算子为数据库分支，其余为搜索分支）
        """
        AgentBranchOperator.__init__(self, **kwargs)
        MixinLLMOperator.__init__(self, llm_client, save_model_output=False)
        self._datasource = datasource
        self._prompt_template = prompt_template
        self._database_brands = database_brands
        self._model = model
        self._database_task_name = database_task_name
        self._web_search_task_name = web_search_task_name

    async def _get_comment_by_show_create(self, connector, table_name: str) -> str:
        """用 SHOW CREATE TABLE 解析表级 COMMENT，作为注释兜底。

        information_schema 读不到注释时（如 Doris/StarRocks），SHOW CREATE TABLE
        会带出建表语句中的表级 COMMENT，取最后一个匹配项（表级注释在语句末尾）。
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

    async def _build_table_catalog(self) -> str:
        """列出全部可用表及其内容描述（表名 + 表注释 + 主要字段）。"""
        connector = self._datasource.connector
        table_names = list(
            await self.blocking_func_to_async(connector.get_table_names)
        )
        catalog = []
        for table_name in table_names:
            comment = ""
            try:
                comment_raw = await self.blocking_func_to_async(
                    connector.get_table_comment, table_name
                )
                if isinstance(comment_raw, dict):
                    comment = (comment_raw or {}).get("text") or ""
                else:
                    comment = str(comment_raw or "")
            except Exception as e:
                logger.warning("Get comment of table %s failed: %s", table_name, e)
            comment = str(comment).strip()
            if not comment:
                # Doris/StarRocks 等引擎 information_schema 不填充注释，用 SHOW CREATE TABLE 兜底
                comment = await self._get_comment_by_show_create(connector, table_name)
            line = table_name
            if comment:
                line += f" -- {comment}"
            # 补充主要字段名，帮助 LLM 判断该表是否存了能回答问题的数据
            try:
                columns = await self.blocking_func_to_async(
                    connector.get_columns, table_name
                )
                field_names = [
                    col.get("name", "") for col in columns if col.get("name")
                ]
                if field_names:
                    line += "\n    字段: " + ", ".join(field_names[:40])
            except Exception as e:
                logger.warning("Get columns of table %s failed: %s", table_name, e)
            catalog.append(line)
        return "\n".join(f"{index + 1}. {item}" for index, item in enumerate(catalog))

    async def _detect_database_intent(self, question: str) -> bool:
        """调用 LLM 判断用户问题是否需要查库。"""
        brands = (self._database_brands or "").strip()
        if not brands:
            brands = "未配置（请在该算子参数中填写本库包含的品牌/产品列表，逗号分隔）"
        prompt = self._prompt_template.format(
            db_name=self._datasource._db_name,
            table_catalog=await self._build_table_catalog(),
            brands=brands,
        )
        models = await self.llm_client.models()
        if not models:
            raise ValueError("No models available.")
        request = ModelRequest.build_request(
            self._model or models[0].model,
            messages=[
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt),
                ModelMessage(
                    role=ModelMessageRoleType.HUMAN,
                    content=f"用户问题：{question}",
                ),
            ],
        )
        output: ModelOutput = await self.llm_client.generate(request)
        text = output.text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Can not parse LLM output: {text}")
        result = json.loads(text[start : end + 1])
        intent = result.get(_DATABASE_INTENT_KEY)
        if not isinstance(intent, bool):
            raise ValueError(f"Invalid database intent output: {text}")
        return intent

    async def branches(
        self,
    ) -> Dict[BranchFunc[AgentGenerateContext], BranchTaskType]:
        """根据 LLM 判断结果路由：数据库分支 / 百度搜索分支。"""
        database_task_name = self._database_task_name
        web_search_task_name = self._web_search_task_name
        if not database_task_name or not web_search_task_name:
            for node in self.downstream:
                if isinstance(node, HOSchemaLinkingAgentOperator):
                    database_task_name = node.node_name
                elif not database_task_name:
                    database_task_name = node.node_name
                else:
                    web_search_task_name = node.node_name
        if not database_task_name or not web_search_task_name:
            raise ValueError(
                "Agent Database Branch Operator requires database and web-search "
                "downstream nodes."
            )

        # 共享判断：两个 predicate 会被并行执行，只调用一次 LLM
        lock = asyncio.Lock()
        state = {"done": False, "value": False}

        async def judge(question: str) -> bool:
            if not state["done"]:
                async with lock:
                    if not state["done"]:
                        state["value"] = await self._detect_database_intent(question)
                        state["done"] = True
            return state["value"]

        async def is_database_request(input_value: AgentGenerateContext) -> bool:
            question = input_value.message.content if input_value.message else ""
            return await judge(question or "")

        async def is_web_search_request(input_value: AgentGenerateContext) -> bool:
            question = input_value.message.content if input_value.message else ""
            return not await judge(question or "")

        return {
            is_database_request: database_task_name,
            is_web_search_request: web_search_task_name,
        }
