"""Agent workflow operators for database and web-search routing."""

import asyncio
import json
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

_DATABASE_INTENT_KEY = "database_intent"
_DEFAULT_INTENT_PROMPT = """你是一个请求路由器。只判断用户问题是否属于“查数操作”，不要判断数据库中是否存在对应数据，也不要选表或生成 SQL。

判断规则：
1. 用户要求查询、统计、筛选、汇总、对比、排名或查看业务数据、指标、记录、明细时，database_intent 为 true。
2. 即使问题中的公司、品牌、产品或品类可能不在数据库中，只要用户意图是查数，仍必须返回 true；数据库是否能查到由后续 Text2SQL 链路处理。
3. 用户未指明公司时，不影响判断；只判断是不是查数操作。
4. 新闻、百科、常识、天气、政策解读、行业资讯、开放式知识问答、原因分析、建议或其他不需要查询结构化业务数据的问题，database_intent 为 false，走百度搜索。
5. 严格输出 JSON，不要附加解释：{"database_intent": true} 或 {"database_intent": false}。

参考示例：
- "2025年12月门锁的销量" -> {"database_intent": true}
- "美的公司今年营收是多少" -> {"database_intent": true}
- "查询每个用户的订单总金额" -> {"database_intent": true}
- "空气炸锅销量排名" -> {"database_intent": true}
- "2026年顺德最大的电器公司是哪家" -> {"database_intent": false}
- "空气炸锅为什么受欢迎" -> {"database_intent": false}
- "今天北京的天气怎么样" -> {"database_intent": false}
"""


class AgentDatabaseBranchOperator(MixinLLMOperator, AgentBranchOperator):
    """基于内置智能体分支算子的数据库路由算子。

    在 branches() 内让 LLM 仅判断用户问题是否属于查数操作：
    查数操作 -> 原 Text2SQL 的语义层（Schema Linking）分支；否则 -> 百度搜索分支。
    相比内置 AgentBranchOperator（只按 next_speakers 路由到 Agent 下游），
    本算子支持把请求路由到非 Agent 下游（如语义层算子）。
    """

    metadata = ViewMetadata(
        label=_("Agent Database Branch Operator"),
        name="agent_database_branch_operator",
        description=_(
            "由 LLM 判断用户问题是否属于查数操作：查数则进入原 Text2SQL 的"
            "语义层（Schema Linking）分支，否则进入百度搜索分支。"
        ),
        category=OperatorCategory.AGENT,
        operator_type=OperatorType.BRANCH,
        parameters=[
            Parameter.build_from(
                _("Datasource"),
                "datasource",
                type=DBResource,
                description=_("数据库分支使用的数据源；路由阶段不会检查其中是否存在数据。"),
            ),
            Parameter.build_from(
                _("Intent Prompt Template"),
                "prompt_template",
                type=str,
                optional=True,
                default=_DEFAULT_INTENT_PROMPT,
                description=_("查数意图判断提示词；只判断是否查数，不判断数据是否存在。"),
            ),
            Parameter.build_from(
                _("Database Brands"),
                "database_brands",
                type=str,
                optional=True,
                default=None,
                description=_("兼容旧工作流保留；当前查数意图路由不使用此参数。"),
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
            datasource: 数据库分支使用的数据源（为兼容现有工作流保留）
            prompt_template: 查数意图判断提示词模板
            database_brands: 兼容旧工作流保留，当前不参与路由判断
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

    async def _detect_database_intent(self, question: str) -> bool:
        """只判断是否属于查数操作，不检查数据库是否存在对应数据。"""
        prompt = self._prompt_template or _DEFAULT_INTENT_PROMPT
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
