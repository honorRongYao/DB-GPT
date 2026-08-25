"""Data Scientist Agent."""

import json
import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

from dbgpt.core import (
    ModelMessage,
    ModelMessageRoleType,
    ModelRequest,
)

from ..core.agent import AgentMessage
from ..core.base_agent import ConversableAgent
from ..core.profile import DynConfig, ProfileConfig
from ..resource.database import DBResource
from .actions.chart_action import ChartAction

logger = logging.getLogger(__name__)


class DataScientistAgent(ConversableAgent):
    """Data Scientist Agent."""

    profile: ProfileConfig = ProfileConfig(
        name=DynConfig(
            "Edgar",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_name",
        ),
        role=DynConfig(
            "DataScientist",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_role",
        ),
        goal=DynConfig(
            "Use correct {{dialect}} SQL to analyze and resolve user "
            "input targets based on the data structure information of the "
            "database given in the resource.",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_goal",
        ),
        constraints=DynConfig(
            [
                "Please ensure that the output is in the required format. "
                "Please ensure that each analysis only outputs one analysis "
                "result SQL, including as much analysis target content as possible.",
                "If there is a recent message record, pay attention to refer to "
                "the answers and execution results inside when analyzing, "
                "and do not generate the same wrong answer.Please check carefully "
                "to make sure the correct SQL is generated. Please strictly adhere "
                "to the data structure definition given. The use of non-existing "
                "fields is prohibited. Be careful not to confuse fields from "
                "different tables, and you can perform multi-table related queries.",
                "If the data and fields that need to be analyzed in the target are in "
                "different tables, it is recommended to use multi-table correlation "
                "queries first, and pay attention to the correlation between multiple "
                "table structures.",
                "It is prohibited to construct data yourself as query conditions. "
                "Only the data values given by the famous songs in the input can "
                "be used as query conditions.",
                "Please select an appropriate one from the supported display methods "
                "for data display. If no suitable display type is found, "
                "use 'response_table' as default value. Supported display types: \n"
                "{{ display_type }}",
            ],
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_constraints",
        ),
        desc=DynConfig(
            "Use database resources to conduct data analysis, analyze SQL, and provide "
            "recommended rendering methods.",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_desc",
        ),
    )

    max_retry_count: int = 5
    language: str = "zh"

    def __init__(self, **kwargs):
        """Create a new DataScientistAgent instance."""
        super().__init__(**kwargs)
        self._init_actions([ChartAction])

    def _init_reply_message(
        self,
        received_message: AgentMessage,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> AgentMessage:
        # 保存用户问题，供 correctness_check 做 LLM 语义校验时使用
        self._current_question = received_message.content or ""
        reply_message = super()._init_reply_message(received_message, rely_messages)
        reply_message.context = {
            "display_type": self.actions[0].render_prompt(),
            "dialect": self.database.dialect,
        }
        return reply_message

    @property
    def database(self) -> DBResource:
        """Get the database resource."""
        dbs: List[DBResource] = DBResource.from_resource(self.resource)
        if not dbs:
            raise ValueError(
                f"Resource type {self.actions[0].resource_need} is not supported."
            )
        return dbs[0]

    async def correctness_check(
        self, message: AgentMessage
    ) -> Tuple[bool, Optional[str]]:
        """Verify whether the current execution results meet the target expectations."""
        action_out = message.action_report
        if action_out is None:
            return (
                False,
                f"No executable analysis SQL is generated,{message.content}.",
            )

        if not action_out.is_exe_success:
            return (
                False,
                f"Please check your answer, {action_out.content}.",
            )
        action_reply_obj = json.loads(action_out.content)
        sql = action_reply_obj.get("sql", None)
        if not sql:
            return (
                False,
                "Please check your answer, the sql information that needs to be "
                "generated is not found.",
            )
        try:
            if not action_out.resource_value:
                return (
                    False,
                    "Please check your answer, the data resource information is not "
                    "found.",
                )

            columns, values = await self.database.query(
                sql=sql,
                db=action_out.resource_value,
            )
            if not values or len(values) <= 0:
                return (
                    False,
                    "Please check your answer, the current SQL cannot find the data to "
                    "determine whether filtered field values or inappropriate filter "
                    "conditions are used.",
                )
            else:
                logger.info(
                    f"reply check success! There are {len(values)} rows of data"
                )
                # 增强自检：执行结果非空之外，再用 LLM 校验结果是否完整满足用户问题要求
                question = getattr(self, "_current_question", "") or ""
                if question:
                    check_ok, check_reason = await self._llm_result_check(
                        question, sql, values
                    )
                    if not check_ok:
                        return False, check_reason
                return True, None
        except Exception as e:
            logger.exception(f"DataScientist check exception！{str(e)}")
            return (
                False,
                f"SQL execution error, please re-read the historical information to "
                f"fix this SQL. The error message is as follows:{str(e)}",
            )

    async def _llm_result_check(
        self, question: str, sql: str, values: List[Any]
    ) -> Tuple[bool, Optional[str]]:
        """用 LLM 校验 SQL 与执行结果是否完整满足用户问题的要求。

        在"执行结果非空"的基础上，进一步检查结果"对不对"：
        例如要求 TOP3 但 SQL 没有取前3、枚举类过滤值该用 LIKE 却用了 = 等。
        LLM 不可用或校验结果解析失败时按"通过"处理，不阻塞主流程。
        """
        rows_preview = "\n".join(str(row) for row in values[:20])
        sys_prompt = (
            "你是一个严谨的数据分析结果校验员。用户提出一个数据分析问题，"
            "数据分析 Agent 生成了 SQL 并执行出结果。请判断执行结果是否完整满足"
            "用户问题的所有要求，重点检查：\n"
            "1. 是否遗漏问题的量化要求（例如要求 TOP3/前N/排名，SQL 是否真的用"
            "窗口函数或 LIMIT 取了前N）；\n"
            "2. WHERE 过滤条件是否合理（枚举类取值是否模糊匹配、过滤层级/口径是否"
            "与问题一致）；\n"
            "3. 聚合、分组、排序是否与问题要求一致；\n"
            "4. 结果字段是否覆盖问题关心的所有维度；\n"
            "5. 时间范围是否合理：如果问题包含相对时间描述（如近3个月、最近30天、"
            "上月、今年以来），SQL 的日期过滤范围是否与当前日期推算一致，"
            "日期写错会导致取到错误时段的数据。\n"
            "只输出严格 JSON：{\"pass\": true 或 false, \"reason\": \"不通过时的具体原因，"
            "指出缺了什么、如何修改（用中文）\"}"
        )
        human = (
            f"当前日期：{datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"用户问题：{question}\n\n"
            f"生成的 SQL：\n{sql}\n\n"
            f"执行结果（前 {min(len(values), 20)} 行）：\n{rows_preview}"
        )
        try:
            llm_client = self.not_null_llm_client
            models = await llm_client.models()
            if not models:
                return True, None
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=sys_prompt),
                ModelMessage(role=ModelMessageRoleType.HUMAN, content=human),
            ]
            request = ModelRequest.build_request(models[0].model, messages=messages)
            output = await llm_client.generate(request)
            text = (output.text or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                return True, None
            result = json.loads(text[start : end + 1])
            if not result.get("pass", True):
                reason = str(result.get("reason") or "结果未通过语义校验")
                logger.info(f"LLM result check not pass: {reason}")
                return False, reason
        except Exception as e:
            logger.warning(f"LLM result check failed, skip: {e}")
        return True, None
