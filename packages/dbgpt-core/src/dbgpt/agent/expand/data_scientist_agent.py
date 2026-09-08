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

from ..core.action.base import ActionOutput
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
                "Generate exactly one complete and executable SQL statement in the "
                "required output format.",
                "Use only tables, columns, values, relationships, and business rules "
                "provided in the resource information. Never invent them.",
                "Determine filters, time range, grain, grouping, aggregation, ordering, "
                "ranking, and output fields from the user's question. Prevent one-to-many "
                "joins from causing duplicate counts.",
                "Implement TOP N, per-group ranking, trends, differences, rates, and "
                "proportions directly in SQL when requested. Preserve the semantic order "
                "and scope of each step in multi-step questions.",
                "The display_type field is rendering metadata only. Do not use it for SQL "
                "reasoning or conclusions. The thought must objectively describe the SQL "
                "calculation logic without inventing data conclusions. Supported display "
                "types: \n{{ display_type }}",
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

    async def read_memories(self, question: str) -> str:
        """Do not load long-term memories for SQL generation."""
        return ""

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
        # AgentMessage.success 默认 True，会让"重试纠错反馈（Human 行）"和
        # "未通过校验的中间回复（DS 行）"在写入 gpts_messages 时 is_success=1，
        # 与真实校验结果不符。这里统一先置 False：中间消息如实记录 is_success=0，
        # 最终成败由 base_agent.generate_reply 循环结束时按 verify 结果赋给最后一条消息。
        reply_message.success = False
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
        # action_report.content 理论上是合法 JSON，但一旦不是（如 LLM 输出畸形），
        # json.loads/非 dict 的 .get 会抛异常，直接冲出 correctness_check 被
        # generate_reply 最外层 except 捕获，导致整轮重试中断且原因不可读。
        # 这里单独兜底，按普通校验失败返回明确原因，让重试机制正常工作。
        try:
            action_reply_obj = json.loads(action_out.content or "")
        except Exception:
            return (
                False,
                "Please check your answer, the output content is not valid JSON, "
                "please regenerate a reply strictly in the required format.",
            )
        if not isinstance(action_reply_obj, dict):
            return (
                False,
                "Please check your answer, the output content is not a valid JSON "
                "object, please regenerate a reply strictly in the required format.",
            )
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
                error_desc = (
                    "Please check your answer, the current SQL cannot find the data to "
                    "determine whether filtered field values or inappropriate filter "
                    "conditions are used."
                )
                question = getattr(self, "_current_question", "") or ""
                if question:
                    # 让大模型判断错因并给出改造建议，作为重试反馈写入 gpts_messages，
                    # 而不是只存一句静态文案，否则 Agent 重试时不知道具体改哪里
                    return False, await self._analyze_failed_result(
                        question, sql, error_desc
                    )
                return False, error_desc
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
                    await self._replace_result_summary(
                        question=question,
                        sql=sql,
                        columns=columns,
                        values=values,
                        action_out=action_out,
                        action_reply_obj=action_reply_obj,
                    )
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
            "数据分析 Agent 生成的 SQL 已真实执行且返回了非空结果。"
            "请判断执行结果是否完整满足用户问题的所有要求。\n"
            "判定原则：\n"
            "1. 只依据『用户问题、SQL、执行结果』三者的可验证事实判断，"
            "必须能明确指出具体问题点才能判不通过（fail）；\n"
            "2. SQL 已真实执行且结果非空，本身就是基础正确性的证据；"
            "数值大小（如总数很大/很小）不能作为失败依据，不能仅凭数据量级怀疑结果；\n"
            "3. 无法从现有信息确定存在错误时，判通过（pass）；\n"
            "4. 禁止输出模糊怀疑，如『可能需要核实品类』『可能存在遗漏』"
            "『需进一步确认』这类没有明确问题点的理由，一律视为通过。\n"
            "重点检查（仅限可明确判定的硬性要求）：\n"
            "1. 问题要求 TOP3/前N/排名时，SQL 是否真正用窗口函数或 LIMIT 取了前N，"
            "若没有取前N则明确指出并给出改法；\n"
            "2. 问题包含相对时间（近3个月/最近30天/上月/今年以来/Q1等）时，"
            "SQL 的时间过滤范围是否与当前日期推算一致，若明显不符则明确指出正确范围；\n"
            "3. WHERE 过滤是否明显缺失问题指定的限定（如问题限定品类/品牌/渠道，"
            "SQL 却完全没有对应过滤条件）；\n"
            "4. 聚合、分组、排序是否与问题要求一致（如问题要求按品牌分组统计，"
            "SQL 却没有分组）；是否正确处理一对多关联，避免重复计数；\n"
            "5. 将用户问题拆分为独立要求，检查 SQL 是否逐项实现，不得仅因 SQL 可执行且"
            "结果非空就判定通过；检查每个过滤条件和时间条件是否作用在正确的数据步骤上；\n"
            "6. 检查‘各自、分别、每个、各组’是否按对应实体或维度分别计算，不能错误合并"
            "为全局结果；检查‘最近N个周期’是否严格取事实表中最近N个不同周期；\n"
            "7. SQL 返回字段是否覆盖用户要求的全部维度、指标和计算结果；趋势、差值、变化率、"
            "排名或占比等派生指标如未在 SQL 中直接计算并返回，应判定不通过；\n"
            "8. SQL 中的每个实体、过滤值、表、字段、关联和业务规则是否能在资源或 "
            "Schema Linking 提供的信息中找到；禁止因结果非空就放过未提供的字段、关系或"
            "违反资源中业务规则的计算逻辑。\n"
            "只输出严格 JSON：{\"pass\": true 或 false, \"reason\": \"不通过时必须明确指出"
            "问题点并给出修改方法（中文）；通过时 reason 留空字符串\"}"
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

    async def _replace_result_summary(
        self,
        question: str,
        sql: str,
        columns: List[str],
        values: List[Any],
        action_out: ActionOutput,
        action_reply_obj: dict,
    ) -> None:
        """Generate a concise conclusion from real query results for display."""
        displayed_rows = action_reply_obj.get("data")
        if not isinstance(displayed_rows, list):
            displayed_rows = [dict(zip(columns, row)) for row in values]
        row_count = int(action_reply_obj.get("count") or len(displayed_rows))
        rows_preview = displayed_rows[:50]
        user_question = question.rsplit("用户问题:\n", 1)[-1].strip()
        sys_prompt = (
            "你是严谨的数据分析总结专家。请根据用户问题、已执行 SQL 和真实查询结果，"
            "生成面向用户的简洁中文分析结论。\n"
            "规则：\n"
            "1. 只能使用提供的真实结果，不得虚构、推测或补充结果中不存在的信息；\n"
            "2. 直接回答用户问题，优先概括关键数值、排名、差异、趋势和异常；\n"
            "3. 单值结果应明确说明指标值；多行结果应总结最重要发现，不要逐行复述；\n"
            "4. 必须区分结果总行数与当前提供的预览行数；不得将预览误称为完整结果；\n"
            "5. 不评价 SQL，不描述生成过程，不输出 Markdown 表格；\n"
            "6. 只输出严格 JSON：{\"thought\": \"简洁、客观的中文分析结论\"}。"
        )
        human = (
            f"用户问题：{user_question}\n\n"
            f"已执行 SQL：\n{sql}\n\n"
            f"结果总行数：{row_count}\n"
            f"真实结果预览（最多50行）：\n"
            f"{json.dumps(rows_preview, ensure_ascii=False, default=str)}"
        )
        try:
            llm_client = self.not_null_llm_client
            models = await llm_client.models()
            if not models:
                return
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=sys_prompt),
                ModelMessage(role=ModelMessageRoleType.HUMAN, content=human),
            ]
            request = ModelRequest.build_request(models[0].model, messages=messages)
            output = await llm_client.generate(request)
            text = (output.text or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                return
            result = json.loads(text[start : end + 1])
            thought = str(result.get("thought") or "").strip()
            if not thought:
                return

            action_reply_obj["thought"] = thought
            action_out.content = json.dumps(action_reply_obj, ensure_ascii=False)

            chart = {
                "display_type": action_reply_obj.get(
                    "display_type", "response_table"
                ),
                "sql": sql,
                "thought": thought,
            }
            import pandas as pd

            action_out.view = await self.actions[0].render_protocol.display(
                chart=chart,
                data_df=pd.DataFrame(displayed_rows),
            )
        except Exception as e:
            logger.warning(f"Replace result summary failed, keep original thought: {e}")

    async def _analyze_failed_result(
        self, question: str, sql: str, error_desc: str
    ) -> str:
        """让大模型分析 SQL 校验失败的原因并给出改造建议。

        返回的文本会作为失败原因写入 gpts_messages（重试反馈），
        并在重试时作为输入指导 Agent 修正 SQL。
        大模型不可用或输出解析失败时，回退到 error_desc 原文。
        """
        sys_prompt = (
            "你是一个严谨的数据分析专家。数据分析 Agent 生成了一条 SQL，"
            "但执行后校验失败。请结合用户问题、SQL 和失败原因，判断 SQL 错在哪里，"
            "并给出具体可执行的改造建议。重点排查：\n"
            "1. WHERE 过滤条件是否过严或写错（枚举值应模糊匹配却写成精确匹配、"
            "大小写、空格、中文/英文差异）；\n"
            "2. 时间范围口径是否与问题一致（近3个月/最近30天/上月/今年以来等"
            "相对时间是否推算正确，日期写错会导致取到错误时段或空数据）；\n"
            "3. 是否选错了表或字段；\n"
            "4. 多表关联条件是否正确，是否因关联错误导致结果为空；\n"
            "5. 聚合、分组、排序是否与问题要求一致。\n"
            "只输出严格 JSON：{\"judgment\": \"对错误的判断\", \"suggestion\": \"具体改造建议（中文）\"}"
        )
        human = (
            f"用户问题：{question}\n\n"
            f"生成的 SQL：\n{sql}\n\n"
            f"失败原因：{error_desc}"
        )
        try:
            llm_client = self.not_null_llm_client
            models = await llm_client.models()
            if not models:
                return error_desc
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=sys_prompt),
                ModelMessage(role=ModelMessageRoleType.HUMAN, content=human),
            ]
            request = ModelRequest.build_request(models[0].model, messages=messages)
            output = await llm_client.generate(request)
            text = (output.text or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                return error_desc
            result = json.loads(text[start : end + 1])
            judgment = str(result.get("judgment") or "").strip()
            suggestion = str(result.get("suggestion") or "").strip()
            if not judgment and not suggestion:
                return error_desc
            parts = []
            if judgment:
                parts.append(f"校验失败原因：{judgment}")
            if suggestion:
                parts.append(f"改造建议：{suggestion}")
            return "；".join(parts)
        except Exception as e:
            logger.warning(f"Analyze failed result error, skip: {e}")
            return error_desc
