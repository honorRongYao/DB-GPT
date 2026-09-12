"""Data Scientist Agent."""

import json
import logging
import os
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

# 结论生成的内置兜底提示词：skills/user/result-summary/SKILL.md 缺失或解析失败时使用
_DEFAULT_SUMMARY_PROMPT = (
    "你是资深商业数据分析师。请基于用户问题与真实查询结果，"
    "输出业务人员能直接看懂的中文分析结论。\n"
    "只使用给定数据，不得虚构或补充结果中不存在的信息；"
    "只使用结果中出现的字段，不得自造维度；必须区分结果总行数与预览行数。\n"
    "按以下四段输出，段间空一行，不要使用 Markdown 语法（不要 **、##、表格），"
    "涨跌用 ↑ ↓ 标注：\n"
    "📌 结论\n一句话直接回答用户问题，带最关键的数字，不超过60字。\n"
    "🔍 关键发现\n用 1. 2. 3. 编号，按重要性排序只保留2-4条，"
    "每条为「现象+数字+对比+业务含义」。\n"
    "💡 建议行动\n用 1. 2. 编号，只给2-3条可直接执行的动作。\n"
    "⚠️ 数据说明\n一到两句，只写口径、局限或置信度，无歧义时可省略。\n"
    "不评价 SQL，不描述生成过程。只输出严格 JSON，且只能有 thought 一个键"
    "（禁止 output、reasoning、answer 等其他键）："
    "{\"thought\": \"四段结论\"}，换行用 \\n 转义。"
)

# 结论生成所用的 skill（改文件即可调整结论质量，无需改代码）
_SUMMARY_SKILL_RELATIVE_PATH = os.path.join("user", "result-summary", "SKILL.md")


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
                "Output column naming (mandatory): every column and aggregate "
                "expression in the SELECT list MUST be given a Chinese alias with AS, "
                "for example SELECT dt AS 月份, SUM(units_month) AS 销量, "
                "COUNT(*) AS 评论数. Alias rules: only Chinese characters and digits, "
                "no spaces, no punctuation, and do NOT wrap the alias in any quotes "
                "(quotes behave differently across dialects). Never let a raw database "
                "field name appear as a result column name.",
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
        # 当前重试轮次（0 基），由 _load_thinking_messages 每轮刷新
        self._current_retry: int = 0

    async def read_memories(self, question: str) -> str:
        """Do not load long-term memories for SQL generation."""
        return ""

    async def _load_thinking_messages(
        self, *args, **kwargs
    ) -> Tuple[List[AgentMessage], Optional[Any]]:
        """记录当前重试轮次（0 基），供校验失败时判断是否已是最后一轮。"""
        self._current_retry = kwargs.get("current_retry_counter") or 0
        return await super()._load_thinking_messages(*args, **kwargs)

    def _is_last_round(self) -> bool:
        """当前是否已是重试循环的最后一轮（最后一轮失败将不再重试）。"""
        return getattr(self, "_current_retry", 0) >= self.max_retry_count - 1

    async def _finalize_unverified(
        self,
        reason: str,
        sql: Optional[str] = None,
        action_out: Optional[ActionOutput] = None,
        action_reply_obj: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """最后一轮仍未通过校验时的兜底收口。

        不再返回失败（否则循环结束、前端空白），而是保留已有结果数据、
        把简短提示写进结论，按"校验通过"返回，供前端展示"结果 + 免责说明"。
        """
        # 面向用户只给简短提示：不拼接审核返回的技术性理由（含"修改方法"等
        # 给 Agent 的重试指令），完整原因仅记日志，便于排查。
        warn = "以下结果未经校验通过，可能不满足问题要求，仅供参考。"
        if action_out is not None:
            try:
                obj = action_reply_obj if isinstance(action_reply_obj, dict) else {}
                obj["thought"] = warn
                action_out.content = json.dumps(obj, ensure_ascii=False)
                data = obj.get("data")
                if not isinstance(data, list):
                    data = []
                chart = {
                    "display_type": obj.get("display_type", "response_table"),
                    "sql": sql or "",
                    "thought": warn,
                }
                import pandas as pd

                action_out.view = await self.actions[0].render_protocol.display(
                    chart=chart,
                    data_df=pd.DataFrame(data),
                )
            except Exception as e:
                logger.warning(f"Finalize unverified result failed, skip: {e}")
        logger.info(f"Last round not pass, finalize with reason: {reason}")
        return True, None

    async def _check_fail(
        self,
        reason: str,
        sql: Optional[str] = None,
        action_out: Optional[ActionOutput] = None,
        action_reply_obj: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """校验失败的统一出口：非最后一轮返回失败触发重试，最后一轮兜底收口。"""
        if self._is_last_round():
            return await self._finalize_unverified(
                reason,
                sql=sql,
                action_out=action_out,
                action_reply_obj=action_reply_obj,
            )
        return False, reason

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
        # AgentMessage.success 默认 True，会让"重试纠错反馈（retry_message）"和
        # "未通过校验的中间回复"在写入 gpts_messages 时 is_success=1，与真实校验结果不符。
        # 这里仍先置 False：
        # - retry_message（base_agent.generate_reply 每轮开头创建并 send）在本轮 verify
        #   之前就会落库，只能靠这里保证 is_success=0；
        # - reply_message 则保证首轮 verify 前的中间态不为成功。
        # 每轮 verify 完成后，base_agent.generate_reply 会按 check_pass 覆盖同步
        # reply_message.success（见 base_agent.py generate_reply），循环结束再由
        # is_success 给最后一条消息统一定稿。
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
            return await self._check_fail(
                f"Please check your answer, {action_out.content}.",
                action_out=action_out,
            )
        # action_report.content 理论上是合法 JSON，但一旦不是（如 LLM 输出畸形），
        # json.loads/非 dict 的 .get 会抛异常，直接冲出 correctness_check 被
        # generate_reply 最外层 except 捕获，导致整轮重试中断且原因不可读。
        # 这里单独兜底，按普通校验失败返回明确原因，让重试机制正常工作。
        try:
            action_reply_obj = json.loads(action_out.content or "")
        except Exception:
            return await self._check_fail(
                "Please check your answer, the output content is not valid JSON, "
                "please regenerate a reply strictly in the required format.",
                action_out=action_out,
            )
        if not isinstance(action_reply_obj, dict):
            return await self._check_fail(
                "Please check your answer, the output content is not a valid JSON "
                "object, please regenerate a reply strictly in the required format.",
                action_out=action_out,
            )
        sql = action_reply_obj.get("sql", None)
        if not sql:
            return await self._check_fail(
                "Please check your answer, the sql information that needs to be "
                "generated is not found.",
                action_out=action_out,
                action_reply_obj=action_reply_obj,
            )
        try:
            if not action_out.resource_value:
                return await self._check_fail(
                    "Please check your answer, the data resource information is not "
                    "found.",
                    sql=sql,
                    action_out=action_out,
                    action_reply_obj=action_reply_obj,
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
                    # 让大模型判断是"SQL 写错"还是"SQL 没错、该口径确实无数据"：
                    # - 写错：反馈改造建议作为重试输入，指导 Agent 修正 SQL；
                    # - 没错：按校验通过处理，把无数据原因写回结果供前端展示，
                    #   避免无谓重试后整轮被判失败。
                    sql_error, feedback = await self._analyze_failed_result(
                        question, sql, error_desc
                    )
                    if not sql_error:
                        await self._replace_empty_result_reason(
                            sql=sql,
                            reason=feedback,
                            action_out=action_out,
                            action_reply_obj=action_reply_obj,
                        )
                        logger.info(
                            "reply check success: sql is correct but no data for this "
                            "scope"
                        )
                        return True, None
                    return await self._check_fail(
                        feedback,
                        sql=sql,
                        action_out=action_out,
                        action_reply_obj=action_reply_obj,
                    )
                return await self._check_fail(
                    error_desc,
                    sql=sql,
                    action_out=action_out,
                    action_reply_obj=action_reply_obj,
                )
            else:
                logger.info(
                    f"reply check success! There are {len(values)} rows of data"
                )
                # 增强自检：执行结果非空之外，再用 LLM 校验结果是否完整满足用户问题要求
                question = getattr(self, "_current_question", "") or ""
                if question:
                    check_ok, check_reason = await self._llm_result_check(
                        question, sql, columns, values
                    )
                    if not check_ok:
                        return await self._check_fail(
                            check_reason,
                            sql=sql,
                            action_out=action_out,
                            action_reply_obj=action_reply_obj,
                        )
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
            return await self._check_fail(
                f"SQL execution error, please re-read the historical information to "
                f"fix this SQL. The error message is as follows:{str(e)}",
                sql=sql,
                action_out=action_out,
                action_reply_obj=action_reply_obj,
            )

    @staticmethod
    def _split_question_and_schema(question: str) -> Tuple[str, str]:
        """把语义层注入的内容拆成 (用户问题, 表结构上下文)。

        语义层会把"允许表及关联 + 完整表结构（含取值参考）+ 召回 SQL"注入在真正的
        用户问题之前，用最后一个"用户问题:"作分界：前半段是表结构等上下文，后半段
        才是真正的问题；独立运行（无注入，content 就是纯问题）时拆不开，则整体当作
        问题、表结构为空。
        """
        user_question = (question or "").strip()
        schema_context = ""
        marker = "用户问题:"
        marker_idx = question.rfind(marker) if question else -1
        if marker_idx != -1:
            candidate = question[marker_idx + len(marker) :].strip()
            if candidate:
                schema_context = question[:marker_idx].strip()
                user_question = candidate
        return user_question, schema_context

    @staticmethod
    def _to_bool(value: Any, default: bool) -> bool:
        """把模型输出的布尔字段解析成 bool；缺失/None 时取 default。

        兼容模型把布尔写成字符串（如 "true"/"false"）的情况。
        """
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "否")
        return bool(value)

    async def _call_llm_and_parse_json(
        self, sys_prompt: str, human: str
    ) -> Optional[dict]:
        """调用大模型并解析输出中的 JSON 对象；任何环节失败都返回 None。

        统一负责：取模型 → 组装 messages → 请求 → 截取首个 '{' 到末个 '}' →
        json.loads。调用方只需处理"解析成功/失败"两种情况，并按各自策略兜底。
        """
        try:
            llm_client = self.not_null_llm_client
            models = await llm_client.models()
            if not models:
                return None
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=sys_prompt),
                ModelMessage(role=ModelMessageRoleType.HUMAN, content=human),
            ]
            request = ModelRequest.build_request(models[0].model, messages=messages)
            output = await llm_client.generate(request)
            text = (output.text or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                return None
            result = json.loads(text[start : end + 1])
            return result if isinstance(result, dict) else None
        except Exception as e:
            logger.warning(f"Call llm and parse json failed, skip: {e}")
            return None

    async def _llm_result_check(
        self, question: str, sql: str, columns: List[str], values: List[Any]
    ) -> Tuple[bool, Optional[str]]:
        """用 LLM 校验 SQL 与执行结果是否完整满足用户问题的要求。

        在"执行结果非空"的基础上，进一步检查结果"对不对"：
        例如要求 TOP3 但 SQL 没有取前3、枚举类过滤值该用 LIKE 却用了 = 等。
        判定从严把握"错才判错"：只有能明确指出具体问题点、并给出可执行改法时才判
        不通过；否则一律判通过。LLM 不可用或校验结果解析失败时按"通过"处理，
        不阻塞主流程。
        """
        user_question, schema_context = self._split_question_and_schema(question)
        preview_values = values[:20]
        if columns:
            # 带列名输出，模型才能判断结果字段是否覆盖用户要求的维度/指标
            rows_preview = "\n".join(
                json.dumps(dict(zip(columns, row)), ensure_ascii=False, default=str)
                for row in preview_values
            )
        else:
            rows_preview = "\n".join(str(row) for row in preview_values)
        sys_prompt = (
            "你是一个严谨的数据分析结果校验员。用户提出一个数据分析问题，"
            "数据分析 Agent 生成的 SQL 已真实执行且返回了非空结果。"
            "请结合用户问题、SQL、执行结果，以及下方提供的表结构/取值参考（若提供），"
            "判断执行结果是否完整满足用户问题的所有要求。\n"
            "判定原则：\n"
            "1. 只依据『用户问题、SQL、执行结果』三者的可验证事实判断，"
            "必须能明确指出具体问题点才能判不通过（fail）；\n"
            "2. SQL 已真实执行且结果非空，本身就是基础正确性的证据；"
            "数值大小（如总数很大/很小）不能作为失败依据，不能仅凭数据量级怀疑结果；\n"
            "3. 无法从现有信息确定存在错误时，判通过（pass）；"
            "只提出疑问、给不出可执行的具体改法时，一律判通过；\n"
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
            "7. 结合结果字段名，检查 SQL 返回字段是否覆盖用户要求的全部维度、指标和"
            "计算结果；趋势、差值、变化率、排名或占比等派生指标如未在 SQL 中直接计算"
            "并返回，应判定不通过；\n"
            "8. SQL 中的每个实体、过滤值、表、字段、关联和业务规则是否能在下方表结构中"
            "找到；禁止因结果非空就放过未提供的字段、关系或违反表结构中业务规则的计算"
            "逻辑。\n"
            "输出要求：只有当能明确指出问题点并给出可执行的修改方法时才判 pass=false；"
            "否则 pass=true 且 reason 留空字符串。\n"
            "只输出严格 JSON：{\"pass\": true 或 false, \"reason\": \"不通过时必须明确指出"
            "问题点并给出修改方法（中文）；通过时 reason 留空字符串\"}"
        )
        human_parts = [
            f"当前日期：{datetime.now().strftime('%Y-%m-%d')}",
            f"用户问题：{user_question}",
            f"生成的 SQL：\n{sql}",
            f"执行结果（共 {len(values)} 行，预览前 {len(preview_values)} 行）：\n"
            f"{rows_preview}",
        ]
        if schema_context:
            human_parts.append(f"表结构/关联关系/取值参考：\n{schema_context}")
        human = "\n\n".join(human_parts)
        result = await self._call_llm_and_parse_json(sys_prompt, human)
        if result is None:
            # 模型不可用/输出不可解析：按通过处理，不阻塞主流程
            return True, None
        # 缺失或为 null 时按"通过"处理，与"无法确定就通过"的既定策略保持一致
        if self._to_bool(result.get("pass"), True):
            return True, None
        reason = str(result.get("reason") or "").strip()
        if not reason:
            # 判不通过却给不出具体问题点，无法形成有效修改建议，按通过处理
            return True, None
        logger.info(f"LLM result check not pass: {reason}")
        return False, reason

    def _load_summary_skill_prompt(self) -> Optional[str]:
        """读取结论生成 skill 的提示词：SKILL.md 正文 + references 下的参考文件。

        提示词以 skill 文件形式维护，调整结论质量只需改 md 文件，不必改代码。
        references 目录下的 md 全量拼入 system prompt（当前约 5k 字符）。
        文件缺失或格式不合法时返回 None，由调用方回退到内置提示词。
        """
        try:
            from ...configs.model_config import SKILLS_DIR
            from ..claude_skill import FileBasedSkill

            skill_path = os.path.join(SKILLS_DIR, _SUMMARY_SKILL_RELATIVE_PATH)
            instructions = FileBasedSkill(skill_path).instructions.strip()
            if not instructions:
                return None

            parts = [instructions]
            ref_dir = os.path.join(os.path.dirname(skill_path), "references")
            if os.path.isdir(ref_dir):
                for name in sorted(os.listdir(ref_dir)):
                    if not name.endswith(".md"):
                        continue
                    with open(os.path.join(ref_dir, name), "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        parts.append(content)
            return "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"Load summary skill failed, use builtin prompt: {e}")
            return None

    async def _request_summary_json(
        self, sys_prompt: str, human: str
    ) -> Optional[dict]:
        """请求模型生成结论并解析 JSON 对象；任何环节失败都返回 None。"""
        try:
            llm_client = self.not_null_llm_client
            models = await llm_client.models()
            if not models:
                return None
            messages = [
                ModelMessage(role=ModelMessageRoleType.SYSTEM, content=sys_prompt),
                ModelMessage(role=ModelMessageRoleType.HUMAN, content=human),
            ]
            request = ModelRequest.build_request(models[0].model, messages=messages)
            output = await llm_client.generate(request)
            text = (output.text or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                return None
            result = json.loads(text[start : end + 1])
            return result if isinstance(result, dict) else None
        except Exception as e:
            logger.warning(f"Request summary json failed: {e}")
            return None

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
        sys_prompt = self._load_summary_skill_prompt() or _DEFAULT_SUMMARY_PROMPT
        human = (
            f"用户问题：{user_question}\n\n"
            f"已执行 SQL：\n{sql}\n\n"
            f"结果总行数：{row_count}\n"
            f"真实结果预览（最多50行）：\n"
            f"{json.dumps(rows_preview, ensure_ascii=False, default=str)}"
        )
        try:
            # 要求模型只输出 thought 单键；出现多余键（如 note）或缺少 thought
            # 时重试一次。两次仍不合规则退回"最接近的一次"（有 thought 就用），
            # 保证展示不因模型偶发多键而中断。
            thought, fallback = "", ""
            for attempt in range(2):
                result = await self._request_summary_json(sys_prompt, human)
                if not result:
                    continue
                current = str(result.get("thought") or "").strip()
                if current and not fallback:
                    fallback = current
                if current and set(result.keys()) == {"thought"}:
                    thought = current
                    break
                if attempt == 0:
                    logger.info(
                        f"Summary json keys unexpected: {sorted(result.keys())}, retry"
                    )
            thought = thought or fallback
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

    async def _replace_empty_result_reason(
        self,
        sql: str,
        reason: str,
        action_out: ActionOutput,
        action_reply_obj: dict,
    ) -> None:
        """SQL 正确但该口径确实无数据：把无数据原因写回结果，供前端展示。

        与 _replace_result_summary 不同，这里没有行数据，不做结果总结，
        只把 LLM 判断的"为什么没有数据"作为结论填充到 thought/describe。
        """
        reason = (reason or "").strip() or "该口径下没有查询到数据。"
        action_reply_obj["count"] = 0
        action_reply_obj["data"] = []
        action_reply_obj["thought"] = reason
        action_out.content = json.dumps(action_reply_obj, ensure_ascii=False)

        chart = {
            "display_type": action_reply_obj.get("display_type", "response_table"),
            "sql": sql,
            "thought": reason,
        }
        import pandas as pd

        action_out.view = await self.actions[0].render_protocol.display(
            chart=chart,
            data_df=pd.DataFrame(),
        )

    async def _analyze_failed_result(
        self, question: str, sql: str, error_desc: str
    ) -> Tuple[bool, str]:
        """让大模型分析 SQL 空结果的原因，返回 (sql_error, feedback)。

        - sql_error=True：SQL 存在导致取不到数据的错误，feedback 为
          "校验失败原因/改造建议"，调用方应作为重试反馈写入 gpts_messages，
          指导 Agent 修正 SQL；
        - sql_error=False：SQL 与该口径一致、确实没有数据，feedback 为无数据原因，
          调用方应按"校验通过"处理，把原因返回给前端展示，而不是继续重试。
        大模型不可用或输出解析失败时，按 sql_error=True 回退 error_desc 原文，
        保留原有重试行为。

        question 通常是语义层注入的整段内容（允许表及关联 + 完整表结构 + 召回 SQL +
        "用户问题:"），这里按标记拆开，把表结构作为独立上下文传给模型，避免与
        用户问题混在一起、也去掉重复的"用户问题"标记。
        """
        user_question, schema_context = self._split_question_and_schema(question)
        sys_prompt = (
            "你是一个严谨的数据分析专家。数据分析 Agent 生成的一条 SQL 已成功执行"
            "（语法、表、字段、关联都能跑通），但返回 0 行数据。请结合用户问题、SQL、"
            "失败原因，以及下方提供的表结构/取值参考（若提供），判断 SQL 是否存在"
            "导致取不到数据的错误，并给出最小改动的修正建议。\n"
            "判定原则：\n"
            "1. 只有能明确指出具体问题点时才能判定 SQL 有错；禁止把"
            "『可能需要核实品类』『可能存在遗漏』『值可能有空格/别名』这类猜测当作结论；\n"
            "2. 若下方提供了表结构，其中已给出的表名、字段名和取值（如品类、状态、"
            "区域的取值范围）就是判断依据。若过滤值落在给出的取值范围内，不得以"
            "『实际存储可能不同』为由判定为错误，也不得建议用模糊匹配或 IN 去兜底；\n"
            "3. 结果为空不等于 SQL 有错。若表、字段、过滤条件和时间范围都与用户问题一致"
            "且均有依据，应直接说明『该口径下确实没有数据』，判定 sql_error=false，"
            "此时 suggestion 写『无需修改 SQL』，可附一句仅用于验证的建议并注明"
            "『仅用于验证，不作为最终 SQL』；"
            "严禁为了让结果非空而建议放宽、删除或模糊化本来正确的过滤条件；\n"
            "4. 只针对导致取不到数据的问题给建议，不得顺带新增用户问题未要求的过滤条件"
            "（如擅自增加订单状态限制）或改变统计口径；\n"
            "5. 只给能直接套用的最小改动（指明改哪个条件、改成什么），"
            "不要输出 SELECT DISTINCT 之类逐步探查的步骤清单；\n"
            "6. 建议中引用的表名、字段名必须来自下方表结构，不得臆造；"
            "若未提供表结构，不得臆造表名或字段名。\n"
            "重点核对（结合当前日期）：\n"
            "1. 过滤值是否落在字段的取值范围内，是否存在大小写/中英文不一致；\n"
            "2. 时间范围是否与用户问题一致：按『当前日期』推算近N个月/最近N天/上月/"
            "今年以来等相对时间，明确指出正确的起止日期；\n"
            "3. 是否叠加了用户问题未要求的过滤条件；\n"
            "4. 是否选错了表或字段（如用订单创建时间代替支付时间）；\n"
            "5. 多表关联字段是否正确。\n"
            "输出要求：sql_error 为布尔值，只有能明确指出导致取不到数据的具体问题点时才为 "
            "true，否则为 false；judgment 不超过100字，suggestion 不超过200字且必须是"
            "可执行的修改点。\n"
            '只输出严格 JSON：{"sql_error": true 或 false, '
            '"judgment": "对错误的判断；无错误时直接说明属于该口径无数据", '
            '"suggestion": "最小改动的修正建议（中文）；无需修改时写『无需修改 SQL』"}'
        )
        human_parts = [
            f"当前日期：{datetime.now().strftime('%Y-%m-%d')}",
            f"用户问题：{user_question}",
            f"生成的 SQL：\n{sql}",
            f"失败原因：{error_desc}",
        ]
        if schema_context:
            human_parts.append(f"表结构/关联关系/取值参考：\n{schema_context}")
        human = "\n\n".join(human_parts)
        result = await self._call_llm_and_parse_json(sys_prompt, human)
        if result is None:
            # 模型不可用/输出不可解析：按"有错"保守处理，保留重试
            return True, error_desc
        judgment = str(result.get("judgment") or "").strip()
        suggestion = str(result.get("suggestion") or "").strip()
        if not judgment and not suggestion:
            return True, error_desc
        # 键缺失或为 null 时按"有错"保守处理，保留重试，避免误判为正确
        if not self._to_bool(result.get("sql_error"), True):
            # SQL 正确、该口径无数据：优先用 judgment 作为前端展示的无数据原因，
            # judgment 缺失时给一句明确的默认说明，避免把"无需修改 SQL"当原因展示。
            return False, judgment or "该口径下没有查询到数据，SQL 与问题口径一致。"
        parts = []
        if judgment:
            parts.append(f"校验失败原因：{judgment}")
        if suggestion:
            parts.append(f"改造建议：{suggestion}")
        return True, "；".join(parts)
