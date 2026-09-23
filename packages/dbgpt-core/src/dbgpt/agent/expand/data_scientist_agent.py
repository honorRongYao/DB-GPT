"""Data Scientist Agent."""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dbgpt.core import (
    ModelMessage,
    ModelMessageRoleType,
    ModelRequest,
)

from ..core.action.base import ActionOutput
from ..core.agent import Agent, AgentMessage
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
    "只使用结果中出现的字段，不得自造维度；必须区分三个行数："
    "总行数、展示行数（下发给用户看到的行数）、结论依据行数（你实际据以分析的行数），"
    "不得把展示行数或预览行数当作完整结果（见输入中的【结果样本范围】）。\n"
    "按以下四段输出，段间空一行，不要使用 Markdown 语法（不要 **、##、表格），"
    "涨跌用 ↑ ↓ 标注：\n"
    "📌 结论\n一句话直接回答用户问题，带最关键的数字，不超过60字。\n"
    "🔍 关键发现\n用 1. 2. 3. 编号，按重要性排序只保留2-4条，"
    "每条为「现象+数字+对比+业务含义」。\n"
    "💡 建议行动\n用 1. 2. 编号，只给2-3条可直接执行的动作。\n"
    "⚠️ 数据说明\n一到两句，只写口径、局限或置信度，无歧义时可省略；"
    "但三个行数不一致时不得省略，必须用一句白话写清这三层范围"
    "（如「共查出 5000 行，页面只展示前 200 行，本结论基于前 50 行生成」）。\n"
    "不评价 SQL，不描述生成过程。只输出严格 JSON，且只能有 thought 一个键"
    "（禁止 output、reasoning、answer 等其他键）："
    "{\"thought\": \"四段结论\"}，换行用 \\n 转义。"
)

# 结论生成所用的 skill（改文件即可调整结论质量，无需改代码）
_SUMMARY_SKILL_RELATIVE_PATH = os.path.join("user", "result-summary", "SKILL.md")

# 校验失败兜底时的用户提示文案：区分"结果不合规"与"SQL 执行失败"两种情况，
# 避免业务把"SQL 跑挂了"误读成"结果仅供参考"。
_WARN_UNVERIFIED = "以下结果未经校验通过，可能不满足问题要求，仅供参考。"
_WARN_SQL_FAILED = "本次 SQL 执行失败，没有取到数据，请调整问题或稍后重试。"

# ReAct 校验失败的分类：只用于诊断打点（grep "ReAct retry stats" 看汇总），
# 用来判断"10 轮仍答不对"到底卡在哪一类，而不是靠猜。
_FAIL_SQL_EXEC = "sql_exec_failed"  # SQL 执行报错
_FAIL_EMPTY = "empty_result"  # 执行成功但 0 行，且诊断判定 SQL 写错
_FAIL_SEMANTIC = "semantic_rejected"  # 结果非空但语义校验不通过
_FAIL_FORMAT = "bad_format"  # 回复不是合法 JSON / 缺 sql 等格式问题
_FAIL_EXCEPTION = "check_exception"  # 校验过程抛异常

# 兜底提示里回显"未通过原因"的长度上限，避免把审核的长篇说明整段塞给用户。
_WARN_REASON_MAX_LEN = 200


@dataclass
class _RetryRecord:
    """一次校验失败的记录：诊断统计用，也用于兜底时挑"最像答案"的一轮。"""

    round_no: int
    category: str
    sql: Optional[str]
    reason: str
    is_exe_success: bool
    row_count: int
    action_out: Optional[ActionOutput]
    action_reply_obj: Optional[dict]


# SQL 执行失败后统一做一次诊断：先裁掉异常串里的噪声，再让大模型判断"为什么报错"，
# 把诊断结论（而非原始异常）作为重试反馈回喂给 Agent。
_SQL_ECHO_RE = re.compile(r"\[SQL:.*?(?=\n\(Background on this error|\Z)", re.DOTALL)
_SQL_PARAMS_RE = re.compile(
    r"\[parameters:.*?(?=\n\(Background on this error|\Z)", re.DOTALL
)
_SQL_DOC_RE = re.compile(r"\(Background on this error at:.*?\)", re.DOTALL)

_SQL_ERROR_ANALYZE_PROMPT = (
    "你是一个严谨的数据库专家。数据分析 Agent 生成的一条 SQL 执行失败了，"
    "请判断它为什么报错，并给出最小改动的修正建议。\n"
    "要求：\n"
    "1. 必须以给出的报错信息为依据，明确指出具体原因：哪个对象、哪一步、"
    "违反了哪条规则。不要泛泛而谈，也不要把所有可能的猜测都列一遍；\n"
    "2. 报错信息里已经写明的事实直接采信（例如报错说某表在某库不存在，"
    "就认定该对象不可用），不要反向怀疑报错本身；\n"
    "3. 严禁编造表名、字段名、库名。若报错是表/字段不存在或无权访问，"
    "必须如实说明该对象不可用，不得把它换成另一张表名，也不得臆造新表；\n"
    "4. suggestion 只给能直接套用的最小改动（指明改哪个位置、改成什么）；"
    "现有信息不足以给出可靠改法时，写『需重新核对可用表与字段后重写 SQL』；\n"
    "5. cause 不超过 120 字，suggestion 不超过 200 字。\n"
    '只输出严格 JSON：{"cause": "报错的根本原因", '
    '"suggestion": "最小改动的修正建议（中文）"}'
)


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

    max_retry_count: int = 10
    language: str = "zh"

    def __init__(self, **kwargs):
        """Create a new DataScientistAgent instance."""
        super().__init__(**kwargs)
        self._init_actions([ChartAction])
        # 当前重试轮次（0 基），由 _load_thinking_messages 每轮刷新
        self._current_retry: int = 0
        # 本轮提问里每次校验失败的记录，供诊断打点与兜底挑候选
        self._retry_records: List[_RetryRecord] = []
        # 上游语义层经 received_message.context["resource_prompt"] 注入的参考信息
        # （允许表 + 完整表结构 + 召回来源 SQL + 用户问题），用于覆盖数据源自带的表结构
        self._resource_prompt_override: Optional[str] = None

    async def load_resource(self, question: str, is_retry_chat: bool = False):
        """优先使用上游语义层注入的参考信息，没有再回落到数据源自带的表结构。

        Agent 绑定的 DBResource 只会返回全库原始 DDL（可能过期、且与本问题无关），
        语义层注入的才是精选表结构 + 业务规则，两者同时出现会互相打架。
        """
        if self._resource_prompt_override:
            return self._resource_prompt_override, None
        return await super().load_resource(question, is_retry_chat)

    async def read_memories(self, question: str) -> str:
        """Do not load long-term memories for SQL generation."""
        return ""

    async def _load_thinking_messages(
        self, *args, **kwargs
    ) -> Tuple[List[AgentMessage], Optional[Any]]:
        """记录当前重试轮次（0 基），供校验失败时判断是否已是最后一轮。

        失败记录只能在这里、且只在第 0 轮清空：_init_reply_message 在每次重试时
        都会被调用（base_agent.generate_reply 的 current_retry_counter>0 分支），
        若把清空放在那里，每轮都会把前面的记录丢掉，汇总统计永远只有 1 条、
        distinct_sql 恒为 1，兜底也只能挑到最后一轮。
        """
        self._current_retry = kwargs.get("current_retry_counter") or 0
        if self._current_retry == 0:
            # 新一轮提问（该 Agent 实例会被复用），清掉上一轮的失败记录
            self._retry_records = []
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
        warn: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """最后一轮仍未通过校验时的兜底收口。

        不再返回失败（否则循环结束、前端空白），而是保留已有结果数据、
        把简短提示写进结论，按"校验通过"返回，供前端展示"结果 + 免责说明"。

        warn: 面向用户的提示文案；不传时用通用的"未经校验"说明。
        区分文案是为了让业务能分清"SQL 执行失败"与"结果不合规"两种情况。

        结果数据不再默认取最后一轮：最后一轮往往已被前面几轮（可能是错的）失败
        反馈带偏，这里回退到"最像答案"的一轮——优先执行成功且行数非空，其次执行
        成功，都没有才用最后一轮。用户提示里同时标明取自第几轮、以及未通过的原因。
        """
        best = self._select_best_candidate()
        if best is not None and best.action_out is not None and action_out is not None:
            # 用候选轮的数据覆盖本轮 action_out 的内容，而不是换引用：调用方
            # （message.action_report）持有的是本轮这个对象，换引用前端拿不到替换结果。
            sql = best.sql or sql
            action_reply_obj = best.action_reply_obj
            reason = best.reason
            action_out.is_exe_success = best.action_out.is_exe_success
            # 这条候选是执行成功的那条，不适用"SQL 执行失败"文案
            warn = None
            round_no = best.round_no
        else:
            round_no = self._current_retry + 1
        detail = self._shorten_reason(reason)
        warn = (
            f"{warn or _WARN_UNVERIFIED}（取自第 {round_no} 轮尝试"
            f"{'，未通过原因：' + detail if detail else ''}）"
        )
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
        logger.info(
            f"Last round not pass, finalize with round {round_no} result, "
            f"reason: {reason}"
        )
        return True, None

    async def _check_fail(
        self,
        reason: str,
        sql: Optional[str] = None,
        action_out: Optional[ActionOutput] = None,
        action_reply_obj: Optional[dict] = None,
        warn: Optional[str] = None,
        category: str = _FAIL_FORMAT,
    ) -> Tuple[bool, Optional[str]]:
        """校验失败的统一出口：非最后一轮返回失败触发重试，最后一轮兜底收口。

        category 只服务诊断打点：把每次失败归类记下来，最后一轮输出汇总，用来判断
        "10 轮仍答不对"主要卡在 SQL 报错、结果为空还是语义校验，而不是靠猜。
        """
        self._record_retry_failure(
            category,
            reason,
            sql=sql,
            action_out=action_out,
            action_reply_obj=action_reply_obj,
        )
        if self._is_last_round():
            self._log_retry_stats()
            return await self._finalize_unverified(
                reason,
                sql=sql,
                action_out=action_out,
                action_reply_obj=action_reply_obj,
                warn=warn,
            )
        # 返回的重试反馈会在下一轮拼成 [2] HUMAN 的内容。base_agent 已用
        # "【校验失败原因与改造建议】" 作标题，所以原因放在最前、SQL 补在后面，
        # 否则标题下方先出现 SQL 段，读起来像个空标题。
        # 补 SQL 的原因：重试时模型只有系统提示词、原始问题和这段反馈，拿不到自己
        # 上一轮写过的 SQL（read_memories 返回空串，历史消息也不会重建），不给 SQL
        # 就无从"修正"，只能每轮从零重写。
        if sql:
            return (
                False,
                f"{reason}\n\n【上一轮提交的 SQL（已执行，未通过校验）】\n{sql}",
            )
        return False, reason

    def _record_retry_failure(
        self,
        category: str,
        reason: Optional[str],
        sql: Optional[str] = None,
        action_out: Optional[ActionOutput] = None,
        action_reply_obj: Optional[dict] = None,
    ) -> None:
        """记下一次校验失败，并打一行分轮日志（含已出现的不重复 SQL 条数）。"""
        row_count = 0
        if isinstance(action_reply_obj, dict):
            _, values = self._extract_result_rows(action_reply_obj)
            row_count = len(values)
        record = _RetryRecord(
            round_no=self._current_retry + 1,
            category=category,
            sql=sql,
            reason=reason or "",
            is_exe_success=bool(action_out is not None and action_out.is_exe_success),
            row_count=row_count,
            action_out=action_out,
            action_reply_obj=action_reply_obj,
        )
        self._retry_records.append(record)
        logger.info(
            f"ReAct round {record.round_no} failed: category={category}, "
            f"exe_success={record.is_exe_success}, rows={row_count}, "
            f"distinct_sql={self._distinct_sql_count()}"
        )

    def _log_retry_stats(self) -> None:
        """汇总本次 ReAct 的失败构成，定位"10 轮仍答不对"的根因。"""
        counters: Dict[str, int] = {}
        for record in self._retry_records:
            counters[record.category] = counters.get(record.category, 0) + 1
        best = self._select_best_candidate()
        logger.info(
            "ReAct retry stats: rounds=%s, %s, distinct_sql=%s, best_candidate=%s",
            self._current_retry + 1,
            ", ".join(f"{key}={value}" for key, value in sorted(counters.items())),
            self._distinct_sql_count(),
            f"round{best.round_no}(rows={best.row_count})" if best else "none",
        )

    def _select_best_candidate(self) -> Optional[_RetryRecord]:
        """挑一条最像正确答案的失败记录：执行成功且行数非空优先，同档取最晚一轮。"""
        with_data = [
            record
            for record in self._retry_records
            if record.is_exe_success and record.row_count > 0
        ]
        if with_data:
            return with_data[-1]
        executed = [record for record in self._retry_records if record.is_exe_success]
        return executed[-1] if executed else None

    def _distinct_sql_count(self) -> int:
        """已尝试过的不同 SQL 条数：数值接近轮数说明模型没有重复自己。"""
        return len({record.sql for record in self._retry_records if record.sql})

    @staticmethod
    def _shorten_reason(reason: Optional[str]) -> str:
        """把"未通过原因"压成一行并限长，供用户提示使用。"""
        return " ".join((reason or "").split())[:_WARN_REASON_MAX_LEN]

    def _init_reply_message(
        self,
        received_message: AgentMessage,
        rely_messages: Optional[List[AgentMessage]] = None,
    ) -> AgentMessage:
        # 上游语义层把「参考信息」正文（允许表 + 表结构 + 召回 SQL + 用户问题）放在
        # 消息 context 里，message.content 只留用户问题。这里取出来：
        # - 它同时作为系统提示词的 {resource_prompt}（见 load_resource 覆盖）；
        # - 也用来拼 _current_question，因为这段文本带着"用户问题:"标记，
        #   _split_question_and_schema 才能拆出表结构，供 correctness_check
        #   等 LLM 校验使用；若只用 content（纯问题）则会丢掉表结构上下文。
        received_context = received_message.get_dict_context()
        override = received_context.get("resource_prompt")
        self._resource_prompt_override = override if isinstance(override, str) else None
        self._current_question = (
            self._resource_prompt_override or received_message.content or ""
        )
        reply_message = super()._init_reply_message(received_message, rely_messages)
        # 提示词里的 {dialect} 取这里。不能直接用连接层的 dialect：Doris、StarRocks
        # 这类库走 MySQL 协议，引擎方言名只会是 "mysql"，但它们的语法约束与 MySQL
        # 并不等同（例如 Doris 的 WHERE 中不允许出现聚合函数），提示词里写 "mysql"
        # 会诱导模型按 MySQL 习惯生成 SQL，在真实库上报错。所以优先报资源上真实的
        # 库类型 db_type，取不到时才退回 dialect。
        try:
            prompt_dialect = self.database.db_type or self.database.dialect
        except ValueError:
            prompt_dialect = self.database.dialect
        reply_message.context = {
            "display_type": self.actions[0].render_prompt(),
            "dialect": prompt_dialect,
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

    @staticmethod
    def _extract_result_rows(
        action_reply_obj: dict,
    ) -> Tuple[List[str], List[Any]]:
        """从本轮结果里取出 (列名, 行数据)，用于校验与结论生成。

        数据来源是 ChartAction 执行 SQL 后回填的 data/count（由真实查询结果生成，
        不是 LLM 输出），因此这里直接复用，不必再把同一条 SQL 执行一次。
        执行成功但没有 data 键即"查询成功但 0 行"，返回空列表。
        """
        rows = action_reply_obj.get("data")
        if not isinstance(rows, list) or not rows:
            return [], []
        first = rows[0]
        if isinstance(first, dict):
            columns = list(first.keys())
            values = [
                [row.get(col) for col in columns]
                for row in rows
                if isinstance(row, dict)
            ]
            return columns, values
        return [], [list(row) for row in rows]

    async def verify(
        self,
        message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[str]]:
        """校验本轮回复，SQL 执行失败时先让大模型诊断再重试。

        基类 verify 在 action 执行失败时会直接返回 action_output.content（原始英文
        异常串）并短路，永远不会走到 correctness_check。这里前置拦截并替换掉那句
        转发，用诊断结论当重试反馈——原始异常串含 SQL 回显与文档链接，直接回喂会
        干扰模型。

        不分简单/复杂，统一都让大模型判断"为什么报错"。ActionOutput.content 只有
        报错、不含 SQL，SQL 从 LLM 原始回复里取。
        """
        action_out = message.action_report
        review_approved = not message.review_info or message.review_info.approve
        if action_out is not None and not action_out.is_exe_success and review_approved:
            question = getattr(self, "_current_question", "") or ""
            error_desc = action_out.content or ""
            sql = self._extract_sql_from_reply(message.content)
            if not question:
                logger.warning(
                    "SQL failed but current question is empty, skip analysis"
                )
                feedback = f"Please check your answer, {error_desc}."
            else:
                feedback = await self._analyze_sql_error(
                    question,
                    sql,
                    error_desc,
                )
                logger.info(f"SQL error analyzed, retry feedback: {feedback}")
            return await self._check_fail(
                feedback,
                sql=sql,
                action_out=action_out,
                warn=_WARN_SQL_FAILED,
                category=_FAIL_SQL_EXEC,
            )
        return await super().verify(message, sender, reviewer, **kwargs)

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
            # 执行失败的情况已在 verify 里前置拦截并做过诊断，走不到这里；
            # 保底仍按失败返回，避免被直接调用时把报错串当成结果解析。
            return False, action_out.content or ""
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
                category=_FAIL_FORMAT,
            )
        if not isinstance(action_reply_obj, dict):
            return await self._check_fail(
                "Please check your answer, the output content is not a valid JSON "
                "object, please regenerate a reply strictly in the required format.",
                action_out=action_out,
                category=_FAIL_FORMAT,
            )
        sql = action_reply_obj.get("sql", None)
        if not sql:
            return await self._check_fail(
                "Please check your answer, the sql information that needs to be "
                "generated is not found.",
                action_out=action_out,
                action_reply_obj=action_reply_obj,
                category=_FAIL_FORMAT,
            )
        try:
            if not action_out.resource_value:
                return await self._check_fail(
                    "Please check your answer, the data resource information is not "
                    "found.",
                    sql=sql,
                    action_out=action_out,
                    action_reply_obj=action_reply_obj,
                    category=_FAIL_FORMAT,
                )

            # 直接复用 ChartAction 已执行的真实结果（data 由 ChartAction 用查询结果
            # 回填，不是 LLM 生成），避免把同一条 SQL 再执行一次；
            # 执行成功但没有 data 键，即"查询成功但 0 行"。
            columns, values = self._extract_result_rows(action_reply_obj)
            if not values:
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
                        category=_FAIL_EMPTY,
                    )
                return await self._check_fail(
                    error_desc,
                    sql=sql,
                    action_out=action_out,
                    action_reply_obj=action_reply_obj,
                    category=_FAIL_EMPTY,
                )
            else:
                logger.info(
                    f"reply check success! There are {len(values)} rows of data"
                )
                # 增强自检：执行结果非空之外，再用 LLM 校验结果是否完整满足用户问题要求
                question = getattr(self, "_current_question", "") or ""
                if question:
                    check_ok, check_reason = await self._llm_result_check(
                        question,
                        sql,
                        columns,
                        values,
                        total_rows=action_reply_obj.get("count"),
                    )
                    if not check_ok:
                        return await self._check_fail(
                            check_reason,
                            sql=sql,
                            action_out=action_out,
                            action_reply_obj=action_reply_obj,
                            category=_FAIL_SEMANTIC,
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
            question = getattr(self, "_current_question", "") or ""
            if question:
                # 与执行失败路径一致：统一先诊断，用结论替代原始异常串回喂
                feedback = await self._analyze_sql_error(question, sql, str(e))
            else:
                feedback = (
                    "SQL execution error, please re-read the historical information "
                    f"to fix this SQL. The error message is as follows:{str(e)}"
                )
            return await self._check_fail(
                feedback,
                sql=sql,
                action_out=action_out,
                action_reply_obj=action_reply_obj,
                category=_FAIL_EXCEPTION,
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
        self,
        question: str,
        sql: str,
        columns: List[str],
        values: List[Any],
        total_rows: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """用 LLM 校验 SQL 与执行结果是否完整满足用户问题的要求。

        在"执行结果非空"的基础上，进一步检查结果"对不对"：
        例如要求 TOP3 但 SQL 没有取前3、枚举类过滤值该用 LIKE 却用了 = 等。
        判定从严把握"错才判错"：只有能明确指出具体问题点、并给出可执行改法时才判
        不通过；否则一律判通过。LLM 不可用或校验结果解析失败时按"通过"处理，
        不阻塞主流程。时间口径只依据表结构取值参考与执行结果判断，不注入当前日期，
        避免模型用日期推算月份导致误判。

        total_rows: 真实总行数（结果可能被截断展示，values 只是其中一部分），
        缺省时按 len(values) 处理，避免把"展示行数"误报成"结果只有这么多行"。
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
            "『需进一步确认』这类没有明确问题点的理由，一律视为通过；\n"
            "5. 执行结果非空即证明所查询的对象（含时间区间、过滤值）在库中确实有数据，"
            "不得据此反向声称某月『数据尚未更新/尚未产生』；要否定某个过滤值，"
            "必须给出该值在表结构取值参考中不存在、或结果为空的具体依据；\n"
            "6. 下方表结构里『使用 xxx 生成SQL需要参考下面规范』的规范是唯一业务口径。"
            "规范已明文规定的，逐条对照规范判定；规范未规定的事项，"
            "例如趋势用首末月对比还是环比/回归、缺失月份如何填充、"
            "『最高/前N』按哪个时间窗口排名，一律判通过，"
            "不得以『更严谨/更精确/更全面/更符合分析习惯』为由否定一种合理实现。\n"
            "重点检查（仅限可明确判定的硬性要求）：\n"
            "1. 问题要求 TOP3/前N/排名时，SQL 是否真正用窗口函数或 LIMIT 取了前N，"
            "若没有取前N则明确指出并给出改法；\n"
            "2. 时间口径只能依据可验证的数据事实判断：① 表结构/取值参考中列出的时间字段"
            "真实取值（如 dt 的枚举值）；② SQL 执行结果里实际出现的时间值。"
            "禁止用当前日期或自行假设的数据更新规则推算『最新月份应该是哪个月』；"
            "SQL 过滤的月份在上述任一事实中出现过、或执行结果非空（说明该月确实有数据），"
            "一律视为有效，不得判错；只有当 SQL 的时间范围与上述事实明确矛盾、"
            "或结果为空时，才能判时间问题；\n"
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
        # 结果可能超过展示上限被截断：总行数用真实值，另标出实际用于校验的行数
        real_total = total_rows if total_rows and total_rows > 0 else len(values)
        shown_scope = f"共 {real_total} 行"
        if real_total > len(values):
            shown_scope += f"，本次仅取前 {len(values)} 行用于校验"
        # 不注入当前日期：时间口径一律以表结构取值参考和实际执行结果为准。
        # 注入当天日期会诱导模型用日期推算"最新月份应该是哪个月"，把正确月份判成错误。
        human_parts = [
            f"用户问题：{user_question}",
            f"生成的 SQL：\n{sql}",
            f"执行结果（{shown_scope}，其中预览前 {len(preview_values)} 行）：\n"
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

    @staticmethod
    def _ensure_sample_scope_note(
        thought: str, row_count: int, shown_rows: int, preview_rows: int
    ) -> str:
        """结论没说明样本范围时，按代码事实补一句，避免展示文本误导业务。

        ⚠️ 数据说明 段允许省略，模型偶尔会整段省掉；此时业务会默认"表里展示的
        行就是分析过的全部行"。展示行数小于真实总行数、或结论依据行数小于展示
        行数时需要点明三者关系；模型已经写了总行数就不再补。
        """
        if row_count <= preview_rows or str(row_count) in thought:
            return thought
        if row_count > shown_rows:
            note = (
                f"本结论基于前 {preview_rows} 行生成；"
                f"本次结果共 {row_count} 行，页面仅展示前 {shown_rows} 行。"
            )
        else:
            note = (
                f"本结论基于前 {preview_rows} 行生成；"
                f"本次结果共 {row_count} 行，已全部展示。"
            )
        if "⚠️" in thought:
            return f"{thought}{note}"
        return f"{thought}\n\n⚠️ 数据说明\n{note}"

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
        # 样本范围是结论可靠性的前提：三个行数都是代码事实——真实总行数、
        # 随结果下发给用户的行数（展示有上限）、实际喂给模型生成结论的行数。
        # 三者不一致时必须让模型知道，否则它会把展示行数当成完整结果。
        shown_rows = len(displayed_rows)
        preview_rows = len(rows_preview)
        if row_count > shown_rows:
            sample_scope = (
                f"共 {row_count} 行；其中前 {shown_rows} 行随结果展示给用户"
                f"（展示有上限，真实结果不止这些）；本结论仅基于前 {preview_rows} 行生成。"
            )
        elif row_count > preview_rows:
            sample_scope = (
                f"共 {row_count} 行，已全部展示给用户；"
                f"本结论仅基于其中前 {preview_rows} 行生成。"
            )
        else:
            sample_scope = f"共 {row_count} 行，已全部展示，并全部作为本结论依据。"
        # 与校验环节共用同一套拆分口径：标记写法若对不上就不会拆分，
        # 会把整段表结构当成"用户问题"传给结论模型，导致结论跑偏且浪费 token
        user_question, _ = self._split_question_and_schema(question)
        sys_prompt = self._load_summary_skill_prompt() or _DEFAULT_SUMMARY_PROMPT
        human = (
            f"用户问题：{user_question}\n\n"
            f"已执行 SQL：\n{sql}\n\n"
            f"结果总行数：{row_count}\n"
            f"结果样本范围：{sample_scope}\n"
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
            thought = self._ensure_sample_scope_note(
                thought, row_count, shown_rows, preview_rows
            )

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

    @staticmethod
    def _trim_sql_error(text: str) -> str:
        """裁剪 SQLAlchemy 异常串，只保留数据库返回的核心报错。

        原始串里 [SQL: ...] 会回显整条 SQL、[parameters: ...] 回显参数、末尾还有
        sqlalche.me 文档链接，这些都是噪声；去掉它们并压缩空白，避免重试时把大段
        无用内容塞进上下文干扰模型。
        """
        raw = (text or "").strip()
        if not raw:
            return ""
        for pattern in (_SQL_ECHO_RE, _SQL_PARAMS_RE, _SQL_DOC_RE):
            raw = pattern.sub("", raw)
        return re.sub(r"\s+", " ", raw).strip()

    @staticmethod
    def _extract_sql_from_reply(text: str) -> str:
        """从 LLM 原始回复里取 sql 字段。

        SQL 执行失败时 ActionOutput.content 只有报错、不携带 SQL，只能从 LLM 回复
        里取；取不到就返回空串，由诊断提示词仅依据报错信息判断。
        """
        raw = text or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return ""
        try:
            obj = json.loads(raw[start : end + 1])
        except Exception:
            return ""
        if not isinstance(obj, dict):
            return ""
        return str(obj.get("sql") or "")

    async def _analyze_sql_error(
        self, question: str, sql: str, error_desc: str
    ) -> str:
        """让大模型判断 SQL 为什么执行失败，返回给 Agent 的重试反馈。

        不分简单/复杂，执行失败统一先做一次诊断：反馈用"报错原因 + 修正建议"，
        而不是原始异常串。大模型不可用或输出解析失败时，退回裁剪后的原始报错，
        保留原有重试行为。
        """
        trimmed = self._trim_sql_error(error_desc)
        user_question, schema_context = self._split_question_and_schema(question)
        human_parts = [
            f"用户问题：{user_question}",
            f"生成的 SQL：\n{sql or '(本轮未取到 SQL 文本，请仅依据报错信息判断)'}",
            f"数据库报错：\n{trimmed or error_desc}",
        ]
        if schema_context:
            human_parts.append(f"表结构/关联关系/取值参考：\n{schema_context}")
        result = await self._call_llm_and_parse_json(
            _SQL_ERROR_ANALYZE_PROMPT, "\n\n".join(human_parts)
        )
        if not result:
            return trimmed or error_desc
        cause = str(result.get("cause") or "").strip()
        suggestion = str(result.get("suggestion") or "").strip()
        if not cause and not suggestion:
            return trimmed or error_desc
        parts = []
        if cause:
            parts.append(f"报错原因：{cause}")
        if suggestion:
            parts.append(f"修正建议：{suggestion}")
        return "本次 SQL 执行失败。" + "；".join(parts)
