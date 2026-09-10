"""GPTs memory."""

import asyncio
import json
import logging
from asyncio import Queue
from collections import defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from dbgpt.util.executor_utils import blocking_func_to_async
from dbgpt.vis.client import VisAgentMessages, VisAgentPlans, VisAppLink, vis_client

from ...action.base import ActionOutput
from ...schema import Status
from .base import GptsMessage, GptsMessageMemory, GptsPlansMemory
from .default_gpts_memory import DefaultGptsMessageMemory, DefaultGptsPlansMemory

NONE_GOAL_PREFIX: str = "none_goal_count_"
# 语义层折叠行的目标标题：语义层各阶段不再各占一行计划，而是合并成一条
# “正在进行检索/选表”的整行；行内不放气泡，各阶段文字分行放在该行 markdown 中。
SEMANTIC_LAYER_GOAL: str = "[语义层]:正在查询语义层资料完成检索"

logger = logging.getLogger(__name__)


class GptsMemory:
    """GPTs memory."""

    def __init__(
        self,
        plans_memory: Optional[GptsPlansMemory] = None,
        message_memory: Optional[GptsMessageMemory] = None,
        executor: Optional[Executor] = None,
    ):
        """Create a memory to store plans and messages."""
        self._plans_memory: GptsPlansMemory = (
            plans_memory if plans_memory is not None else DefaultGptsPlansMemory()
        )
        self._message_memory: GptsMessageMemory = (
            message_memory if message_memory is not None else DefaultGptsMessageMemory()
        )
        self._executor = executor or ThreadPoolExecutor(max_workers=2)
        self.messages_cache: defaultdict = defaultdict(list)
        self.channels: defaultdict = defaultdict(Queue)
        self.enable_vis_map: defaultdict = defaultdict(bool)
        self.start_round_map: defaultdict = defaultdict(int)

    @property
    def plans_memory(self) -> GptsPlansMemory:
        """Return the plans memory."""
        return self._plans_memory

    @property
    def message_memory(self) -> GptsMessageMemory:
        """Return the message memory."""
        return self._message_memory

    def init(
        self,
        conv_id: str,
        enable_vis_message: bool = True,
        history_messages: Optional[List[GptsMessage]] = None,
        start_round: int = 0,
    ):
        """Gpt memory init."""
        self.channels[conv_id] = asyncio.Queue()
        self.enable_vis_map[conv_id] = enable_vis_message
        self.messages_cache[conv_id] = history_messages if history_messages else []
        self.start_round_map[conv_id] = start_round

    def enable_vis_message(self, conv_id):
        """Enable conversation message vis tag."""
        return self.enable_vis_map[conv_id] if conv_id in self.enable_vis_map else True

    def queue(self, conv_id: str):
        """Get conversation message queue."""
        return self.channels[conv_id] if conv_id in self.channels else None

    def clear(self, conv_id: str):
        """Clear gpt memory."""
        # clear last message queue
        queue = self.channels.pop(conv_id)  # noqa
        del queue
        # clear messages cache'
        if self.messages_cache.get(conv_id):
            cache = self.messages_cache.pop(conv_id)  # noqa
            del cache

        # clear vis_enable_tag
        vis_enable_tag = self.enable_vis_map.pop(conv_id)  # noqa
        del vis_enable_tag

        # clear start_roun
        start_round = self.start_round_map.pop(conv_id)  # noqa
        del start_round

    async def push_message(self, conv_id: str, temp_msg: Optional[str] = None):
        """Push conversation message."""
        queue = self.queue(conv_id)
        if not queue:
            # 客户端已断连时，memory.clear() 已删除该 conv 的队列，跳过推送即可
            logger.warning(f"push_message skip, queue not found for conv_id: {conv_id}")
            return
        enable_vis_tag = self.enable_vis_message(conv_id=conv_id)
        if enable_vis_tag:
            # VIS 协议下，逐 token 的"临时流文本帧"不再推送：
            # - 这类帧会携带正在生成的原始文本（常含中间 SQL），不是过程"问题与解决"，
            #   也不属于最终结果，与展示口径冲突；
            # - 每个 token 都要重建并推送一次整帧，是 SSE 膨胀的主因之一。
            # 阶段进度、消息落库（append_message）时仍会推送完整帧，交互不受影响。
            if temp_msg:
                return
            # 直接从短期记忆中发布最后消息
            message_view = await self.app_link_chat_message(conv_id)
            await queue.put(message_view)

        else:
            # 非VIS消息模式，直接推送简单消息列表即可，不做任何处理
            message_views = await self.simple_message(conv_id)
            if temp_msg:
                temp_view = await self.agent_stream_message(temp_msg, False)
                if temp_view and len(temp_view) > 0:
                    message_views.extend(temp_view)
            await queue.put(message_views)

    async def complete(self, conv_id: str):
        """Complete conversation message."""
        queue = self.queue(conv_id)
        if not queue:
            # 客户端已断连时，memory.clear() 已删除该 conv 的队列，无需通知前端
            logger.warning(f"complete skip, queue not found for conv_id: {conv_id}")
            return

        await queue.put("[DONE]")

    async def append_message(self, conv_id: str, message: GptsMessage):
        """Append message."""
        self.messages_cache[conv_id].append(message)
        await blocking_func_to_async(
            self._executor, self.message_memory.append, message
        )

        # 消息记忆后发布消息
        await self.push_message(conv_id)

    async def get_messages(self, conv_id: str) -> List[GptsMessage]:
        """Get message by conv_id."""
        messages = self.messages_cache[conv_id]
        if not messages:
            messages = await blocking_func_to_async(
                self._executor, self.message_memory.get_by_conv_id, conv_id
            )
        return messages

    async def get_agent_messages(
        self, conv_id: str, agent_role: str
    ) -> List[GptsMessage]:
        """Get agent messages."""
        gpt_messages = self.messages_cache[conv_id]
        result = []
        for gpt_message in gpt_messages:
            if gpt_message.sender == agent_role or gpt_messages.receiver == agent_role:
                result.append(gpt_message)
        return result

    async def get_agent_history_memory(
        self, conv_id: str, agent_role: str
    ) -> List[ActionOutput]:
        """Get agent history memory."""

        agent_messages = await blocking_func_to_async(
            self._executor, self.message_memory.get_by_agent, conv_id, agent_role
        )
        new_list = []
        for i in range(0, len(agent_messages), 2):
            if i + 1 >= len(agent_messages):
                break
            action_report = None
            if agent_messages[i + 1].action_report:
                action_report = ActionOutput.from_dict(
                    json.loads(agent_messages[i + 1].action_report)
                )
            new_list.append(
                {
                    "question": agent_messages[i].content,
                    "ai_message": agent_messages[i + 1].content,
                    "action_output": action_report,
                    "check_pass": agent_messages[i + 1].is_success,
                }
            )

        # Just use the action_output now
        return [m["action_output"] for m in new_list if m["action_output"]]

    async def _message_group_vis_build(self, message_group, vis_items: list):
        num: int = 0
        terminate_content = None  # Store terminate output for pure text display

        if message_group:
            last_goal = next(reversed(message_group))
            last_goal_message = None
            if not last_goal.startswith(NONE_GOAL_PREFIX):
                last_goal_messages = message_group[last_goal]
                last_goal_message = last_goal_messages[-1]

                # Check if the last message is a terminate action
                if last_goal_message and last_goal_message.action_report:
                    try:
                        action_out = ActionOutput.from_dict(
                            json.loads(last_goal_message.action_report)
                        )
                        if action_out and action_out.terminate:
                            # Extract terminate content for pure text display
                            terminate_content = (
                                action_out.content
                                or action_out.view
                                or "Task completed"
                            )
                            # Don't show the last goal message in agents vis
                            # (it will be shown as text)
                            last_goal_message = None
                    except Exception:
                        pass  # If parsing fails, treat as normal message

            plan_temps: List[dict] = []
            need_show_singe_last_message = False

            # Get all keys to identify the last one
            message_keys = list(message_group.keys())

            for idx, key in enumerate(message_keys):
                is_last = idx == len(message_keys) - 1
                value = message_group[key]
                num = num + 1

                if key.startswith(NONE_GOAL_PREFIX):
                    vis_items.append(await self._messages_to_plan_vis(plan_temps))
                    plan_temps = []
                    num = 0
                    # 无 goal 分组通常出现在队尾，其最后一条消息按"最终消息"完整展示
                    vis_items.append(
                        await self._messages_to_agents_vis(value, is_last_message=True)
                    )
                else:
                    # Skip the last plan item if it was a terminate action
                    if is_last and terminate_content:
                        continue

                    num += 1
                    plan_temps.append(
                        {
                            "name": self._compact_title(key),
                            "num": num,
                            "status": "complete",
                            "agent": value[0].receiver if value else "",
                            "markdown": await self._messages_to_agents_vis(value),
                        }
                    )
                    need_show_singe_last_message = True

            if len(plan_temps) > 0:
                vis_items.append(await self._messages_to_plan_vis(plan_temps))
            if need_show_singe_last_message and last_goal_message:
                vis_items.append(
                    await self._messages_to_agents_vis([last_goal_message], True)
                )

            # If there was a terminate action, append its content as pure text
            if terminate_content:
                vis_items.append(terminate_content)

        return "\n".join(vis_items)

    async def agent_stream_message(
        self,
        message: Union[Dict, str],
        enable_vis_message: bool = True,
    ):
        """Get agent stream message."""
        messages_view = []
        if isinstance(message, dict):
            messages_view.append(
                {
                    "sender": message["sender"],
                    "receiver": message["receiver"],
                    "model": message["model"],
                    "markdown": message["markdown"],
                }
            )
        else:
            messages_view.append(
                {
                    "sender": "?",
                    "receiver": "?",
                    "model": "?",
                    "markdown": message,
                }
            )
        if enable_vis_message:
            return await vis_client.get(VisAgentMessages.vis_tag()).display(
                content=messages_view
            )
        else:
            return messages_view

    async def _plan_vis_build(self, plan_group: dict[str, list]):
        num: int = 0
        plan_items = []
        terminate_content = None  # Store terminate output content for pure text display

        # Convert dict to list to handle last item specially
        plan_entries = list(plan_group.items())

        for idx, (key, value) in enumerate(plan_entries):
            is_last = idx == len(plan_entries) - 1

            # Check if this is the last item and it's a terminate action
            if is_last and value:
                last_message = value[-1] if value else None
                if last_message and last_message.action_report:
                    try:
                        action_out = ActionOutput.from_dict(
                            json.loads(last_message.action_report)
                        )
                        # If this is a terminate action, extract content
                        # and skip plan display
                        if action_out and action_out.terminate:
                            terminate_content = (
                                action_out.content
                                or action_out.view
                                or "Task completed"
                            )
                            continue  # Skip adding to plan_items
                    except Exception:
                        pass  # If parsing fails, treat as normal plan item

            num = num + 1
            plan_items.append(
                {
                    "name": self._compact_title(key),
                    "num": num,
                    "status": "complete",
                    "agent": value[0].receiver if value else "",
                    "markdown": await self._messages_to_agents_vis(value),
                }
            )

        plan_vis = await self._messages_to_plan_vis(plan_items)

        # If there was a terminate action, append its content as pure text (not in plan)
        if terminate_content:
            return f"{plan_vis}\n{terminate_content}" if plan_vis else terminate_content

        return plan_vis

    async def simple_message(self, conv_id: str):
        """Get agent simple message."""
        messages_cache = self.messages_cache[conv_id]
        if messages_cache and len(messages_cache) > 0:
            messages = messages_cache
        else:
            messages = await blocking_func_to_async(
                self._executor, self.message_memory.get_by_conv_id, conv_id=conv_id
            )

        simple_message_list = []
        for message in messages:
            if message.sender == "Human":
                continue

            action_report_str = message.action_report
            view_info = message.content
            action_out = None
            if action_report_str and len(action_report_str) > 0:
                action_out = ActionOutput.from_dict(json.loads(action_report_str))
            if action_out is not None:
                view_info = action_out.content

            simple_message_list.append(
                {
                    "sender": message.sender,
                    "receiver": message.receiver,
                    "model": message.model_name,
                    "markdown": view_info,
                }
            )

        return simple_message_list

    async def app_link_chat_message(self, conv_id: str):
        """Get app link chat message."""
        messages = []
        if conv_id in self.messages_cache:
            messages_cache = self.messages_cache[conv_id]
            if messages_cache and len(messages_cache) > 0:
                start_round = (
                    self.start_round_map[conv_id]
                    if conv_id in self.start_round_map
                    else 0
                )
                messages = messages_cache[start_round:]
        else:
            messages = await blocking_func_to_async(
                self._executor, self.message_memory.get_by_conv_id, conv_id=conv_id
            )

        # VIS消息组装
        temp_group: Dict = {}
        app_link_message: Optional[GptsMessage] = None
        app_lanucher_message: Optional[GptsMessage] = None

        none_goal_count = 1
        for message in messages:
            if message.sender in [
                "Intent Recognition Expert",
                "App Link",
            ] or message.receiver in ["Intent Recognition Expert", "App Link"]:
                if (
                    message.sender in ["Intent Recognition Expert", "App Link"]
                    and message.receiver == "AppLauncher"
                ):
                    app_link_message = message
                if message.receiver != "Human":
                    continue

            if message.sender == "AppLauncher":
                if message.receiver == "Human":
                    app_lanucher_message = message
                continue

            current_gogal = message.current_goal

            last_goal = next(reversed(temp_group)) if temp_group else None
            if last_goal:
                last_goal_messages = temp_group[last_goal]
                if current_gogal:
                    if current_gogal == last_goal:
                        last_goal_messages.append(message)
                    else:
                        temp_group[current_gogal] = [message]
                else:
                    temp_group[f"{NONE_GOAL_PREFIX}{none_goal_count}"] = [message]
                    none_goal_count += 1
            else:
                if current_gogal:
                    temp_group[current_gogal] = [message]
                else:
                    temp_group[f"{NONE_GOAL_PREFIX}{none_goal_count}"] = [message]
                    none_goal_count += 1

        vis_items: list = []
        if app_link_message:
            vis_items.append(
                await self._messages_to_app_link_vis(
                    app_link_message, app_lanucher_message
                )
            )

        # 语义层折叠模式：缓存里存在单条“[语义层]”整行消息时，计划卡只保留这一条，
        # 其余真实消息（用户问题/校验说明/最终结果）按原顺序进入下方消息区展示。
        collapse_semlink = (not app_link_message) and any(
            key == SEMANTIC_LAYER_GOAL for key in temp_group
        )
        if collapse_semlink:
            return await self._semlink_collapsed_vis(temp_group, vis_items)

        return await self._message_group_vis_build(temp_group, vis_items)

    async def _semlink_collapsed_vis(
        self, temp_group: Dict, vis_items: list
    ) -> str:
        """语义层折叠模式的整帧组装：计划卡一行 + 消息区气泡。

        - 计划卡：只保留“[语义层]:正在查询语义层资料完成检索”这一行，行内不放
          agent-messages 气泡，各阶段文字分行放入 markdown；
        - 消息区：语义层行之外的真实消息按缓存顺序渲染（用户问题裁剪为真问题、
          最终成功消息完整展示图/表/SQL，失败的中间轮只给一句话原因）。
        """
        plan_lines: List[str] = []
        other_messages: List[GptsMessage] = []
        for key, value in temp_group.items():
            if key == SEMANTIC_LAYER_GOAL:
                for m in value:
                    content = (m.content or "").strip()
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            plan_lines.append(line)
            else:
                other_messages.extend(value)
        if not plan_lines:
            plan_lines.append("语义层：正在检索语义层资料、选表并注入表结构")

        # 是否已产出“最终成功”消息：已成功 -> 整行收尾为 complete；执行中 -> running
        done = False
        if other_messages:
            last = other_messages[-1]
            if getattr(last, "is_success", False) and last.action_report:
                try:
                    action_out = ActionOutput.from_dict(
                        json.loads(last.action_report)
                    )
                    done = bool(action_out and action_out.is_exe_success)
                except Exception:
                    done = True
        plan_item = {
            "name": SEMANTIC_LAYER_GOAL,
            "num": 1,
            "status": "complete" if done else "running",
            "agent": "语义层",
            # 行前加不可断空格，避免段落以 "1. " 开头被 markdown 解析成有序列表
            "markdown": "<br/>".join(
                f"\u00a0{idx}. {line}"
                for idx, line in enumerate(plan_lines, start=1)
            ),
        }
        if vis_items:
            # 理论上前方不应有 app_link 类 vis 项（collapse 模式已排除），保险起见仍拼接
            return "\n".join(vis_items) + "\n" + await self._messages_to_plan_vis(
                [plan_item]
            )
        plan_vis = await self._messages_to_plan_vis([plan_item])
        if not other_messages:
            return plan_vis
        msg_vis = await self._messages_to_agents_vis(
            other_messages, is_last_message=True
        )
        return "\n".join(filter(None, [plan_vis, msg_vis]))

    async def _messages_to_agents_vis(
        self, messages: List[GptsMessage], is_last_message: bool = False
    ):
        """把消息列表渲染成 agent-messages 气泡帧。

        展示口径（与产品确认）：
        - 中间/失败/重试轮：不展示整段 SQL、图表与结果明细，只保留一句话说明
          （thought 或执行错误，校验失败原因由紧随其后的文本消息给出）；
        - 最终一条消息（is_last_message=True 时最后的气泡）：完整展示 view
          （图/表 + 最终 SQL + 结论 thought）；
        - 无 ActionReport 的文本（用户问题/语义层大文本/校验反馈）：裁剪，语义层注入
          的大文本只保留末尾真正的"用户问题"部分。
        裁剪只发生在"生成帧"这一步，不影响消息落库（messages_cache 仍是全量）。
        """
        if messages is None or len(messages) <= 0:
            return ""
        messages_view = []
        last_idx = len(messages) - 1
        for idx, message in enumerate(messages):
            # 只有"真正校验通过的最后一条消息"才完整展示。重试过程中失败的中间轮
            # 即使正好是队列末尾（is_last_message=True），也不得携带完整图表反复入帧。
            is_final_msg = (
                is_last_message
                and idx == last_idx
                and bool(getattr(message, "is_success", False))
            )
            action_report_str = message.action_report
            view_info = message.content
            if action_report_str and len(action_report_str) > 0:
                try:
                    action_out = ActionOutput.from_dict(json.loads(action_report_str))
                except Exception:
                    action_out = None
                if action_out is not None:
                    if is_final_msg and action_out.is_exe_success:
                        # 最终成功消息：完整展示（图/表 + 最终 SQL + 结论）
                        view = action_out.view
                        view_info = view if view else action_out.content
                    else:
                        view_info = self._agent_bubble_summary(
                            action_out, message.content
                        )
            else:
                view_info = self._plain_text_display(message.content)

            messages_view.append(
                {
                    "sender": message.sender,
                    "receiver": message.receiver,
                    "model": message.model_name,
                    "markdown": view_info,
                    "resource": (
                        message.resource_info if message.resource_info else None
                    ),
                }
            )
        return await vis_client.get(VisAgentMessages.vis_tag()).display(
            content=messages_view
        )

    @classmethod
    def _plain_text_display(cls, text: Optional[str], limit: int = 800) -> str:
        """展示无 ActionReport 的文本气泡。

        - 语义层注入的 user 消息很长（允许表 + 完整表结构 + 召回 SQL + 用户问题），
          过程帧不需要这些中间产物，只保留末尾真正的"用户问题"；
        - 校验失败原因、语义层阶段文本等较短，超长时裁剪首尾。
        """
        if not text:
            return ""
        content = str(text).strip()
        if content.startswith("允许使用的表及表间关联关系"):
            marker = "用户问题:\n"
            idx = content.rfind(marker)
            if idx != -1:
                content = content[idx + len(marker) :].strip()
        return cls._compact_display(content, limit)

    @classmethod
    def _agent_bubble_summary(
        cls, action_out: ActionOutput, content: Optional[str], limit: int = 600
    ) -> str:
        """中间/失败轮次只给一句话说明，不携带 SQL、图表与结果明细。

        执行失败轮：错误本身就是要看的原因；
        执行成功但未过校验的中间轮：展示其 thought（若 JSON 可解析），否则给占位说明，
        具体的"校验失败原因"由紧随其后的文本消息展示。
        """
        if action_out is not None and not action_out.is_exe_success:
            err = action_out.content or content or "该步骤执行失败"
            return cls._compact_display(str(err).strip(), limit)
        thought = cls._extract_json_thought(content)
        if thought:
            return cls._compact_display(thought, limit)
        return "（已生成并执行 SQL，正在进行结果校验…）"

    @staticmethod
    def _extract_json_thought(text: Optional[str]) -> Optional[str]:
        """从 ActionOutput.content 的 JSON 里取出 thought 文本（不取 sql/data）。"""
        if not text:
            return None
        try:
            content = str(text)
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end <= start:
                return None
            obj = json.loads(content[start : end + 1])
        except Exception:
            return None
        if isinstance(obj, dict):
            thought = obj.get("thought")
            if isinstance(thought, str) and thought.strip():
                return thought.strip()
        return None

    @staticmethod
    def _compact_title(text: Optional[str], limit: int = 160) -> str:
        """计划卡条目标题过长时裁剪（如 current_goal 携带语义层大文本）。"""
        if not text:
            return ""
        content = str(text).strip().replace("\n", " ")
        if len(content) <= limit:
            return content
        return content[:limit] + "…"

    @staticmethod
    def _compact_display(text: Optional[str], limit: int = 1500) -> str:
        """裁剪超长展示文本，只保留首尾，中间用省略说明代替。"""
        if not text:
            return ""
        content = str(text)
        if len(content) <= limit:
            return content
        head_len = int(limit * 0.6)
        tail_len = int(limit * 0.35)
        head = content[:head_len]
        tail = content[-tail_len:]
        omitted = len(content) - head_len - tail_len
        return f"{head}\n……[中间省略 {omitted} 字符]……\n{tail}"

    async def _messages_to_plan_vis(self, messages: List[Dict]):
        if messages is None or len(messages) <= 0:
            return ""
        return await vis_client.get(VisAgentPlans.vis_tag()).display(content=messages)

    async def _messages_to_app_link_vis(
        self, link_message: GptsMessage, lanucher_message: Optional[GptsMessage] = None
    ):
        logger.info("app link vis build")
        if link_message is None:
            return ""
        param = {}
        link_report_str = link_message.action_report
        if link_report_str and len(link_report_str) > 0:
            action_out = ActionOutput.from_dict(json.loads(link_report_str))
            if action_out is not None:
                if action_out.is_exe_success:
                    temp = json.loads(action_out.content)

                    param["app_code"] = temp["app_code"]
                    param["app_name"] = temp["app_name"]
                    param["app_desc"] = temp.get("app_desc", "")
                    param["app_logo"] = ""
                    param["status"] = Status.RUNNING.value

                else:
                    param["status"] = Status.FAILED.value
                    param["msg"] = action_out.content

        if lanucher_message:
            lanucher_report_str = lanucher_message.action_report
            if lanucher_report_str and len(lanucher_report_str) > 0:
                lanucher_action_out = ActionOutput.from_dict(
                    json.loads(lanucher_report_str)
                )
                if lanucher_action_out is not None:
                    if lanucher_action_out.is_exe_success:
                        param["status"] = Status.COMPLETE.value
                    else:
                        param["status"] = Status.FAILED.value
                        param["msg"] = lanucher_action_out.content
        else:
            param["status"] = Status.COMPLETE.value
        return await vis_client.get(VisAppLink.vis_tag()).display(content=param)

    async def chat_messages(
        self,
        conv_id: str,
    ):
        """Get chat messages."""
        while True:
            queue = self.queue(conv_id)
            if not queue:
                break
            item = await queue.get()
            if item == "[DONE]":
                queue.task_done()
                break
            else:
                yield item
                await asyncio.sleep(0.005)
