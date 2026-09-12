# 智能体对话 SSE 输出说明

适用接口：`POST /api/v1/chat/completions`（`chat_mode = chat_agent`）

---

## 格式

| 项 | 说明 |
|---|---|
| 分隔符 | 事件之间用空行分隔 |
| 数据行 | `data:` 后接 JSON，`data:` 后**无空格** |
| 心跳 | `:` 开头，忽略 |
| 结束 | `{"vis":"[DONE]"}` |
| 异常 | `data:` 后不是 JSON 时，该帧跳过 |
| 编码 | UTF-8 |

---

## `agent-plans` 字段说明

payload 为 JSON 数组，每个元素即一条计划行。

| 字段 | 类型 | 使用说明 |
|---|---|---|
| `name` | string | 计划行标题，格式 `[<执行者>]:<描述>` |
| `num` | integer | 步骤序号，帧内唯一，用于排序与同步骤覆盖 |
| `status` | string | 步骤状态 |
| `agent` | string | 执行者名称 |
| `markdown` | string | 步骤正文 |

---

## `agent-messages` 字段说明

payload 为 JSON 数组，每个元素即一个气泡。

| 字段 | 类型 | 使用说明 |
|---|---|---|
| `sender` | string | 发送方 |
| `receiver` | string | 接收方，气泡方向为 `sender → receiver` |
| `model` | string \| null | 产出该气泡的模型名 |
| `markdown` | string | 气泡正文，可为空串，可能内嵌其它围栏 |
| `resource` | object \| array \| null | 引用资源，无则 null |

