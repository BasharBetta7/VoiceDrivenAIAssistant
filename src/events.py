from typing import Callable, Any, Literal, TypedDict

class AgentEvent(TypedDict):
    type: Literal["status", "message", "message_delta", "tool_started", "tool_result", "user_prompt", "error_message"]
    data: dict[str, Any]

EventHandler = Callable[[AgentEvent], None]
