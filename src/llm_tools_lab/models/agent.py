from pydantic import BaseModel


class AgentResponse(BaseModel):
    message: str
    response_time_ms: int
    tool_calls_count: int
