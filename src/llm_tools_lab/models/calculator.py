from pydantic import BaseModel, Field


class CalculatorResponse(BaseModel):
    steps: list[str] = Field(description="A list of steps about the operations")
    result: float = Field(description="The result of the operation")
