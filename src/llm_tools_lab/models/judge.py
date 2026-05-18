from pydantic import BaseModel, Field, model_validator


class JudgeResponse(BaseModel):
    accuracy: int = Field(
        description="Factual correctness based on context", ge=1, le=5
    )
    relevance: int = Field(description="Answers the question", ge=1, le=5)
    completeness: int = Field(description="Response completeness", ge=1, le=5)
    reasoning: str = Field(description="Brief explanation of the scores")
    overall_score: float = Field(default=0.0, description="Average of all scores")

    @model_validator(mode="after")
    def calculate_overall(self) -> "JudgeResponse":
        self.overall_score = round(
            (self.accuracy + self.relevance + self.completeness) / 3, 2
        )
        return self
