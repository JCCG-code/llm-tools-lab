from typing import Literal

from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    sport: Literal["Basketball", "Football", "Tennis"] = Field(
        description="The sport category of the input"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the classification"
    )
