from pydantic import BaseModel, Field


class WeatherForecast(BaseModel):
    city: str = Field(description="The name of the city")
    temperature: float = Field(description="The current temperature in Celsius")
    condition: str = Field(
        description="The weather condition, e.g. sunny, cloudy, rainy"
    )
    humidity: int = Field(description="The humidity percentage 0-100", ge=0, le=100)
    summary: str = Field(description="A brief summary of the current weather")
