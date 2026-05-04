import instructor
from instructor.exceptions import InstructorRetryException

from llm_tools_lab.models.weather import WeatherForecast


def get_weather_forecast(city: str, model: str = "qwen3:8b") -> WeatherForecast:
    """Get structured weather forecast for a city"""
    # Creates instructor client
    client = instructor.from_provider(f"ollama/{model}", async_client=False)
    try:
        # Calls to model
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a weather assistant. Return weather data for the requested city.",
                },
                {
                    "role": "user",
                    "content": f"What's the weather like in {city} today?",
                },
            ],
            response_model=WeatherForecast,
            max_retries=3,
        )
        return response
    except InstructorRetryException as e:
        raise ValueError(
            f"Model failed to return valid structured output after retries: {e}"
        ) from e
