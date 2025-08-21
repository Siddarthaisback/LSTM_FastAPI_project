from pydantic import BaseModel, Field
from typing import Optional

class WeatherData(BaseModel):
    """Pydantic model for weather data input validation."""
    MinTemp: Optional[float] = Field(None, description="Minimum temperature")
    MaxTemp: Optional[float] = Field(None, description="Maximum temperature")
    Rainfall: Optional[float] = Field(None, description="Rainfall amount")
    Evaporation: Optional[float] = Field(None, description="Evaporation amount")
    Sunshine: Optional[float] = Field(None, description="Hours of sunshine")
    WindGustSpeed: Optional[float] = Field(None, description="Wind gust speed")
    WindSpeed9am: Optional[float] = Field(None, description="Wind speed at 9am")
    WindSpeed3pm: Optional[float] = Field(None, description="Wind speed at 3pm")
    Humidity9am: Optional[float] = Field(None, description="Humidity at 9am")
    Humidity3pm: Optional[float] = Field(None, description="Humidity at 3pm")
    Pressure9am: Optional[float] = Field(None, description="Pressure at 9am")
    Pressure3pm: Optional[float] = Field(None, description="Pressure at 3pm")
    Cloud9am: Optional[float] = Field(None, description="Cloud cover at 9am")
    Cloud3pm: Optional[float] = Field(None, description="Cloud cover at 3pm")
    Temp9am: Optional[float] = Field(None, description="Temperature at 9am")
    Temp3pm: Optional[float] = Field(None, description="Temperature at 3pm")

    class Config:
        # Allow population by field name or alias
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "Humidity3pm": 50.0,
                "Pressure3pm": 1010.0,
                "Cloud3pm": 4.0,
                "Rainfall": 0.0,
                "WindGustSpeed": 40.0
            }
        }