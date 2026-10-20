"""Request and response models for the prediction API.

Field limits mirror the clinically plausible ranges used during cleaning, so
obviously wrong inputs (age 300, negative cholesterol) are rejected with a 422
before they ever reach the model.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 54, "sex": 1, "cp": 4, "trestbps": 140, "chol": 239, "fbs": 0, "restecg": 0,
    "thalach": 160, "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 3,
}


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_PATIENT})

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=70, le=220, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(
        ..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="ST slope: 1 up, 2 flat, 3 down")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(
        ..., description="Thallium test: 3 normal, 6 fixed defect, 7 reversible defect")


class BatchRequest(BaseModel):
    patients: list[Patient] = Field(..., min_length=1, max_length=100)


class Factor(BaseModel):
    feature: str
    effect: Literal["raises_risk", "lowers_risk"]
    weight: float = Field(..., description="Contribution to the log-odds of disease")


class Prediction(BaseModel):
    prediction: Literal[0, 1] = Field(..., description="1 = heart disease predicted")
    label: Literal["heart_disease", "no_heart_disease"]
    probability: float = Field(..., description="Probability of heart disease")
    confidence: float = Field(..., description="Probability of the predicted class")
    risk_band: Literal["low", "moderate", "high"]
    top_factors: list[Factor] = Field(default_factory=list,
                                      description="Features that pushed this prediction most")


class PredictResponse(Prediction):
    model_version: str
    request_id: str


class BatchResponse(BaseModel):
    model_version: str
    request_id: str
    count: int
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
    uptime_seconds: float
