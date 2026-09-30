from pydantic import BaseModel
from typing import List

class SentimentModelPredictPayload(BaseModel):
    text: List[str]
# End Class

class Prediction(BaseModel):
    label: str
    confidence_score: float
# End Class

class SentimentModelPredictResponse(BaseModel):
    prediction: List[Prediction]
# End Class