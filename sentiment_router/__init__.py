from fastapi import APIRouter, Body
from typing import Annotated
from .schema import (
    SentimentModelPredictPayload,
    SentimentModelPredictResponse
)
from models.inference import load_model, inference

router = APIRouter()
sentiment_classifier = load_model(
    vectorizer_path = "./models/text_count_based_vectorizer.pkl",
    model_path = "./models/naive_bayes_model.pkl"
)

@router.post(
    path = "/predict",
    response_model = SentimentModelPredictResponse,
    responses = {
        200: {
            "description": "The prediction done successfully!",
            "content": {
                "application/json":{
                    "example": [
                        {"label": "Negative", "confidence_score": 0.23}
                    ]
                }
            }
        }
    }
)
def predict(
    payload: Annotated[
        SentimentModelPredictPayload, 
        Body(examples = [{"text": ["this is very bad product!"]}])
    ]
):
    prediction = inference(payload.text, sentiment_classifier.model, sentiment_classifier.vectorizer, ["negative", "neutral", "positive"])
    return {
        "prediction": prediction
    }
# End Func
    