from fastapi import FastAPI
from sentiment_router import router as sentiment_api_router

app = FastAPI()

app.include_router(sentiment_api_router, prefix = "/sentiment_api")