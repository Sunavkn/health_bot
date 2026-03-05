from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.summarize import router as summarize_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="On-Device Medical Assistant API",
    version="0.1.0",
    description="Private, safety-first medical assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(summarize_router, prefix="/summarize", tags=["summarize"])

@app.get("/health")
def health():
    return {"status": "ok"}
