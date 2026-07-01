import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agent import handle_chat
from app.retrieval import get_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up():
    # Pre-build the retriever (loads/embeds the catalog) at startup rather
    # than on the first request, so the first real /chat call isn't slow.
    logger.info("Warming up retriever...")
    get_retriever()
    logger.info("Retriever ready.")

@app.get("/")
def root():
    return {
        "service": "SHL Assessment Recommender",
        "status": "running",
        "endpoints": {"health": "/health", "chat": "/chat (POST)", "docs": "/docs"},
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return handle_chat(request.messages)
