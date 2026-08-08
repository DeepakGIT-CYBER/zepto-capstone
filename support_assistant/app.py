from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from models import AskRequest, AskResponse
from graph import ask_zepto

app = FastAPI(
    title="Zepto Support Assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    response = ask_zepto(
        request.query
    )
    return response