from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import AskRequest, AskResponse
from graph import ask_zepto


app = FastAPI(
    title="Zepto Support Assistant"
)


# Allow frontend running on port 5500
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant API running"
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    response = ask_zepto(
        request.query
    )

    return response