"""
KeaBuilder Similarity API
Run: uvicorn api:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from similarity import find_most_similar, find_top_k, SAMPLE_INPUTS

app = FastAPI(title="KeaBuilder Similarity API", version="1.0")


class QueryRequest(BaseModel):
    query: str
    corpus: List[str] = SAMPLE_INPUTS   # use defaults or pass your own
    top_k: int = 1


class MatchResult(BaseModel):
    index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    matches: List[MatchResult]


@app.get("/")
def root():
    return {"message": "KeaBuilder Similarity Engine is running"}


@app.post("/find-similar", response_model=QueryResponse)
def find_similar(request: QueryRequest):
    results = find_top_k(request.query, request.corpus, k=request.top_k)
    return QueryResponse(
        query=request.query,
        matches=[MatchResult(**r) for r in results]
    )


# Example curl:
# curl -X POST http://localhost:8000/find-similar \
#      -H "Content-Type: application/json" \
#      -d '{"query": "email automation for new leads", "top_k": 2}'

