"""
KeaBuilder  Lead/Prompt Similarity Engine
Q1: Find the most similar input to a given query using cosine similarity
"""

import math
from typing import List, Tuple



# ── Simple TF-IDF-style vector builder (no external deps) ──────────────────

def tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_vocab(corpus: List[str]) -> List[str]:
    vocab = set()
    for doc in corpus:
        vocab.update(tokenize(doc))
    return sorted(vocab)


def vectorize(text: str, vocab: List[str]) -> List[float]:
    tokens = tokenize(text)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [freq.get(word, 0) for word in vocab]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Sample KeaBuilder lead/prompt inputs ───────────────────────────────────

SAMPLE_INPUTS = [
    "I want to build a sales funnel for my e-commerce store",
    "How do I capture leads from my landing page?",
    "Set up an automated email sequence for new subscribers",
    "Create a chatbot to answer customer questions on my website",
    "Generate AI content for my product description pages",
]


# ── Core similarity function ────────────────────────────────────────────────

def find_most_similar(query: str, corpus: List[str]) -> Tuple[str, float, int]:
    """
    Returns (best_match_text, similarity_score, index)
    """
    all_texts = corpus + [query]
    vocab = build_vocab(all_texts)

    query_vec = vectorize(query, vocab)
    scores = []

    for text in corpus:
        vec = vectorize(text, vocab)
        score = cosine_similarity(query_vec, vec)
        scores.append(score)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return corpus[best_idx], round(scores[best_idx], 4), best_idx


def find_top_k(query: str, corpus: List[str], k: int = 3) -> List[dict]:
    """
    Returns top-k matches sorted by similarity descending.
    """
    all_texts = corpus + [query]
    vocab = build_vocab(all_texts)
    query_vec = vectorize(query, vocab)

    results = []
    for i, text in enumerate(corpus):
        vec = vectorize(text, vocab)
        score = cosine_similarity(query_vec, vec)
        results.append({"index": i, "text": text, "score": round(score, 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


# ── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "I need an automated follow-up email flow for leads"

    print("=" * 60)
    print("KeaBuilder Similarity Engine")
    print("=" * 60)
    print(f"\nQuery: \"{query}\"\n")

    best_text, best_score, best_idx = find_most_similar(query, SAMPLE_INPUTS)
    print(f" Top Match (index {best_idx}):")
    print(f"   \"{best_text}\"")
    print(f"   Similarity Score: {best_score}\n")

    print(" Top 3 Matches:")
    for rank, match in enumerate(find_top_k(query, SAMPLE_INPUTS, k=3), 1):
        print(f"  {rank}. [{match['score']}] {match['text']}")

    print("\n" + "=" * 60)



