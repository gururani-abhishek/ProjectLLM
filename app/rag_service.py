import os
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "context_data.json")

_client = OpenAI()

def load_context_data(path: str = DATA_PATH) -> List[Dict]:
    with open(path, "r") as f:
        raw = json.load(f)
    data = [r for r in raw if r.get("field")]
    return data

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def build_doc_text(item: Dict) -> str:
    parts = [
        f"Field: {item.get('field','')}",
        f"Meaning: {item.get('meaning','')}",
        f"Importance: {item.get('importance','')}",
        f"WhyRecord: {item.get('why_record','')}",
        f"Example: {item.get('example','')}",
        f"GapReason: {item.get('gap_reason','')}",
    ]
    return " | ".join(p for p in parts if p.split(':',1)[-1].strip())

_CONTEXT_DATA = load_context_data()
_DOC_TEXTS = [build_doc_text(d) for d in _CONTEXT_DATA]
_DOC_EMB = embed_texts(_DOC_TEXTS)
_DOC_NORM = _DOC_EMB / np.linalg.norm(_DOC_EMB, axis=1, keepdims=True)

def retrieve(query: str, top_k: int = 5, field_filter: Optional[str] = None) -> List[Tuple[Dict, float]]:
    q_emb = embed_texts([query])[0]
    q_emb = q_emb / np.linalg.norm(q_emb)
    sims = (_DOC_NORM @ q_emb)
    idx_scores = list(enumerate(sims))
    if field_filter:
        idx_scores = [(i,s) for i,s in idx_scores if field_filter.lower() in _CONTEXT_DATA[i]["field"].lower()] or idx_scores
    top = sorted(idx_scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [(_CONTEXT_DATA[i], float(score)) for i, score in top]

def build_prompt(user_query: str, retrieved: List[Tuple[Dict, float]]) -> str:
    if not retrieved:
        context = "No relevant SDLC references retrieved."
    else:
        context = "\n\n".join(
            "\n".join(
                f"{k.capitalize()}: {doc.get(k, '')}"
                for k in ["field", "meaning", "importance", "why_record", "example", "gap_reason"]
            )
            for doc, score in retrieved
        )
    prompt = f"""You are an SDLC expert assistant.
Here is some context:
{context}

User question: {user_query}
Answer the question only using the context above,
and make them believe it is an important field to record in SDLC."""
    return prompt

def call_chatgpt(prompt: str) -> str:
    resp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You provide structured SDLC guidance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

def rag_pipeline(
    query: str
) -> Dict:
    if not is_sdlc_related(query):
        return {
            "answer": "Your question does not appear to have explicity mentioned SDLC gap type/s. Please read the note below and ask an SDLC-related question.",
            "retrieved": []
        }
    retrieved = retrieve(query, top_k=1)
    prompt = build_prompt(query, retrieved)
    answer = call_chatgpt(prompt)
    return {
        "answer": answer
    }

def is_sdlc_related(query: str) -> bool:
    check_prompt = (
        "You are an expert in software development lifecycle (SDLC). "
        "Does the following query relate to SDLC concepts, phases, documentation, or practices? "
        "Reply only with 'yes' or 'no'.\n\n"
        f"Query: {query}"
    )
    resp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an SDLC expert."},
            {"role": "user", "content": check_prompt}
        ],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip().lower()
    return answer.startswith("y")
