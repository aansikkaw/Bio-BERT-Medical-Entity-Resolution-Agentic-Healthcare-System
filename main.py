from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

app = FastAPI(title="Bio-BERT Entity Resolution API")

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

class TextPair(BaseModel):
    anchor: str
    comparison: str

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

@app.post("/predict")
def predict_similarity(pair: TextPair):
    emb1 = get_embedding(pair.anchor)
    emb2 = get_embedding(pair.comparison)
    
    cosine_sim = F.cosine_similarity(emb1, emb2).item()
    
    is_match = bool(cosine_sim > 0.90)
    
    return {
        "anchor": pair.anchor,
        "comparison": pair.comparison,
        "cosine_similarity": round(cosine_sim, 4),
        "is_match": is_match
    }
