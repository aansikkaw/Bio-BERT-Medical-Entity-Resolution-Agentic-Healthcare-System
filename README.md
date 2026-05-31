# 🧬 Bio-BERT Medical Entity Resolution & Agentic Healthcare System

<div align="left">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge" alt="LangChain" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="GCP" />
</div>

## 📌 Executive Summary
This project demonstrates an end-to-end Deep Learning and MLOps pipeline designed for the healthcare domain. It leverages a custom-trained **Bio-BERT** model optimized with contrastive learning to perform highly accurate medical entity resolution. Furthermore, the architecture is extended into an **Agentic AI ecosystem** via a Model Context Protocol (MCP) server, integrating real-world APIs with synthetic patient data to generate predictive health alerts.

---

## 🛠️ Deep Learning Architecture & Training

### Model Fine-Tuning & Optimization
* **Base Architecture:** Utilized a pre-trained Bio-BERT model, adapting its contextual embeddings for domain-specific entity resolution.
* **Triplet Network Design:** Engineered a custom PyTorch training loop utilizing a Triplet Dataset structure (Anchor, Positive, Negative) to map semantically similar medical terms closer together in the vector space while pushing apart distinct entities.
* **Loss Optimization:** Implemented contrastive loss functions to strictly penalize embedding overlaps between distinct medical codes, optimizing the manifold for semantic search retrieval.

### Performance Metrics
| Metric | Result |
| :--- | :--- |
| **Training Loss** | `0.0035` (Converged in 3 Epochs) |
| **Evaluation Accuracy** | **100%** |
| **Similarity Metric** | Cosine Similarity |
| **Decision Threshold** | `> 0.90` |

---

## 🚀 MLOps & Agentic AI Integration

### Scalable Deployment
* **Containerization:** The inference pipeline and API dependencies are entirely containerized using Docker, guaranteeing zero environment drift between local and production stages.
* **Cloud Infrastructure:** Deployed as a high-concurrency microservice on Google Cloud Platform (GCP) utilizing the asynchronous capabilities of FastAPI.

### Agentic AI Development (In Progress)
* **Model Context Protocol (MCP):** Engineering a LangChain-powered AI Agent operating on a Fast-MCP server architecture.
* **Context-Aware Generation:** The agent autonomously fetches real-time environmental data (via the OpenWeatherMap API) and cross-references it with synthetic Electronic Health Records (EHRs) to output proactive, context-aware health alerts for vulnerable patient profiles.

---

## 💻 System Execution

```bash
git clone [https://github.com/aansikkaw/bio-bert-entity-resolution.git](https://github.com/aansikkaw/bio-bert-entity-resolution.git)
cd bio-bert-entity-resolution

pip install -r requirements.txt

docker build -t biobert-api .
docker run -d -p 8000:8000 biobert-api
