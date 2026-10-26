# Project LUMEN

**Logical User Monitoring & Explainable Network Defense**
*A privacy-first, explainable AI approach to phishing detection.*

LUMEN detects phishing without relying on a cloud service that reads your messages. A small **local LLM** (run through [Ollama](https://ollama.com)) acts as a frozen feature extractor, and a **GraphSAGE** graph neural network makes the final Safe / Phishing decision from the email's metadata graph.

> **Status:** research prototype built for the Smart India Hackathon 2026. The ML pipeline in this repository is implemented and runnable. The wider ecosystem (browser extension, SHAP alerts, honey-tokens, federated-lite learning) is the planned architecture and is **not** part of this codebase yet. See [Scope](#scope-implemented-vs-planned).

---

## Why LUMEN

| Problem | LUMEN's approach |
| :-- | :-- |
| Security tools scan message content on central servers, which is a privacy trade-off | Inference runs locally. The LLM runs on-device through Ollama, with no cloud or external API calls |
| Generic "suspicious message" banners cause alert fatigue | The LLM produces interpretable semantic scores (urgency, threat, impersonation, ...) that explain *why* a message is risky |
| Keyword filters are easy to evade | A graph model combines metadata signals with semantic scores |

## How it works

```
                    RAW EMAIL
                        |
        +---------------+----------------+
        |               |                |
     Headers           URLs           Wording
        |               |                |
        v               v                v
 deterministic       regex       local Ollama LLM
   parser            parser      (frozen extractor)
        |               |                |
        +---------------+----------------+
                        |
                        v
                   EMAIL GRAPH
                        |
                    GraphSAGE
                        |
               Safe / Phishing
```

**The LLM is never the classifier.** It only extracts six scores in the range 0-1; GraphSAGE is the only trained model.

### Graph structure

Each email becomes a small graph:

- **Email node:** `reply_to_mismatch`, `keyword_urgency`, plus six LLM scores: `urgency`, `credential_request`, `threat`, `impersonation`, `financial_request`, `suspicious_wording`
- **Sender domain node:** `normalized_domain_length`, `is_ip_domain`
- **URL nodes:** `is_ip_url`, `uses_shortener`
- **Edges:** Email → Domain and Email → each URL

The model uses a node-type-aware input layer (a separate projection per node type plus a learned type embedding), followed by two `SAGEConv` layers and `global_mean_pool`.

### Privacy by design

- No TF-IDF, bag-of-words, word embeddings or generic text classifiers over the email body
- The body is only passed to the **local** LLM, truncated to `MAX_BODY_CHARS = 5000`
- LLM outputs are cached in `llm_features.json`, so repeated runs make no new LLM calls
- Malformed LLM output falls back to neutral scores and is logged. Unparseable emails are skipped

## Project structure

```
.
├── config.py                      # Global configuration (seed, model, paths, hyperparameters)
├── train.py                       # End-to-end pipeline: data -> features -> graphs -> train -> evaluate
├── try.py                         # Inference and inspection on held-out emails
├── requirements.txt
├── instructions.txt               # Original build specification for the prototype
├── lumen/
│   ├── dataset.py                 # Downloads and parses the SpamAssassin corpus
│   ├── features_deterministic.py  # Reply-To mismatch, keyword urgency, domain and URL features
│   ├── features_llm.py            # Ollama feature extraction with JSON parsing and caching
│   ├── graph.py                   # Builds PyTorch Geometric graphs
│   ├── model.py                   # Node-type-aware GraphSAGE classifier
│   └── eval.py                    # Training loop, prediction and metrics
├── lumen_hybrid_graphsage.pt                    # Hybrid checkpoint (smollm2:360m)
├── lumen_hybrid_graphsage_weights.pth           # Weights only
├── lumen_hybrid_graphsage_smollm2_1_7b.pt       # Hybrid checkpoint (smollm2:1.7b)
└── lumen_hybrid_graphsage_smollm2_1_7b_weights.pth
```

## Getting started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed, with the default model pulled:

```bash
ollama pull smollm2:360m
ollama serve        # keep this running (default: http://localhost:11434)
```

### Install

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### Train and evaluate

```bash
python train.py
```

This will:

1. Download and parse the SpamAssassin Public Corpus automatically
2. Extract LLM features, using the cache where available
3. Build email graphs and make a stratified 80/20 train/test split (`SEED = 42`)
4. Train **GNN-only** and **LLM + GNN** models
5. Print accuracy for the keyword baseline, GNN-only and the hybrid model, with precision, recall and F1
6. Save the hybrid checkpoint and weights

### Inspect predictions

```bash
python try.py
```

Loads the saved checkpoint and prints:

- 10 random held-out emails with metadata, deterministic features and all six LLM scores
- One example each of a true positive, true negative, false positive and false negative
- A NetworkX graph visualization of one test email (`data/lumen_graph_viz.png`). Set `NO_DISPLAY=1` on headless machines.

### Using a different Ollama model

```bash
LUMEN_OLLAMA_MODEL=smollm2:1.7b python train.py
```

Cache, checkpoint and results files are named per model, so runs do not overwrite each other.

## Experiments

`train.py` compares three approaches on the same held-out test set:

| Approach | Description |
| :-- | :-- |
| **A. Keyword baseline** | Predicts phishing whenever `keyword_urgency = 1` |
| **B. GNN only** | GraphSAGE on deterministic features only |
| **C. LLM + GNN** | GraphSAGE on deterministic + LLM features |

The script prints the measured accuracy and the improvement of C over B (`hybrid - GNN-only`). It reports whatever the data shows and does not assume the LLM helps. Results are also saved to `data/cache/results.json`.

## Configuration

Key settings in `config.py`:

| Setting | Default |
| :-- | :-- |
| `SEED` | `42` |
| `OLLAMA_MODEL` | `smollm2:360m` (override with `LUMEN_OLLAMA_MODEL`) |
| `OLLAMA_CONTEXT` | `8192` |
| `OLLAMA_TEMPERATURE` | `0.0` (deterministic) |
| `MAX_BODY_CHARS` | `5000` |
| `HIDDEN_DIM` / `NUM_SAGE_LAYERS` | `64` / `2` |
| `GNN_EPOCHS` / `LEARNING_RATE` | `30` / `0.001` |
| `TEST_SPLIT` | `0.2` |

## Dataset

[SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/):

- Spam: `20050311_spam_2`
- Ham: `20030228_easy_ham_2`

Downloaded and extracted automatically on first run into `data/`.

## Scope: implemented vs. planned

**Implemented in this repository**

- Deterministic header, URL and domain feature extraction
- Local LLM semantic feature extraction through Ollama, with caching and robust JSON parsing
- Node-type-aware GraphSAGE classifier
- Keyword vs. GNN-only vs. LLM + GNN ablation
- Inspection tooling and graph visualization

**Planned architecture (not yet in this codebase)**

- **Edge deployment:** browser extension / mobile app in React + TypeScript, with TensorFlow.js or ONNX Runtime Web inference
- **Explainability:** SHAP-based feature attribution turned into plain-English alerts
- **Active defense:** DOM link stripping and honey-token credential injection
- **Federated-lite learning:** server-side pretraining on PhishTank / OpenPhish, local fine-tuning, Gaussian differential-privacy noise on weight deltas, and cohort FedAvg
- **Multi-channel support:** SMS and enterprise chat (Slack/Teams)

## References

- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.*
- McMahan, H. B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* (FedAvg)
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* (SHAP)
- Wu, Z., et al. (2020). *A Comprehensive Survey on Graph Neural Networks.*

## Author

**Arkadyuti Bhattacharjee**, IIITM Gwalior
