# 🔄 Recursive Language Models (RLMs) — Convergence Research

> **Adaptive Convergence Detection, Intervention Strategies & Auto-Optimization for Self-Correcting LLM Pipelines**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This project investigates **Recursive Language Models (RLMs)** — systems where an LLM iteratively refines its own output until convergence is achieved. We built a complete research pipeline spanning 5 phases, from naive baseline loops to a publication-ready, self-improving architecture.

### Key Question
> *When an LLM is asked to recursively improve its own answer, does it actually converge — or does it degrade, oscillate, or loop?*

### Key Finding
Base models (GPT-2) and even large instruction-tuned models (GPT-OSS 20B) **fail to naturally converge**. They either regurgitate formatting, hallucinate, or oscillate between different structures. Our **Adaptive Detection + Intervention framework** solves this by statistically monitoring the recursion and dynamically altering prompts in real-time.

---

## 🏗️ Architecture

```
User Prompt → LLM Generation → Advanced Detector → Intervention Engine → Final Output
                    ↑                                      |
                    └──────────────────────────────────────┘
                         (Recursive refinement loop)
```

**Detection Metrics:** Semantic Similarity, Keyword Stability (Jaccard), Oscillation Score, Variance Stationarity  
**Intervention Strategies:** Adaptive Depth Control, Dynamic Prompt Modification, Diversity Injection, Error Correction

---

## 📂 Project Structure

| File | Description |
|:-----|:------------|
| `rlm.py` | Phase 1 — Basic RLM with GPT-2 and cosine similarity convergence |
| `rlm_convergence.py` | Phase 1.5 — Enhanced reflective prompt loop |
| `convergence_log.py` | Data structures for experiment logging |
| `rlm_ollama.py` | Phase 2 — Ollama API integration for GPT-OSS 20B testing |
| `rlm_advanced_detection.py` | Phase 2.5 — Advanced statistical convergence detection |
| `convergence_detector.py` | Standalone advanced detector with smart stopping |
| `phase3_interventions.py` | Phase 3 — Intervention-aware RLM with baseline comparison |
| `phase4_validation.py` | Phase 4 — Grid-search auto-tuner, robustness stress-tests |
| `phase5_advanced_research.py` | Phase 5 — Meta-learning, ensemble voting, scalability analysis |
| `phase4_results.json` | Optimized deployment configuration and benchmark data |
| `phase5_research_results.json` | Meta-learning outcomes and scalability projections |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers sentence-transformers scikit-learn numpy scipy
```

### Run the Full Pipeline
```bash
# Phase 1: Baseline convergence testing
python rlm.py

# Phase 3: Baseline vs Intervention comparison
python phase3_interventions.py

# Phase 4: Auto-tuning and validation
python phase4_validation.py

# Phase 5: Advanced research framework
python phase5_advanced_research.py
```

### Test with Local Ollama (GPT-OSS 20B)
```bash
# Ensure Ollama is running with the model loaded
ollama run gpt-oss:20b

# Run the Ollama-based RLM
python rlm_ollama.py

# Run advanced detection against the 20B model
python rlm_advanced_detection.py
```

---

## 📊 Key Results

### Optimal Configuration (Phase 4 Auto-Tuner)
| Parameter | Value |
|:----------|:------|
| Similarity Threshold | `0.85` |
| Max Iterations | `3` |
| Temperature | `0.7` |
| Quality Score | `0.885` |

### Robustness
- **100% success rate** on adversarial inputs (empty, vague, nonsensical prompts)

### Scalability (Phase 5 Projections)
| Scale | Users | Throughput | Latency | CPU Cores | RAM |
|:------|:------|:-----------|:--------|:----------|:----|
| 1x | 10 | 5 req/s | 200ms | 1 | 4 GB |
| 10x | 100 | 50 req/s | 661ms | 16 | 40 GB |

### Intervention Impact
- Adaptive Depth Control saves **~20% compute** while maintaining output quality
- Improvement over baseline: **+17.3%** convergence score

---

## 🔬 Research Phases

| Phase | Focus | Status |
|:------|:------|:-------|
| **Phase 1** | Baseline RLM + failure mode analysis | ✅ Complete |
| **Phase 1.5** | Enhanced reflective prompts | ✅ Complete |
| **Phase 2** | GPT-OSS 20B via Ollama API | ✅ Complete |
| **Phase 2.5** | Advanced statistical detection | ✅ Complete |
| **Phase 3** | Dynamic intervention strategies | ✅ Complete |
| **Phase 4** | Validation & auto-optimization | ✅ Complete |
| **Phase 5** | Meta-learning & publication readiness | ✅ Complete |

---

## 🧠 Identified Failure Modes

1. **Repetitive Degeneration** — Model regurgitates prompt format instead of refining content
2. **Semantic Collapse** — Output devolves into meaningless tokens (hyphens, numbers)
3. **Factual Drift** — Model gradually forgets original topic across iterations
4. **Oscillation** — Large models alternate between different valid structures instead of converging

---

## 📚 Publication

Paper outline generated: ***"Adaptive Convergence Detection in Recursive Language Models"***

Sections: Introduction → Methodology → Experiments → Discussion → Future Work

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **Models Used**: GPT-2 (HuggingFace), GPT-OSS 20B (Ollama)
- **Similarity**: all-MiniLM-L6-v2 (Sentence-Transformers)
- **Frameworks**: PyTorch, Transformers, scikit-learn, SciPy
