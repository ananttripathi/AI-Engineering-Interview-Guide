<div align="center">
  <img src="cover.png" alt="AI Engineering Interview Guide Cover" />

  # The Ultimate AI Engineering Interview Guide

  [![GitHub](https://img.shields.io/badge/GitHub-ananttripathi-181717?style=flat&logo=github)](https://github.com/ananttripathi)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Anant_Tripathi-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/ananttripathiak/)
  ![Questions](https://img.shields.io/badge/Questions-200%2B-brightgreen)
  ![Sections](https://img.shields.io/badge/Sections-18-blue)
  ![Updated](https://img.shields.io/badge/Updated-2025-orange)

  <p>The definitive interview preparation guide for modern AI Engineering roles — covering Foundation Models, RAG, Agentic AI, Fine-Tuning, LLMOps, Infrastructure, Multimodal Systems, and AI Safety. Built by studying hundreds of real interview reports and top open-source resources.</p>
</div>

> **Roles this guide is curated for:** AI Engineer • GenAI Engineer • LLM Engineer • Agentic AI Engineer • AI Solutions Architect • AI Platform Engineer • MLOps/LLMOps Engineer

---

## 🗺️ How to Use This Guide

**Difficulty levels used throughout:**
- 🟢 **Junior** — Expected for 0–2 year roles; conceptual understanding
- 🟡 **Mid** — Applied knowledge; can implement and debug
- 🔴 **Senior** — Architectural trade-offs; design under constraints; first-principles reasoning

**Recommended study paths by role:**

| Role | Priority Sections |
|---|---|
| **AI/LLM Engineer** | 1 → 2 → 3 → 4 → 5 → 14 |
| **GenAI Platform Engineer** | 1 → 8 → 9 → 11 → 6 → 14 |
| **MLOps/LLMOps Engineer** | 1 → 9 → 10 → 11 → 7 → 14 |
| **AI Research Engineer** | 2 → 7 → 16 → 12 → 10 → 17 |
| **AI Solutions Architect** | 1 → 8 → 4 → 5 → 9 → 15 |

---

## 📌 Table of Contents

* [1. Core Must-Know Concepts](#1-core-must-know-concepts)
* [2. LLM Fundamentals & Architecture](#2-llm-fundamentals--architecture)
* [3. Prompt Engineering & Optimization](#3-prompt-engineering--optimization)
* [4. Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
* [5. AI Agents & Agentic Systems](#5-ai-agents--agentic-systems)
* [6. Vector Databases & Embeddings](#6-vector-databases--embeddings)
* [7. Fine-Tuning & Model Alignment](#7-fine-tuning--model-alignment)
* [8. AI System Design](#8-ai-system-design)
* [9. Production AI & LLMOps](#9-production-ai--llmops)
* [10. Evaluation & Benchmarking](#10-evaluation--benchmarking)
* [11. AI Infrastructure & Scalability](#11-ai-infrastructure--scalability)
* [12. Multi-Modal AI](#12-multi-modal-ai)
* [13. AI Safety, Ethics & Security](#13-ai-safety-ethics--security)
* [14. Coding & Practical Implementation](#14-coding--practical-implementation)
* [15. Behavioral & Scenario-Based Questions](#15-behavioral--scenario-based-questions)
* [16. Scaling Laws & Training Dynamics](#16-scaling-laws--training-dynamics)
* [17. Quantization & Model Compression](#17-quantization--model-compression)
* [18. Key Papers, Models & Tools Reference](#18-key-papers-models--tools-reference)

---

## 1. Core Must-Know Concepts
*Master these fundamentals before anything else — expect every interview to touch at least three of these.*

| Concept | One-line definition |
|---|---|
| **Transformer** | Encoder-decoder (or decoder-only) architecture using self-attention to model token relationships in parallel |
| **RAG** | Embed → Retrieve → Augment prompt → Generate — grounds LLM output in up-to-date external knowledge |
| **Function Calling / Tool Use** | Structured mechanism by which an LLM emits a JSON payload to trigger deterministic external code |
| **MCP** | Model Context Protocol — standardizes how foundation models communicate with external tools and data sources |
| **A2A** | Agent-to-Agent protocol — standardizes inter-agent communication across frameworks |
| **LoRA / PEFT** | Freezes base model weights; trains low-rank factorized matrices ΔW = BA to adapt with <1% of parameters |
| **Quantization** | Reduces weight precision (FP16 → INT8 → INT4) to cut memory; GGUF/GPTQ/AWQ are the dominant formats |
| **KV Cache** | Stores computed Key/Value tensors from prior tokens to avoid recomputation during autoregressive generation |
| **RLHF / DPO** | Aligns model behavior to human preferences via reward model + RL (RLHF) or direct contrastive loss (DPO) |
| **Agentic Loop** | Perceive → Plan → Act → Observe cycle; the core execution model for autonomous AI agents |
| **MoE** | Mixture-of-Experts — routes each token to a sparse subset of "expert" FFN layers, scaling params without scaling FLOPs |
| **Hallucination** | Model generates plausible-sounding but factually incorrect or unsupported output |

---

## 2. LLM Fundamentals & Architecture
*Deep dives into how Transformers actually work under the hood.*

- 🟢 What are foundation models, and how do they differ from task-specific ML models?
- 🟢 Break down the Transformer architecture: Encoders, Decoders, and Attention Mechanisms.
- 🟢 Explain tokenization (BPE, WordPiece, SentencePiece). How can tokenization negatively impact code generation or mathematical reasoning?
- 🟢 What are positional embeddings? Compare absolute, relative, and Rotary Position Embeddings (RoPE).
- 🟡 Explain the Query (Q), Key (K), and Value (V) matrices in self-attention. Walk through the matrix multiplication step by step.
- 🟡 Why is the attention dot product scaled by $\sqrt{d_k}$? What happens to gradients without this scaling?
- 🟡 Explain causal masking and why it's critical for auto-regressive generation.
- 🟡 Detail the purpose of Multi-Head Attention (MHA) vs. Grouped-Query Attention (GQA) vs. Multi-Query Attention (MQA). What are the memory trade-offs of each?
- 🟡 What is the KV Cache? Explain PagedAttention and how it mitigates memory bottlenecks during sequence generation.
- 🟡 What are logits? How do `temperature`, `top-k`, and `top-p` (nucleus) sampling manipulate the probability distribution?
- 🔴 What is a Mixture of Experts (MoE) architecture? Compare dense vs. sparse models. How does DeepSeek-V3's MoE routing strategy differ from classic MoE?
- 🔴 Explain Flash Attention and why it dramatically reduces memory usage during training even though the mathematical result is identical to standard attention.
- 🔴 What is Sliding Window Attention (used in Mistral)? When does full attention remain necessary despite its quadratic cost?
- 🔴 Explain the mechanics of Retrieval-Augmented Generation at the architecture level — how does a model like Atlas or RAG-Token differ from naive prompt stuffing?
- 🔴 What is test-time compute scaling? Explain how Process Reward Models (PRMs) and Monte Carlo Tree Search (MCTS) enable models like o1/o3 to "think longer" for harder problems.
- **Scenario 🔴:** Your LLM is generating text that repeats phrases in long outputs. How do you penalize repetition mathematically using presence penalty and frequency penalty?

---

## 3. Prompt Engineering & Optimization
*Controlling model behaviour through sophisticated input shaping.*

- 🟢 Contrast zero-shot, one-shot, and few-shot prompting. When does few-shot fail?
- 🟢 What is Chain-of-Thought (CoT) prompting? How does it differ from Tree-of-Thoughts (ToT) or multi-path reasoning?
- 🟡 What is ReAct (Reasoning + Acting) prompting? How does it differ from a standard tool-calling loop?
- 🟡 How do you engineer prompts for reliable structured JSON generation? (e.g., using schemas, XML tags, grammar-constrained decoding).
- 🟡 What is the "Lost in the Middle" phenomenon? How do you mitigate it when important context is buried in a long prompt?
- 🟡 Explain Prompt Caching (Anthropic/OpenAI). How does it reduce costs on repetitive long-context tasks?
- 🟡 What is **prompt brittleness**? Why does a small phrasing change sometimes cause a large quality drop, and how do you test for it?
- 🔴 What is DSPy? How does it replace manual prompt engineering with a gradient-like optimization loop over prompt programs?
- 🔴 Explain self-consistency prompting. When does sampling multiple CoT paths and voting outperform a single greedy decode?
- **Scenario 🟡:** You are hitting context window limits aggressively. How do you summarize or partition context dynamically while preserving semantic coherence?
- **Scenario 🔴:** Your few-shot examples cause the LLM to overfit on format, producing rigid answers. How do you increase variance without degrading quality?

---

## 4. Retrieval-Augmented Generation (RAG)
*Mastering dynamic knowledge injection.*

- 🟢 Explain the end-to-end architecture of a production-grade Chunk-and-Embed RAG pipeline.
- 🟡 Detail text chunking strategies: fixed-size, recursive, semantic, and parent-child chunking. When does each win?
- 🟡 What is Hybrid Search? Explain how BM25 (sparse) and Vector Search (dense) are combined via Reciprocal Rank Fusion (RRF).
- 🟡 Why is Re-ranking (Cross-Encoders or Cohere Rerank) necessary after initial retrieval? What is the precision/latency trade-off?
- 🟡 What is Query Transformation? Explain Query Expansion, Multi-Query formulation, and Step-Back Prompting.
- 🔴 What is Self-RAG/Corrective RAG (CRAG)? How does it grade the quality of retrieved context and decide when to re-retrieve vs. fall back to parametric knowledge?
- 🔴 Explain Graph RAG and Knowledge Graphs. When should you use Graph RAG over dense vector RAG for multi-hop reasoning?
- 🔴 What is contextual retrieval (Anthropic's approach)? How does prepending chunk-level context summaries before embedding improve retrieval precision?
- 🔴 Describe the RAGas evaluation framework. What do the following metrics measure and how are they computed?

  | RAGas Metric | What it Measures |
  |---|---|
  | **Context Precision@k** | Fraction of retrieved chunks that are actually relevant |
  | **Context Recall** | Fraction of ground-truth evidence chunks that were retrieved |
  | **Faithfulness** | Whether the answer is supported by (not contradicted by) the retrieved context |
  | **Answer Relevance** | Semantic alignment between the question and the generated answer |
  | **Answer Semantic Similarity** | Similarity of generated answer to the reference answer |

- **Scenario 🟡:** Your enterprise RAG system returns contradictory answers from different HR policy documents in the vector DB. How do you architect a conflict-resolution layer?
- **Scenario 🔴:** RAG retrieval is extremely slow for a database of 1M+ embeddings. Walk through the full indexing and query optimization stack.

---

## 5. AI Agents & Agentic Systems
*Building autonomous, reasoning software layers.*

- 🟢 What defines an AI Agent versus an LLM prompt chain? What properties must hold for a system to be called "agentic"?
- 🟡 Explain the Plan-and-Execute agent framework. How does it differ from ReAct-style single-step agents?
- 🟡 Deep dive into Function Calling: How does the model decide when to emit a tool call vs. text? What does the token-level probability distribution look like at the branch point?
- 🟡 Describe multi-agent setups (LangGraph, AutoGen, CrewAI, Google ADK, Claude Agent SDK). When is a supervisor/worker hierarchy better than a flat peer architecture?
- 🟡 What are the types of agent memory?

  | Memory Type | Scope | Example |
  |---|---|---|
  | **Working / Short-term** | Within one agent session | Current conversation context window |
  | **Episodic** | Stored past experiences | Summarized past conversations in a DB |
  | **Semantic** | Factual knowledge store | External vector DB for RAG |
  | **Procedural** | Learned action policies | Fine-tuned tool-calling behavior |

- 🔴 What is the Model Context Protocol (MCP)? How does it standardize context injection, tool registration, and sampling requests between a host model and external data sources?
- 🔴 What is the Agent-to-Agent (A2A) protocol? How does it enable interoperability between agents built on different frameworks?
- 🔴 How do you implement reliable long-horizon planning? What failure modes emerge when an agent needs more than 10 sequential tool calls?
- **Scenario 🟡:** Your agent enters an infinite loop, calling a search tool with the same query. How do you implement circuit breakers and loop-detection heuristics?
- **Scenario 🔴:** An autonomous data-cleaning agent executed a `DROP TABLE` command in production. How do you architect human-in-the-loop (HitL) execution with sandboxed dry-run environments?

---

## 6. Vector Databases & Embeddings
*Handling high-dimensional data representation.*

- 🟢 What are semantic embeddings? Differentiate sparse embeddings (SPLADE, BM25) from dense embeddings.
- 🟡 Which distance metrics apply to vector search? When do you use Cosine Similarity vs. Euclidean (L2) vs. Dot Product?
- 🟡 Explain the HNSW (Hierarchical Navigable Small World) algorithm. How does the layered graph enable sub-linear approximate nearest neighbour search?
- 🟡 Describe Product Quantization (PQ) and Scalar Quantization (SQ). How does each compress vector footprint, and what accuracy do you trade away?
- 🟡 Explain Metadata Filtering in Vector DBs: what is pre-filtering vs. post-filtering and which is faster in practice?
- 🔴 Compare leading vector databases on key axes:

  | DB | Hosting | Filtering | Hybrid Search | Best For |
  |---|---|---|---|---|
  | **Pinecone** | Managed cloud | Strong | Yes (sparse+dense) | Production scale, minimal ops |
  | **Weaviate** | Self-hosted / cloud | Strong | Native | Hybrid + GraphQL queries |
  | **Qdrant** | Self-hosted / cloud | Strong | Yes | Rust-native, high throughput |
  | **Chroma** | Embedded / cloud | Basic | Limited | Local dev, prototyping |
  | **pgvector** | Postgres extension | Full SQL | Limited | Existing Postgres stacks |
  | **Milvus** | Self-hosted / cloud | Strong | Yes | Billion-scale deployments |

- **Scenario 🔴:** You swapped embedding models from `text-embedding-ada-002` to `text-embedding-3-large`. How do you migrate 10M vectors with zero downtime? Walk through the dual-index blue-green strategy.

---

## 7. Fine-Tuning & Model Alignment
*Adapting weights to domain-specific knowledge.*

- 🟢 Provide a decision matrix: when to use Prompt Engineering vs. RAG vs. Fine-Tuning vs. Pre-training from scratch?
- 🟡 Explain Supervised Fine-Tuning (SFT) and what makes a high-quality instruction dataset (diversity, format, seed data curation).
- 🟡 What is LoRA (Low-Rank Adaptation)? Explain the math: the base weight $W_0$ is frozen; the update is expressed as $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d,k)$.
- 🟡 What is catastrophic forgetting? Explain how replay buffers, elastic weight consolidation (EWC), and LoRA adapters each mitigate it.
- 🟡 Explain RLHF end-to-end: SFT phase → Reward Model training → PPO policy optimization loop. What does the reward signal look like?
- 🟡 What is DPO (Direct Preference Optimization)? Why has it displaced classical RLHF for most teams?
- 🔴 Compare modern alignment algorithms:

  | Algorithm | Key Idea | Relative Complexity | When to Use |
  |---|---|---|---|
  | **RLHF + PPO** | Reward model + RL loop | High | When you have a reliable reward model |
  | **DPO** | Direct contrastive loss on preference pairs | Low | Most fine-tuning alignment tasks |
  | **GRPO** | Group-based relative policy optimization | Medium | Reasoning tasks (math, code) |
  | **ORPO** | Odds-ratio penalty directly in SFT loss | Low | Single-stage SFT + alignment |
  | **SimPO** | Reference-free DPO variant | Low | When no reference model is available |

- 🔴 What are synthetic data generation pipelines? Explain how frontier-model distillation, Auto-Evol, and Branch-Solve-Merge are used to bootstrap post-training datasets.
- 🔴 What is model merging? Explain the SLERP, DARE, and TIES algorithms for merging fine-tuned LoRA adapters or full models without further training (via MergeKit).
- 🔴 What is "abliteration" (refusal suppression via weight surgery)? How does it differ from fine-tuning on refusal-override examples?
- **Scenario 🔴:** Your fine-tuned LLM excels at your domain task but has lost general reasoning capabilities. How do you diagnose and recover general capability without restarting from scratch?

---

## 8. AI System Design
*Architecting real-world generative applications.*

> For each scenario below, structure your answer around: **Components → Data flow → Latency budget → Failure modes → Cost model**

- 🟡 **AI Coding Assistant (Copilot clone):** Address IDE context fetching, ghost-text streaming, fill-in-the-middle (FIM) prompting format, and speculative decoding to hit <100ms P50 latency.
- 🟡 **Enterprise Knowledge Assistant:** Design for millions of documents with row-level access control (Slack, G-Drive, Jira). How do you enforce permissions at retrieval time, not just at display time?
- 🔴 **Real-time Fraud Detection:** Design a pipeline where transaction logs are scored by a fast classification LLM. How do you handle the latency requirement (<50ms) vs. model quality trade-off?
- 🔴 **Async Batch Document Parser:** Design a system parsing 100-page financial PDFs into structured JSON overnight. Address chunking, OCR, table extraction, and idempotent job orchestration.
- 🔴 **Multi-tier LLM Router:** Design a router that classifies query complexity and routes to the cheapest model capable of handling it (e.g., Haiku → Sonnet → Opus). How do you measure routing accuracy and set confidence thresholds?
- 🔴 **Conversational AI with Memory:** Design persistent user memory that survives across sessions. Compare vector-based memory (similarity search over past turns) vs. structured memory (entity extraction → knowledge graph).
- 🔴 **GenAI API Gateway:** Explain rate-limiting, cost-monitoring (per-tenant budget caps), prompt/response logging, PII scrubbing at ingress, and multi-model failover patterns.

---

## 9. Production AI & LLMOps
*Deploying and maintaining AI systems at scale.*

- 🟢 What is the difference between traditional MLOps and LLMOps? What new concerns does a frozen foundation model introduce?
- 🟡 Explain the 7-phase LLMOps lifecycle:

  | Phase | Key Activities |
  |---|---|
  | **1. Pre-Development** | Literature survey, model selection, ethical review, build-vs-buy decision |
  | **2. Data Preparation** | Curation, PII scrubbing, formatting, synthetic data generation, dataset versioning |
  | **3. Model Development** | SFT, alignment (DPO/RLHF), evaluation against benchmarks |
  | **4. Optimization** | Quantization, distillation, prompt optimization (DSPy), latency profiling |
  | **5. Deployment & Integration** | Containerization, serving (vLLM/TGI), API gateway, feature flags for prompt versions |
  | **6. Post-Deployment Monitoring** | TTFT, ITL, cost/request, hallucination rate, user feedback loops |
  | **7. Continuous Improvement** | RLHF from production logs, model refresh cadence, A/B prompt experiments |

- 🟡 How do you manage API key rotation, Secrets management, and enterprise gateways for OpenAI/Anthropic APIs?
- 🟡 Explain the CI/CD lifecycle for a prompt template change. What tests run in the pipeline before it reaches production?
- 🟡 What observability metrics are non-negotiable for a production LLM pipeline?
  - **TTFT** (Time To First Token) — perceived responsiveness
  - **ITL** (Inter-Token Latency) — streaming smoothness
  - **Cost per request** — per model, per tenant
  - **Hallucination rate** — sampled LLM-as-judge evaluations
  - **Context utilization** — are you over/under-filling the context window?
- 🔴 Describe distributed tracing for multi-stage agentic chains (LangSmith, Arize Phoenix, Langfuse). What does a span tree look like for a 5-step RAG + agent chain?
- 🔴 Distinguish model drift from prompt drift from data drift in the context of frozen foundational models. How do you detect each without ground truth labels?
- **Scenario 🔴:** Traffic spikes 10× at peak. Walk through your strategy: continuous batching, request queuing, autoscaling GPU replicas, and graceful degradation to a smaller model.

---

## 10. Evaluation & Benchmarking
*Proving your AI actually works.*

- 🟢 Contrast lexical metrics (BLEU, ROUGE-L) with semantic metrics (BERTScore, MoverScore). Why do lexical metrics fail for open-ended generation?
- 🟡 Explain the `LLM-as-a-judge` concept. What biases does it introduce?
  - **Position bias** — prefers answers in the first position
  - **Verbosity bias** — prefers longer answers regardless of correctness
  - **Self-enhancement bias** — the judge model favors outputs from models in its own family
- 🟡 What are faithfulness and answer relevance in RAG evaluation? Describe the RAGAS framework and how each metric is computed.
- 🟡 What is G-Eval? How does it use chain-of-thought scoring rubrics to produce more calibrated judgments than single-score prompts?
- 🟡 Discuss standard benchmarks and what each actually measures:

  | Benchmark | Measures |
  |---|---|
  | **MMLU** | Multitask language understanding across 57 subjects |
  | **HumanEval / MBPP** | Code generation correctness (pass@k) |
  | **GSM8K / MATH** | Multi-step arithmetic and formal math reasoning |
  | **HellaSwag** | Commonsense NLI / story completion |
  | **TruthfulQA** | Calibration against common misconceptions |
  | **MT-Bench / LMSYS Chatbot Arena** | Instruction-following quality via human ELO ratings |
  | **RULER** | Long-context recall at various context lengths |
  | **Berkeley Function Calling** | Tool-use and JSON schema adherence accuracy |

- 🔴 What is SelfCheckGPT? How does sampling the same prompt multiple times and measuring consistency across samples detect hallucinations without a reference answer?
- 🔴 Design an offline evaluation pipeline for a production RAG system when you have no ground-truth question-answer pairs. Walk through synthetic QA generation, RAGAS scoring, and human spot-check sampling.
- **Scenario 🔴:** Two evaluators (one human, one LLM judge) sharply disagree on prompt quality. How do you reconcile and establish a reliable ground truth?

---

## 11. AI Infrastructure & Scalability
*Hardware, inferencing, and distributed deployments.*

- 🟡 Discuss GPU architecture for training vs. inference. What dictates VRAM limits for a given model size? (Rough rule: 2 bytes/param for FP16 inference, 16–20 bytes/param for full AdamW training.)
- 🟡 Explain Tensor Parallelism (TP) vs. Pipeline Parallelism (PP) vs. Data Parallelism (DP) when deploying a 70B model across multiple GPUs.
- 🟡 Compare inferencing engines on key dimensions:

  | Engine | Best For | Strengths | Weaknesses |
  |---|---|---|---|
  | **vLLM** | High-throughput serving | PagedAttention, continuous batching, OpenAI-compatible | Higher memory overhead |
  | **TGI (HuggingFace)** | Flexible open-model serving | Wide model support, streaming | Less optimized than vLLM |
  | **TensorRT-LLM** | NVIDIA GPU max throughput | Kernel fusion, INT8/FP8, fastest raw speed | NVIDIA-only, complex setup |
  | **llama.cpp / GGUF** | CPU/edge inference | Runs without GPU, GGUF format | Slower than GPU engines |
  | **MLC LLM** | Browser / mobile / iOS / Android | WebGPU, cross-platform | Early ecosystem |

- 🟡 What is Speculative Decoding? How does a small "draft" model generate token candidates that the main model verifies in parallel to increase tokens/second?
- 🔴 Describe FSDP (Fully Sharded Data Parallel) and DeepSpeed ZeRO stages 1, 2, and 3. What is sharded at each stage?
- 🔴 What is continuous batching (iteration-level batching)? How does it differ from static batching and why does it improve GPU utilization dramatically for variable-length sequences?
- 🔴 Explain edge deployment considerations: quantization to INT4 (GGUF/MLX), MLC LLM for WebGPU/iOS/Android, and the latency vs. quality trade-off when running a 7B model on-device.
- **Scenario 🔴:** You are out of GPU VRAM during inference. Walk through a decision tree: quantization → weight offloading to CPU RAM → model parallelism across GPUs → switching to a smaller model.

---

## 12. Multi-Modal AI
*Moving beyond text.*

- 🟡 Explain how Vision-Language Models (VLMs) process images: ViT patch embeddings, projection layers, and how visual tokens are merged into the language model's token stream.
- 🟡 How does CLIP align text and image embedding spaces via contrastive pre-training? What does the loss function look like?
- 🟡 Detail the mechanics of Diffusion models: the forward noising process $q(x_t | x_{t-1})$ and the learned reverse denoising process $p_\theta(x_{t-1} | x_t)$. What does the U-Net predict?
- 🟡 Discuss audio/speech integration: Whisper architecture for STT, autoregressive vs. flow-matching TTS, and voice activity detection.
- 🔴 What is LLaVA and how does it compare to GPT-4V architecturally? What are the limitations of naive image-text interleaving vs. cross-attention fusion (Flamingo-style)?
- 🔴 Explain the role of RLHF/DPO in aligning multimodal models. What new failure modes (e.g., hallucinating image content) emerge vs. text-only alignment?
- **Scenario 🔴:** Your multimodal RAG pipeline needs to search across video assets. How do you simultaneously extract frame-level CLIP embeddings, Whisper transcripts, and LLM-generated structured metadata — and merge these three retrieval signals at query time?

---

## 13. AI Safety, Ethics & Security
*Safeguarding open-ended models.*

- 🟢 Classify types of adversarial attacks on LLMs:

  | Attack | Description |
  |---|---|
  | **Direct Prompt Injection** | Attacker input overrides system prompt instructions |
  | **Indirect Prompt Injection** | Malicious instructions embedded in retrieved documents (RAG poisoning) |
  | **Jailbreaking** | Crafted prompts that bypass safety fine-tuning |
  | **Data Poisoning** | Corrupting training/fine-tuning data to induce backdoor behaviors |
  | **Model Inversion** | Extracting training data from model outputs |
  | **Membership Inference** | Determining whether a specific sample was in the training set |
  | **Backdoor Attacks** | Inserting trigger patterns that activate malicious behavior at inference |

- 🟡 Explain defenses against prompt injection: sentinel tokens, instruction hierarchy (system > user > tool), randomized delimiters, and input sanitization.
- 🟡 What is the TrustLLM alignment taxonomy? Name the 9 dimensions: Truthfulness, Safety, Fairness, Robustness, Privacy, Machine Ethics, Transparency, Accountability, Regulations.
- 🟡 Discuss PII scrubbing strategies for LLM ingress: regex NER, transformer-based NER (Presidio), and the challenge of contextual PII that regex can't catch.
- 🔴 Explain the EU AI Act's risk tiers (Unacceptable → High → Limited → Minimal). What obligations apply to a company deploying a High-Risk generative AI solution?
- 🔴 What is differential privacy in the context of LLM fine-tuning (DP-SGD)? What privacy-utility trade-off does the $\epsilon$ parameter represent?
- 🔴 What is red-teaming for LLMs? Describe how automated red-teaming tools (Garak, PyRIT) work and what attack categories they probe.
- **Scenario 🔴:** A user discovers they can make your enterprise support chatbot guarantee a $10,000 refund via a crafted prompt. Walk through the full mitigation stack: detection, guardrails, output validation, and legal/audit logging.

---

## 14. Coding & Practical Implementation
*Proving you can build it.*

> Expect to write these in 20–40 minutes in a live coding interview. Know the pattern cold.

**Foundational Patterns**

- 🟡 Build a dynamic few-shot prompt selector: embed user input, compute cosine similarity against an example pool, return top-k as few-shot demonstrations.
- 🟡 Write a recursive text chunker from scratch (no LangChain) that splits on `\n\n` → `\n` → `. ` → ` ` in order, ensuring chunks respect a `max_tokens` limit without cutting mid-sentence.
- 🟡 Implement an LLM call wrapper that retries on JSON parse failure with exponential backoff, and appends the parse error to the next prompt to self-correct.
- 🟡 Build a generic tool-call router that maps a `function_call` JSON payload `{"name": "...", "arguments": {...}}` to actual Python callables registered in a dict.
- 🟡 Write a local semantic cache: on each new query, check cosine similarity of its embedding against cached query embeddings; if similarity > threshold, return cached response.

**Advanced Patterns**

- 🔴 Implement a streaming LLM response handler using `httpx` async streaming that progressively yields tokens and handles mid-stream `[DONE]` termination.
- 🔴 Build a simple agentic loop: the agent calls tools in a while loop, appends tool results to the message history, and terminates when the model emits a `finish_reason: stop` with no tool calls.
- 🔴 Implement a parallel LLM orchestrator using `asyncio.gather` that fires 10 independent LLM calls simultaneously and aggregates results with a timeout and partial-failure fallback.
- 🔴 Build a minimal RAG pipeline: chunk a text file, embed chunks with the OpenAI Embeddings API, store in a numpy matrix, and answer queries using cosine-similarity retrieval + GPT-4o generation.
- 🔴 Implement a LoRA weight merge: given a base model weight matrix and two LoRA adapters (A, B matrices), compute the merged weight $W_{merged} = W_0 + \alpha \cdot BA$ and verify the output matches running the adapter at inference time.

---

## 15. Behavioral & Scenario-Based Questions
*For senior and architectural roles. Use STAR format: Situation → Task → Action → Result.*

- 🟡 Tell me about a time an AI project you launched hallucinated in front of users. How did you diagnose the root cause and what guardrails did you add?
- 🟡 How do you convince non-technical leadership that RAG is a better investment than full fine-tuning for a knowledge-intensive use case?
- 🟡 How do you stay current in a field where state-of-the-art changes every two weeks?
- 🔴 Describe an architecture decision you made early in an ML system that you later had to drastically refactor. What signals did you miss?
- 🔴 With only $500/month in API budget, how do you architect a platform expected to serve 10,000 distinct daily users? Walk through your model routing, caching, and batching strategy.
- 🔴 How do you balance the trade-off between maximizing prompt accuracy (more context, better answers) vs. real-time responsiveness (token budgets, streaming)?
- 🔴 Your team wants to fine-tune a model but has no labeled data. How do you bootstrap a high-quality training dataset using synthetic data generation?
- 🔴 Describe a situation where you had to make a build-vs-buy decision for a core AI component. What was your framework?

---

## 16. Scaling Laws & Training Dynamics
*The physics of making large models work — critical for research-adjacent roles.*

- 🟡 What are scaling laws (Kaplan et al., 2020)? State the empirical relationships between model loss $L$, number of parameters $N$, training tokens $D$, and compute $C$.
- 🔴 What are Chinchilla scaling laws (Hoffmann et al., 2022)? How do they revise Kaplan's findings — specifically the compute-optimal ratio of training tokens to parameters?
  > Key result: for a compute budget $C$, the optimal model size $N^*$ and tokens $D^*$ scale equally: $N^* \propto C^{0.5}$, $D^* \propto C^{0.5}$. This means GPT-3 (175B parameters, 300B tokens) was significantly undertrained relative to its compute budget.
- 🔴 What is the "emergent abilities" debate? What do Wei et al. (2022) claim and what does Schaeffer et al. (2023) argue in rebuttal?
- 🔴 Explain the neural scaling law for data: what is the effect of dataset quality vs. quantity? How does the FineWeb paper (HuggingFace, 2024) empirically demonstrate that careful data curation outperforms raw data scale?
- 🔴 What is learning rate warm-up and cosine decay scheduling? Why is the peak LR so critical to final model quality, and what happens when you resume training from a checkpoint with a decayed LR?
- 🔴 What is gradient clipping? Why is it essential when training on long sequences or with high learning rates? What value is typically used?
- 🔴 Explain the "grokking" phenomenon: why do some models suddenly generalize long after training loss plateaus? What does this imply about evaluation checkpointing strategy?
- **Scenario 🔴:** You have a $1M compute budget. Using Chinchilla-optimal ratios, determine the ideal model size and dataset size for training. How does this change if you prioritize inference cost over training cost?

---

## 17. Quantization & Model Compression
*Getting big models to run on smaller hardware.*

- 🟡 Explain the difference between post-training quantization (PTQ) and quantization-aware training (QAT). When is each appropriate?
- 🟡 Compare quantization formats:

  | Format | Precision | Method | Use Case | Tooling |
  |---|---|---|---|---|
  | **GGUF** | INT4/INT8 | Weight-only, CPU-friendly | Local CPU/GPU inference | llama.cpp, Ollama |
  | **GPTQ** | INT4 | RTN + second-order correction | GPU inference, single device | AutoGPTQ, HuggingFace |
  | **AWQ** | INT4 | Activation-aware weight quantization | GPU inference, better accuracy than GPTQ | AutoAWQ |
  | **EXL2** | Mixed 2–8 bit | Per-layer importance weighting | Max accuracy at target bitwidth | ExLlamaV2 |
  | **FP8** | 8-bit float | Hardware-native on H100/A100 | Training + inference on Hopper | TensorRT-LLM, vLLM |
  | **BitsAndBytes** | INT8/INT4 | Dynamic quantization | Quick prototyping | HuggingFace transformers |

- 🟡 What is weight-only quantization vs. activation quantization? Why is quantizing activations harder than weights?
- 🔴 Explain SmoothQuant: how does it mathematically migrate quantization difficulty from activations (hard) to weights (easy) via a per-channel scaling factor?
- 🔴 What is knowledge distillation? Explain the difference between output distillation (matching logit distributions), feature distillation (matching intermediate activations), and the role of temperature in softening the teacher's distribution.
- 🔴 What is model pruning? Compare unstructured pruning (zeroing individual weights), structured pruning (removing entire attention heads or FFN rows), and the lottery ticket hypothesis.
- **Scenario 🔴:** You need to serve a 70B model on a single 80GB A100. Walk through your options: FP8 serving, GPTQ INT4, tensor parallelism across multiple GPUs, and speculative decoding with a 7B draft model. How do you benchmark the quality-latency-cost frontier?

---

## 18. Key Papers, Models & Tools Reference
*Signal to interviewers that you track the field — know the "what introduced what" timeline.*

### Milestone Papers

| Year | Paper | Key Contribution |
|---|---|---|
| 2017 | **Attention Is All You Need** (Vaswani et al.) | The Transformer architecture |
| 2018 | **BERT** (Devlin et al.) | Bidirectional encoder pre-training; GLUE SOTA |
| 2020 | **GPT-3** (Brown et al.) | Few-shot in-context learning at 175B scale |
| 2020 | **Scaling Laws** (Kaplan et al.) | Power-law relationships between compute, params, data, loss |
| 2022 | **InstructGPT** (Ouyang et al.) | RLHF for instruction following; the template for ChatGPT |
| 2022 | **Chain-of-Thought** (Wei et al.) | Reasoning via step-by-step prompting |
| 2022 | **Chinchilla** (Hoffmann et al.) | Compute-optimal training: equal scaling of N and D |
| 2023 | **LLaMA** (Touvron et al.) | Open-weight competitive large language model |
| 2023 | **DPO** (Rafailov et al.) | Preference alignment without explicit reward model |
| 2023 | **Mamba** (Gu & Dao) | State Space Model as Transformer alternative |
| 2023 | **Mistral 7B** (Mistral AI) | Sliding window attention + GQA; beats Llama2-13B |
| 2024 | **DeepSeek-V2/V3** (DeepSeek) | Multi-head Latent Attention (MLA) + fine-grained MoE |
| 2024 | **Llama 3** (Meta) | 8B–405B family; GQA + long context (128k) |
| 2025 | **DeepSeek-R1** (DeepSeek) | RL-trained reasoning model rivaling o1; open weights |

### Open Model Families to Know

| Family | Org | Notable Variants | Strength |
|---|---|---|---|
| **Llama 3.x** | Meta | 8B, 70B, 405B | General purpose; most-used open base |
| **Qwen 2.5** | Alibaba | 0.5B–72B, Coder, Math, VL | Best small-model quality; strong multilingual |
| **DeepSeek** | DeepSeek | R1, V3, Coder, MoE | Reasoning, cost-efficiency, open weights |
| **Mistral** | Mistral AI | 7B, 8×7B MoE, Mixtral | Efficient architecture; sliding window attention |
| **Gemma 2** | Google | 2B, 9B, 27B | On-device and research-friendly |
| **Phi-3/4** | Microsoft | 3.8B, 7B, 14B | Strong reasoning at tiny scale |

### Tools & Frameworks Cheat Sheet

| Category | Tools |
|---|---|
| **Orchestration** | LangChain, LlamaIndex, DSPy, Haystack |
| **Agent Frameworks** | LangGraph, AutoGen, CrewAI, Google ADK, Claude Agent SDK |
| **Serving / Inference** | vLLM, TGI, TensorRT-LLM, llama.cpp, Ollama, MLC LLM |
| **Fine-tuning** | HuggingFace TRL, Unsloth, Axolotl, LLaMA-Factory |
| **Evaluation** | RAGAS, DeepEval, TruLens, Promptfoo, Giskard |
| **Observability** | LangSmith, Arize Phoenix, Langfuse, Helicone |
| **Vector DBs** | Pinecone, Weaviate, Qdrant, Chroma, Milvus, pgvector |
| **Quantization** | AutoGPTQ, AutoAWQ, BitsAndBytes, llama.cpp |
| **Safety / Red-teaming** | Garak, PyRIT, Guardrails AI, NeMo Guardrails |

---

<div align="center">
  <h3>Contributions, issues, and feature requests are welcome!</h3>
  <p>Constructed with ❤️ by <b>Anant Tripathi</b></p>
  <p><i>If you found this helpful, please give the repository a ⭐ — it helps others find it</i></p>
</div>
