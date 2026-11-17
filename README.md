# Tndrr AI/ML Internship Assignment — Tasks 1, 2 & 3

**Candidate:** Aadit Choudhary  
**GitHub Repository:** https://github.com/Aadit23042006/tndrr-ai-ml-assignment  
**Assignment:** AI/ML Internship Technical Assignment  
**Tasks Completed:** Task 1 — LoRA Fine-Tuning | Task 2 — Retrieval-Augmented Generation | Task 3 — Local LLM Inference and Quantization  
**Execution Environment:** Google Colab — NVIDIA Tesla T4 GPU  
**Final Submission:** Complete project source code, experimental results, trained adapter, supporting documents and four-page PDF report.

---

## 1. Project Overview

This repository contains the implementation and experimental evaluation of three AI/ML engineering tasks.

| Task | Project | Objective |
|---|---|---|
| Task 1 | LoRA Fine-Tuning | Fine-tune a language model to convert English calendar requests into structured JSON. |
| Task 2 | Retrieval-Augmented Generation | Build and evaluate a document retrieval pipeline using dense retrieval, BM25 and hybrid search. |
| Task 3 | Local LLM Inference | Benchmark FP16 and 4-bit quantized Ollama models and analyze memory requirements and concurrent serving. |

The project emphasizes practical implementation, reproducible experiments, quantitative evaluation, failure analysis and deployment considerations.

---

## 2. Repository Structure

```text
tndrr-assignment/
│
├── task1_lora/
│   ├── data/
│   ├── outputs/
│   │   └── lora_r16/
│   │       ├── adapter_model.safetensors
│   │       ├── adapter_config.json
│   │       └── train_summary.json
│   ├── results/
│   ├── make_dataset.py
│   ├── train_lora.py
│   ├── evaluate.py
│   ├── failures.py
│   ├── memory_calc.py
│   └── run_all.sh
│
├── task2_rag/
│   ├── docs/
│   │   ├── attention.pdf
│   │   ├── bert.pdf
│   │   ├── lora.pdf
│   │   └── qlora.pdf
│   ├── results/
│   │   ├── retrieval_metrics.json
│   │   └── retrieval_failures.md
│   ├── questions.json
│   ├── rag_lib.py
│   ├── evaluate_retrieval.py
│   ├── find_passage.py
│   └── ask.py
│
├── task3_local_inference/
│   ├── results/
│   │   ├── benchmark.json
│   │   ├── quality_comparison.md
│   │   ├── concurrency_20_users.json
│   │   └── serving_plan.json
│   ├── benchmark_ollama.py
│   ├── compare_quality.py
│   ├── kv_cache_calc.py
│   └── serving_plan.py
│
├── report/
│   ├── my_answers.md
│   ├── build_report.py
│   └── report.pdf
│
├── study_notes/
├── tests/
├── run_on_colab.ipynb
├── requirements.txt
├── run_tests.sh
└── README.md
```

The repository also contains supporting scripts, configuration files, evaluation outputs and documentation required to reproduce or inspect the experiments.

---

# 3. Task 1 — LoRA Fine-Tuning

## Objective

Fine-tune a pretrained language model to convert natural-language calendar requests into structured JSON containing fields such as:

- Event title
- Date
- Time
- Duration
- Location
- Attendees

## Implementation

**Base model:** `Qwen/Qwen2.5-0.5B`

**Training configuration:**

| Parameter | Value |
|---|---:|
| Training examples | 1,500 |
| Test examples | 100 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 0.0002 |
| Epochs | 3 |
| Batch size | 16 |
| Maximum sequence length | 256 |
| Precision | BF16 |
| Trainable parameters | 8,798,208 |
| Total model parameters | 502,830,976 |
| Trainable parameter percentage | 1.7497% |

LoRA adapters were applied to the attention and feed-forward projection layers.

The implementation uses a visible PyTorch training loop with PEFT-based LoRA configuration.

## Experimental Results

Evaluation on 100 test examples:

| Metric | Result |
|---|---:|
| Valid JSON | 100% |
| Schema compliance | 100% |
| Exact match | 93% |
| Title accuracy | 100% |
| Date accuracy | 100% |
| Time accuracy | 100% |
| Duration accuracy | 94% |
| Location accuracy | 99% |
| Attendees accuracy | 100% |

### In-Distribution and Shifted Evaluation

| Evaluation group | Examples | Exact match |
|---|---:|---:|
| In-distribution | 70 | 100% |
| Shifted distribution | 30 | 76.67% |
| Overall | 100 | 93% |

### Baseline Comparison

| Model configuration | Valid JSON | Schema compliance | Exact match |
|---|---:|---:|---:|
| Base short | 51% | 0% | 0% |
| Base instruct | 100% | 78% | 20% |
| Fine-tuned LoRA | 100% | 100% | 93% |

### Training Performance

| Measurement | Result |
|---|---:|
| Training duration | 612.669 seconds |
| Measured peak GPU memory | 8.425496 GB |
| Trainable parameters | 8.80 million |

### Failure Analysis

The evaluation identified errors involving:

1. Unseen duration values, including 75-minute and 105-minute requests.
2. Generalization to location names that were absent from the training examples.
3. Differences between training and shifted evaluation distributions.

The failure analysis compares incorrect predictions with the available training data to identify potential data coverage limitations.

### Memory Analysis

The project includes memory estimates for full fine-tuning, LoRA and QLoRA, along with activation-memory assumptions.

The calculated estimates and measured GPU memory are documented separately to distinguish theoretical estimates from actual measurements.

**Main artifact:** `task1_lora/outputs/lora_r16/adapter_model.safetensors`

---

# 4. Task 2 — Retrieval-Augmented Generation

## Objective

Implement a document retrieval pipeline that retrieves relevant passages from technical PDF documents and evaluates retrieval quality using multiple retrieval strategies.

## Document Corpus

Four technical documents were used:

| Document | Pages |
|---|---:|
| Attention | 15 |
| BERT | 16 |
| LoRA | 26 |
| QLoRA | 26 |
| **Total** | **83** |

The corpus contains approximately 43,290 extracted words.

## Implementation

The retrieval pipeline includes:

- PDF text extraction using PyPDF.
- Document preprocessing and text chunking.
- Dense retrieval using embeddings.
- FAISS/NumPy-based similarity search.
- BM25 keyword retrieval.
- Hybrid retrieval combining dense and lexical methods.
- Optional reranking.
- Gold-passage-based retrieval evaluation.

The evaluation uses character-span overlap with annotated gold passages to compare retrieval results across different chunk sizes.

## Evaluation Configuration

| Parameter | Value |
|---|---|
| Annotated questions | 15 |
| Document count | 4 |
| Total pages | 83 |
| Chunk sizes | 100 and 250 words |
| Retrieval methods | Dense, BM25 and Hybrid |
| Reranking | Evaluated |
| Main metrics | Recall@5 and MRR |

All 15 gold passage annotations passed exact passage validation against the extracted document corpus.

## Retrieval Results

| Chunk size | Retrieval method | Recall@5 | MRR |
|---|---|---:|---:|
| 100 | Dense | 0.4667 | 0.3556 |
| 100 | BM25 | 0.6667 | 0.3400 |
| 100 | Hybrid | 0.5333 | 0.3735 |
| 100 | Dense + Rerank | 0.4667 | 0.4150 |
| 100 | BM25 + Rerank | 0.5333 | 0.4511 |
| 100 | Hybrid + Rerank | 0.5333 | 0.4511 |
| 250 | Dense | 0.6667 | 0.3744 |
| 250 | BM25 | 0.6000 | 0.3590 |
| 250 | Hybrid | **0.7333** | **0.4500** |
| 250 | Dense + Rerank | 0.6000 | 0.3650 |
| 250 | BM25 + Rerank | 0.6667 | 0.3733 |
| 250 | Hybrid + Rerank | 0.6000 | 0.3595 |

## Key Findings

- Hybrid retrieval with 250-word chunks achieved the highest Recall@5 of 0.7333.
- BM25 with reranking at 100-word chunks achieved an MRR of 0.4511.
- Reranking improved ranking quality in some configurations but did not consistently improve recall.
- Chunk size and retrieval strategy significantly influenced retrieval performance.

### Failure Analysis

Retrieval failures were examined to identify questions for which the relevant gold passage was not retrieved effectively.

Examples included questions involving multi-head attention and the purpose of next-sentence prediction.

The analysis is documented in `task2_rag/results/retrieval_failures.md`.

**Main artifacts:**

- `task2_rag/questions.json`
- `task2_rag/results/retrieval_metrics.json`
- `task2_rag/results/retrieval_failures.md`
- `task2_rag/docs/`

---

# 5. Task 3 — Local LLM Inference and Quantization

## Objective

Compare FP16 and 4-bit quantized local language models using Ollama, evaluate inference speed and output quality, and estimate memory requirements for concurrent serving.

## Models

Two quantization configurations of the same model were evaluated:

- `qwen2.5:1.5b-instruct-fp16`
- `qwen2.5:1.5b-instruct-q4_K_M`

## Benchmark Configuration

| Parameter | Value |
|---|---|
| Benchmark prompts | 10 |
| Repetitions | 2 |
| Maximum generated tokens | 200 |
| Context length | 2048 |
| Runtime | Ollama |
| GPU | NVIDIA Tesla T4 |

## Benchmark Results

| Metric | FP16 | Q4_K_M |
|---|---:|---:|
| Mean generation speed | 62.23 tokens/sec | 135.50 tokens/sec |
| Model load time | 64.68 sec | 25.31 sec |
| Model memory | 3.216 GB | 1.109 GB |
| NVIDIA GPU memory delta | 3,202 MiB | 1,080 MiB |
| Process RSS | 38.97 MiB | 39.15 MiB |
| Automated checks passed | 6/7 | 7/7 |

### Quantization Findings

- Q4_K_M achieved approximately 2.18 times the generation speed of FP16.
- Q4_K_M reduced Ollama-reported model memory by approximately 65.5%.
- Six out of ten corresponding outputs were identical.
- Average text similarity was approximately 0.76.
- Some generated responses were truncated due to the 200-token generation limit.

These results demonstrate the practical trade-off between model memory, inference speed and output quality.

## KV-Cache Analysis

The KV-cache calculator uses the following assumptions:

| Parameter | Value |
|---|---:|
| Transformer layers | 28 |
| KV heads | 2 |
| Head dimension | 128 |
| Data type | FP16 |
| Context length | 2,048 tokens |
| Number of users | 20 |

Estimated KV-cache memory:

- Approximately 28,672 bytes per token.
- Approximately 0.0587 GB per user at 2,048 tokens.
- Approximately 1.17 GB for 20 users.

## Concurrent Serving Experiment

A 20-user concurrency experiment was conducted using the Q4_K_M model.

| Measurement | Result |
|---|---:|
| Concurrent requests | 20 |
| Successful requests | 20 |
| Failed requests | 0 |
| Total wall-clock time | 10.683 seconds |
| Individual request time range | 3.974–10.632 seconds |

This is an experimental concurrency result, not a production SLA guarantee.

## Serving Plan

The serving calculator evaluates FP16, Q4 and Q4 with an FP8 KV-cache assumption.

The estimates use a reported 14.56 GB T4 GPU memory capacity, 90% usable memory, and an overhead assumption of 2 GB.

| Configuration | Estimated memory | Estimated maximum users |
|---|---:|---:|
| FP16 | 6.25 GB | 136 |
| Q4 | 4.04 GB | 174 |
| Q4 + FP8 KV cache | 3.45 GB | 348 |

These are planning estimates based on stated assumptions, not measured production capacities. FP8 KV-cache support was hypothetical in this experiment and was not enabled in Ollama.

**Main artifacts:**

- `task3_local_inference/results/benchmark.json`
- `task3_local_inference/results/quality_comparison.md`
- `task3_local_inference/results/concurrency_20_users.json`
- `task3_local_inference/results/serving_plan.json`

---

# 6. Hardware and Software Environment

## Hardware

| Component | Environment |
|---|---|
| Execution platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Reported GPU memory | 14.56 GB |
| Model training | CUDA GPU |
| Local inference | Ollama |

## Main Software Components

| Component | Version / Technology |
|---|---|
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.22.1 |
| PEFT | 0.21.2 |
| Accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| Ollama | 0.40.16 |
| PDF generation | ReportLab |
| Retrieval | Embeddings, FAISS/NumPy and BM25 |

Exact environment details and experiment configurations are retained in the corresponding project artifacts.

---

# 7. Reproducing the Experiments

The following commands describe the main execution workflow. Dependencies and model availability may vary depending on the execution environment.

## Prerequisites

- Python environment.
- CUDA-compatible GPU for GPU training and applicable quantization workflows.
- Required Python packages from the task-specific requirements files.
- Ollama for Task 3.
- Sufficient disk space for model downloads and experiment outputs.

## Task 1

```bash
cd task1_lora

pip install -r requirements.txt

bash run_all.sh

bash ablation.sh
```

The training and evaluation outputs are stored under `outputs/` and `results/`.

## Task 2

```bash
cd task2_rag

pip install -r requirements.txt

python fetch_docs.py

python evaluate_retrieval.py \
    --chunk-sizes 100 250 \
    --rerank
```

The document corpus, question annotations and retrieval evaluation results are stored in the `docs/` and `results/` directories.

## Task 3

Pull the model variants:

```bash
ollama pull qwen2.5:1.5b-instruct-fp16

ollama pull qwen2.5:1.5b-instruct-q4_K_M
```

Run the benchmark:

```bash
cd task3_local_inference

python benchmark_ollama.py \
    --models \
    qwen2.5:1.5b-instruct-fp16 \
    qwen2.5:1.5b-instruct-q4_K_M
```

Run the supporting analysis:

```bash
python compare_quality.py

python kv_cache_calc.py --ctx 2048 --users 20

python serving_plan.py --users 20 --ctx 2048
```

The committed result files contain the actual experiment configurations and outputs. Re-running experiments may produce different results depending on hardware, software and runtime conditions.

---

# 8. Testing and Verification

The project includes offline tests for core implementation logic.

Run:

```bash
bash run_tests.sh
```

The offline test suite covers components such as:

- Dataset generation and scoring.
- Label masking.
- Training and evaluation using a small random model.
- RAG chunking and BM25 retrieval.
- FAISS/NumPy retrieval logic.
- Retrieval metrics.
- Ollama benchmarking with a mock server.
- PDF report generation.

Offline tests validate code logic; they are not substitutes for the actual GPU training, document retrieval or Ollama experiments.

## Final Submission Verification

The final submission archive was generated and checked.

| Verification | Result |
|---|---|
| Required project files | 16/16 PASS |
| Final PDF page count | 4 pages — PASS |
| ZIP integrity test | PASS |
| Files packaged | 74 |
| ZIP size | 36.94 MB |
| Task 1–3 results | Included |
| Final report | Included |

**Final archive:** `tndrr-assignment-FINAL-SUBMISSION.zip`

---

# 9. Final Report

The final report is available at:

```text
report/report.pdf
```

It was generated using:

```text
report/build_report.py
```

The report includes:

- Task 1 LoRA configuration, evaluation and failure analysis.
- Task 1 memory analysis.
- Task 2 chunk-size and retrieval comparisons.
- Task 2 retrieval failure analysis.
- Task 3 quantization and quality observations.
- KV-cache calculations.
- Concurrent serving experiment.
- Serving capacity estimates.

The final PDF was verified to contain exactly four pages.

---

# 10. Important Notes and Limitations

1. Evaluation results are specific to the datasets, models, prompts and hardware used in the experiments.
2. The Task 1 shifted-distribution evaluation demonstrates that performance can decrease on examples that differ from the training distribution.
3. Task 2 retrieval metrics are based on 15 annotated questions and should not be interpreted as universal retrieval performance.
4. Task 3 benchmark results depend on Ollama, hardware utilization, model configuration and generation settings.
5. Serving capacity figures are estimates based on documented memory and bandwidth assumptions.
6. The 20-user concurrency experiment is a functional test and does not establish production-grade latency or throughput guarantees.
7. Offline tests validate implementation logic but do not replace real-model experiments.

---

# 11. Submission

**Candidate:** Aadit Choudhary

**Repository:**  
https://github.com/Aadit23042006/tndrr-ai-ml-assignment

**Final deliverables:**

- Complete source code for Tasks 1, 2 and 3.
- Trained LoRA adapter and training summary.
- RAG document corpus and retrieval evaluation results.
- Ollama benchmark and serving analysis results.
- Final four-page PDF report.
- Complete ZIP submission archive.

**Submission deadline specified in the assignment:** Thursday, 19 November 2026, 11:59 PM IST.

---

## Acknowledgement

This project was completed as a practical AI/ML engineering assignment involving parameter-efficient fine-tuning, retrieval evaluation, local language-model inference, quantization and deployment-oriented memory analysis.

The repository preserves the implementation, experiment outputs and supporting documentation for review and reproducibility.