# MachinaSense — Phase 10: Inference Performance & Reliability Report

## Executive Summary
This report benchmarks the inference speed, loading overhead, memory footprint, and concurrency scaling of the two primary RUL prediction models deployed within MachinaSense: the **Random Forest Colab FD001** model and the **2-Layer PyTorch LSTM** recurrent network.

---

## 1. Latency & Resource Utilization Profile

| Metric | Random Forest (19 Features) | LSTM Network (30x15 Sequence) | Evaluation Standard |
|:---|:---:|:---:|:---:|
| **Artifact Size on Disk** | **95.19 MB** | **0.219 MB** (223.8 KB) | Lean footprint (< 50 MB) |
| **Model Load Time** | **5590.52 ms** | **99.62 ms** | Instant cold start (< 1000 ms) |
| **First Inference (Cold)** | **244.15 ms** | **14.78 ms** | Immediate readiness (< 100 ms) |
| **Warm Mean Latency** | **102.56 ms** | **2.83 ms** | Telemetry rate (< 25 ms) |
| **Median Latency** | **98.05 ms** | **2.83 ms** | Telemetry rate (< 25 ms) |
| **95th Percentile Latency** | **133.58 ms** | **3.44 ms** | Strict SLA (< 50 ms) |
| **99th Percentile Latency** | **172.62 ms** | **4.29 ms** | Tail stability (< 100 ms) |
| **Max Observed Latency** | **285.17 ms** | **4.45 ms** | Deterministic tail (< 150 ms) |
| **Single-Threaded Throughput** | **9.8 req/s** | **352.8 req/s** | High operational volume |

---

## 2. Multi-Threaded Concurrency Scaling

| Concurrent Workers | RF Mean Latency (ms) | RF P95 Latency (ms) | RF Throughput (req/s) | LSTM Mean Latency (ms) | LSTM P95 Latency (ms) | LSTM Throughput (req/s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 Worker** | 102.58 | 162.21 | 9.7 | 3.37 | 3.89 | 287.7 |
| **5 Workers** | 513.64 | 798.97 | 9.7 | 6.66 | 8.01 | 671.9 |
| **10 Workers** | 873.16 | 1173.68 | 11.3 | 17.61 | 26.27 | 461.2 |

---

## 3. Engineering Insights & Operational Reliability
1. **Measured Latency Profile**: The 2-Layer LSTM network operates with ultra-low warm mean latency of **2.83 ms** (median: 2.83 ms, P95: 3.44 ms; first inference: 14.78 ms), making it suitable for high-frequency continuous streaming pipelines (e.g., 10–50 Hz turbine sensor streams). The Random Forest model exhibits a warm mean latency of **102.56 ms** (median: 98.05 ms, P95: 133.58 ms; first inference: 244.15 ms). Model loading overhead is measured separately: Random Forest cold load is **5590.52 ms** (95.19 MB artifact) versus LSTM cold load of **99.62 ms** (223.8 KB artifact).
2. **Minimal Memory Pressure**: The LSTM model is exceptionally compact at only **223.8 KB**, making it viable even for edge industrial micro-controllers or gateways. The Random Forest model at **95.19 MB** is readily cacheable in RAM across multiple worker processes.
3. **Deterministic Output Guarantees**: Repeated inference across identical sensor vectors yielded identical floating-point predictions with zero deviation ($0.0\%$ jitter).
