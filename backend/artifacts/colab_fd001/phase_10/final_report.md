# MachinaSense — Phase 10: Explainability, Robustness & Deployment Readiness Audit
**Official Final Report & Engineering Verification Audit**
**Date**: November 2026 | **Dataset**: NASA C-MAPSS FD001 | **Status**: CONDITIONALLY READY (Static Verification Passed; Live Container Runtime Pending)

---

## Executive Summary
Phase 10 upgrades the engineering rigor and production readiness of the MachinaSense predictive maintenance platform. Rather than retraining existing models, this phase subjected the deployed **Colab FD001 Random Forest model** and the **2-Layer PyTorch LSTM recurrent network** to comprehensive explainability attributions, noise degradation stress testing, inference latency profiling, and deployment security audits.

All empirical metrics, sample engine predictions, and performance figures reported below have been verified against source evaluation artifacts (`official_test_predictions.csv`, `rf_explainability.json`, `lstm_explainability.json`, and `inference_performance.json`).

---

## 1. Explainability & Model Attribution Intelligence

### 1.1 Random Forest Global Feature Importance
Feature importance was audited using two independent methodologies on the 19 unscaled features:
1. **Gini Impurity (Mean Decrease in Impurity - MDI)** across 200 decision trees.
2. **Test-Set Permutation Feature Importance** ($N=100$ engines, 30 repeats, scoring negative MAE).

| Rank | Feature | Physical Sensor Name | Gini Importance (%) | Permutation MAE Increase (Cycles) |
|:---:|:---|:---|:---:|:---:|
| **1** | `cycle` | Operating Cycle Counter | 21.02% | +3.97 ± 0.98 |
| **2** | `sensor_11` | Static Pressure at HPC Outlet ($P_{30}$) | 14.85% | +2.54 ± 0.44 |
| **3** | `sensor_9` | Physical Fan Speed ($N_f$) | 5.86% | +2.01 ± 0.54 |
| **4** | `sensor_4` | LPT Outlet Temperature ($T_{50}$) | 13.89% | +1.54 ± 0.30 |
| **5** | `sensor_14` | Corrected Core Speed ($N_{c,corr}$) | 4.88% | +1.72 ± 0.47 |

**Physical Finding**: The dominant degradation indicators identified by the model strictly align with gas turbine thermodynamics: Static Pressure at HPC Outlet (`sensor_11`), Low Pressure Turbine Outlet Temp (`sensor_4`), and Core Speed (`sensor_14`) represent the primary physical drivers of turbofan blade erosion and compressor fouling.

### 1.2 LSTM Recurrent Gradient Saliency & Temporal Dynamics
The deployed LSTM model is a standard two-layer `nn.LSTM` (`hidden_size=64, num_layers=2`) operating over a 30-cycle by 15-sensor sequence tensor **without attention mechanisms**. Temporal and sensor attributions were computed using first-order gradient saliency ($|\nabla_X Y|$) and Input $\times$ Gradient across all 100 test engines:
- **Top Sensors**: `sensor_9` (Physical Fan Speed), `sensor_11` (Static Pressure at HPC Outlet), `sensor_8` (Physical Core Speed), `sensor_14` (Corrected Core Speed), and `sensor_13` (Corrected Fan Speed).
- **Recurrent Sensitivity Dynamics**: Temporal gradient saliency exhibits an exponential concentration toward the end of the 30-cycle observation window. Cycles $T-5$ to $T-1$ account for **over 52%** of total recurrent gradient attribution, peaking at cycle $T-2$ (8.49 saliency), confirming that the recurrent architecture prioritizes late-stage degradation dynamics.

### 1.3 Local Engine Explanations (Verified Source Values)
Local feature sensitivity and attributions were evaluated on 5 representative test engines across operational regimes. Predictions and residuals ($Residual = Predicted - Actual$) are confirmed against `official_test_predictions.csv`, `rf_explainability.json`, and `lstm_explainability.json`:

| Engine | Operating Regime | Ground Truth (Capped) | RF Prediction | RF Residual | LSTM Prediction | LSTM Residual |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Unit #1** | Nominal Health (Upper Bound) | 112.00 | 122.73 | +10.73 | 110.20 | -1.80 |
| **Unit #34** | Terminal Failure Domain | 7.00 | 7.00 | 0.00 | 6.20 | -0.80 |
| **Unit #45** | Outlier / Early Truncation | 114.00 | 68.81 | -45.19 | 90.24 | -23.76 |
| **Unit #77** | Mid-Life Wear Regime | 34.00 | 34.44 | +0.44 | 33.15 | -0.85 |
| **Unit #82** | Terminal Failure Domain | 9.00 | 9.05 | +0.05 | 8.91 | -0.09 |

- **Unit #1** (Actual RUL: 112.00): Upper healthy regime. RF predicted 122.73 (residual +10.73); LSTM predicted 110.20 (residual -1.80). Primary sensitivity governed by `sensor_14` and `sensor_9`.
- **Unit #34** (Actual RUL: 7.00): Critical terminal failure. RF predicted 7.00 (residual 0.00); LSTM predicted 6.20 (residual -0.80). Driven by elevated HPC static pressure (`sensor_11`) and LPT temperature (`sensor_4`).
- **Unit #45** (Actual RUL: 114.00): Challenging early-cycle profile. RF predicted 68.81 (residual -45.19); LSTM predicted 90.24 (residual -23.76), demonstrating LSTM's superior multi-cycle sequence modeling for non-monotonic transients.
- **Unit #77** (Actual RUL: 34.00): Mid-life wear. RF predicted 34.44 (residual +0.44); LSTM predicted 33.15 (residual -0.85). Both models achieve sub-cycle accuracy.
- **Unit #82** (Actual RUL: 9.00): Terminal failure. RF predicted 9.05 (residual +0.05); LSTM predicted 8.91 (residual -0.09). Exceptional near-zero residual tracking.

---

## 2. Robustness, Degradation & Fault Tolerance Audit

### 2.1 Gaussian Sensor Noise Sensitivity
Zero-mean Gaussian noise was injected into sensor streams across the official test set:

| Noise Level | RF Mean Δ RUL | RF Stability (|Δ| ≤ 5) | LSTM Mean Δ RUL | LSTM Stability (|Δ| ≤ 5) |
|:---:|:---:|:---:|:---:|:---:|
| **1%** | 0.36 cycles | **100.0%** | 0.12 cycles | **100.0%** |
| **3%** | 0.62 cycles | **100.0%** | 0.27 cycles | **100.0%** |
| **5%** | 0.93 cycles | **99.0%** | 0.61 cycles | **100.0%** |
| **10%** | 1.39 cycles | **98.0%** | 1.15 cycles | **97.0%** |

### 2.2 Boundary & Fault Stress Scenarios
- **Missing Required Feature**: Ingestion payloads omitting any of the 19 required features are strictly rejected at the API boundary with HTTP 400/422 validation errors before reaching any model.
- **NaN / Inf Ingestion**: Non-finite float values are rejected by Pydantic schema validation at the HTTP edge, preventing NaN propagation.
- **Zero Sensor Input ($s_{11} = 0.0$)**: Setting $s_{11}=0.0$ (vacuum pressure) is evaluated as an ordinary numeric float input across decision tree splits. Predictions remain bounded in $[0, 125]$ (mean delta 6.26 cycles). However, **active sensor fault detection, isolation, or imputation (FDIR) is not currently implemented in the architecture**. No architectural sensor-fault tolerance mechanism should be claimed without implementation evidence.
- **±10σ Extreme Sensor Spikes**: Ensembled decision trees saturate at extreme leaf thresholds; predictions are strictly constrained by the $[0, 125]$ operational clamping layer.
- **Sequence Handling (< 30 cycles)**:
  - In operational production (`ml_service.py`), sequences with $\ge 30$ cycles utilize the LSTM inference pipeline.
  - Telemetry sequences shorter than 30 cycles automatically fall back to the single-cycle Random Forest baseline (no runtime padding is applied in production).
  - Temporal edge padding was performed strictly as an offline synthetic experiment in `run_phase_10_robustness.py` (mean delta: 13.04 cycles across 15-cycle truncated inputs) and is not part of the live backend inference flow.

---

## 3. Inference Reliability & Performance Profiling

Model loading overhead and inference latency are measured and reported separately:

| Performance Metric | Random Forest (19 Features) | LSTM Network (30x15 Sequence) | Operational Benchmark |
|:---|:---:|:---:|:---:|
| **Artifact Size on Disk** | 95.19 MB | 223.8 KB (0.219 MB) | Compact storage |
| **Cold Model Loading Overhead** | **5590.52 ms** | **99.62 ms** | Startup / worker init |
| **First Inference (Cold Start)** | **244.15 ms** | **14.78 ms** | Initial request compilation |
| **Warm Mean Latency** | **102.56 ms** | **2.83 ms** | Steady-state telemetry |
| **Median Latency** | **98.05 ms** | **2.83 ms** | Typical request |
| **95th Percentile Latency (P95)** | **133.58 ms** | **3.44 ms** | Tail latency SLA |
| **99th Percentile Latency (P99)** | **172.62 ms** | **4.29 ms** | Tail stability |
| **Max Observed Latency** | **285.17 ms** | **4.45 ms** | Worst-case bound |
| **Single-Threaded Throughput** | **9.8 req/s** | **352.8 req/s** | Single worker |
| **10-Worker Concurrency Throughput** | **11.3 req/s** | **461.2 req/s** | ThreadPool scaling |

**Concurrency Analysis**:
- The PyTorch LSTM exhibits excellent concurrency scaling (from 287.7 req/s at 1 worker to 671.9 req/s at 5 workers and 461.2 req/s at 10 workers).
- The Random Forest model experiences Python GIL contention under multi-threading (mean latency increases from 102.58 ms at 1 worker to 873.16 ms at 10 workers, with throughput plateauing at 11.3 req/s). High-throughput production deployments of the RF model require multi-process process-pooling (e.g., Uvicorn `--workers 4+`) rather than in-process threading.
- Unverified placeholder claims (such as RF averaging ~8ms or both models having cold-start inference below 50ms) are completely retracted.

---

## 4. Deployment Readiness & Security Audit

- **Static Secret Scanning**: 0 hardcoded credentials or private keys detected across 23 backend application files.
- **Authentication**: Clerk RS256 JWT validation enforced; dev mock bypass is strictly restricted to `ENVIRONMENT == 'development'`.
- **CORS Policy**: Wildcard `"*"` disallowed in production; origins locked to verified domains.
- **Health Check**: `GET /health` operational for Kubernetes / Cloud Load Balancer readiness probes.
- **Containerization Assessment**:
  - Multi-stage Dockerfile and Docker Compose syntax verified statically.
  - **Static Docker configuration checks passed. Live container build and runtime verification remain pending** due to an inactive Docker daemon during host audit execution.
- **Frontend SPA**: React Vite build compiles cleanly with zero TypeScript errors in 2.86s.

---

## 5. Artifact Manifest
All Phase 10 artifacts are persisted in `backend/artifacts/colab_fd001/phase_10/`:
- `rf_global_feature_importance.csv`
- `rf_permutation_importance.csv`
- `rf_explainability.json`
- `rf_feature_importance.png`
- `lstm_sensor_importance.csv`
- `lstm_temporal_importance.csv`
- `lstm_explainability.json`
- `lstm_sensor_importance.png`
- `lstm_temporal_importance.png`
- `robustness_test_results.csv`
- `robustness_summary.json`
- `robustness_report.md`
- `sensor_noise_robustness.png`
- `prediction_stability.png`
- `inference_performance.csv`
- `inference_performance.json`
- `inference_performance_report.md`
- `inference_latency_comparison.png`
- `deployment_readiness_checklist.json`
- `security_audit_report.md`
- `deployment_audit_report.md`
- `docker_verification_report.md`
- `phase_10_summary.json`
- `final_report.md`
- `verification_report.md` (Phase 10.1 Independent Audit)
- `phase_10_2_correction_report.md` (Phase 10.2 Evidence-Based Corrections)

---

## Phase 10 Audit Verdict: CONDITIONALLY READY (Static Verification Complete; Live Runtime Pending)
- **Scientific Computations**: Fully verified (PASS). RF Gini/permutation importance, LSTM gradient saliency, noise degradation matrices, and official test evaluations are mathematically sound and reproducible.
- **Security & Packaging**: Fully verified (PASS). Zero hardcoded secrets, strict CORS, and RS256 JWT auth enforced.
- **Operational Qualifications**:
  1. Static Docker configuration checks passed; live container build and runtime verification remain pending active Docker daemon testing.
  2. RF inference latency (102.56 ms mean, 133.58 ms P95) requires multi-process workers (`--workers 4+`) for multi-client concurrent ingestion.
  3. Short sequence inputs (< 30 cycles) fall back to the RF baseline in production; no live temporal padding is implemented.
  4. Zeroed sensor input ($s_{11}=0$) is evaluated as raw numeric input; no active fault isolation mechanism is currently implemented.
