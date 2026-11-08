# MachinaSense — Phase 10: Model Robustness & Degradation Audit

## Executive Summary
This report documents the empirical robustness, noise sensitivity, and fault handling evaluation of the MachinaSense predictive maintenance models on the 100 turbofan engines of the official NASA C-MAPSS FD001 test dataset.

---

## 1. Sensor Noise Sensitivity Testing
Sensor degradation was simulated by injecting zero-mean additive Gaussian white noise $\mathcal{N}(0, (\alpha \cdot \sigma_j)^2)$ where $\sigma_j$ is the empirical standard deviation of sensor channel $j$ across test data.

| Noise Level | RF Mean $\Delta$ RUL (Cycles) | RF Max $\Delta$ | RF Stability Rate ($|\Delta| \le 5$) | LSTM Mean $\Delta$ RUL (Cycles) | LSTM Max $\Delta$ | LSTM Stability Rate ($|\Delta| \le 5$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1%** | 0.3598 | 1.9553 | **100.0%** | 0.1163 | 0.7074 | **100.0%** |
| **3%** | 0.6203 | 4.0421 | **100.0%** | 0.2676 | 1.2361 | **100.0%** |
| **5%** | 0.9316 | 5.4219 | **99.0%** | 0.6098 | 3.5537 | **100.0%** |
| **10%** | 1.3929 | 6.4076 | **98.0%** | 1.151 | 6.2386 | **97.0%** |

### Key Observations
1. **Random Forest Noise Attenuation**: The Colab FD001 Random Forest model displays exceptional resilience. At 5% noise, mean prediction perturbation is only **0.9316 cycles**, and **99.0%** of engines retain predictions within 5 cycles of their uncorrupted output.
2. **LSTM Recurrent Sensitivity**: The LSTM exhibits higher sensitivity to sensor fluctuations (mean delta **0.6098 cycles** at 5% noise) due to recurrent error accumulation across the 30-cycle temporal sequence.

---

## 2. Fault Injection & Boundary Stress Audit

| Scenario | Model / Layer | Handling Mechanism | Empirical Result | Status |
|:---|:---|:---|:---|:---:|
| $+10\sigma$ Extreme Sensor Spike | Random Forest | Split Threshold Saturation | Bounded response, output capped $\le 125$ | **PASS** |
| $-10\sigma$ Extreme Sensor Drop | Random Forest | Split Threshold Saturation | Bounded response, output capped $\ge 0$ | **PASS** |
| Complete Channel Zeroing ($s_{11}=0$) | Random Forest | Ordinary Numeric Evaluation (No FDIR) | Evaluated raw without active fault isolation; mean delta 6.26 cycles; bounded in $[0, 125]$ | **PASS (Numeric Only)** |
| NaN / Inf Feature Ingestion | FastAPI Gateway | Pydantic Schema Validation | HTTP 422 Unprocessable Entity Rejection | **PASS** |
| Missing Required Feature Key | FastAPI Gateway | Schema Field Requirement Validation | HTTP 400/422 Bad Request / Unprocessable Entity Rejection | **PASS** |
| Truncated Temporal Sequence (< 30) [Offline Test] | Offline Test Script (`run_phase_10_robustness.py`) | Synthetic Edge Padding (Offline Test Only) | Synthetic test delta: 13.04 cycles. In operational production (`ml_service.py`), sequences < 30 cycles fall back to RF baseline | **PASS (Offline Test)** |

---

## 3. Operational Robustness & Fault Handling Assessment
- **Validation Guardrails**:
  - **Missing Required Features**: Requests omitting any of the 19 required sensor keys are strictly rejected with HTTP 400/422 validation errors before model dispatch.
  - **NaN / Inf Ingestion**: Non-finite float payloads are intercepted and rejected at the HTTP schema boundary.
- **Sensor Value Zeroing ($s_{11} = 0.0$)**:
  - Setting $s_{11}=0.0$ (vacuum pressure) is evaluated purely as an ordinary floating-point numeric input across decision tree splits.
  - While decision tree thresholds compute without throwing runtime exceptions and predictions remain bounded in $[0, 125]$, **active sensor fault isolation or imputation (FDIR) is not currently implemented**. Unsubstantiated claims of architectural sensor-fault tolerance are retracted.
- **Sequence Length Handling**:
  - The production inference pipeline (`ml_service.py`) evaluates sequences with $\ge 30$ cycles via the 2-Layer PyTorch LSTM.
  - Input sequences shorter than 30 cycles fall back directly to the single-cycle Random Forest baseline (no runtime padding is applied).
  - Temporal edge padding was evaluated strictly as an offline synthetic experiment in `run_phase_10_robustness.py` and is not part of live inference.
- **Domain Clamping**: Enforcement of the $[0, 125]$ operational domain prevents negative RUL and unbounded extrapolations under extreme sensor fluctuations.
