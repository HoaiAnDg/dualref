# Experimental Evaluation & Anomaly Audit Report
- **Generated at:** 2026-11-19 12:40:49
- **Input files:** `Data\stage3_batch1_gsm8k_test_1000.jsonl`
- **SymPy symbolic equivalence engine:** Enabled

## 1. Data Integrity & Coverage Audit
- **Total valid JSON records loaded:** `2000`
- **Malformed JSON lines:** `0`
- **Duplicate `(method, seed, id)` records:** `0`
- **Records missing core schema fields:** `0`
- **Internal field logic mismatches:** `0`

| Method / Seed | Records | Unique IDs | Missing MATH-500 IDs | Sample Missing IDs |
|---|---:|---:|---:|---|
| DIRECT (seed=0) | 1000 | 1000 | 0 | - |
| DUALREF (seed=0) | 1000 | 1000 | 0 | - |

## 2. Overall Performance, Oracle & Compute Efficiency by Method
| Metric | DIRECT | DUALREF |
|---|---:|---:|
| Sample Count (`N`) | 1000 | 1000 |
| **Raw Accuracy (%)** | **78.90%** (789/1000) | **87.00%** (870/1000) |
| 95% Wilson CI | [76.26%, 81.32%] | [84.77%, 88.94%] |
| Suspected False Negatives | 0 | 0 |
| **Adjusted Accuracy (%)** | **78.90%** | **87.00%** |
| Oracle Any Correct (%) | - | 94.60% |
| Oracle Gap (Oracle - Raw) | - | 7.60% |
| PAL Any Correct (%) | - | 86.80% |
| Cross-Validated Rate (%) | - | 83.50% |
| Acc when Cross-Validated | - | 93.29% |
| Acc when NOT Cross-Validated | 78.90% | 55.15% |
| Mean Samples / Query | - | 4.08 |
| Mean LLM Calls / Query | 1.07 | 5.29 |
| Mean Total Tokens | 393.9 | 2358.4 |
| Median (P50) Total Tokens | 342.0 | 1399.0 |
| P95 Total Tokens | 732.4 | 7039.1 |
| Total Tokens (All Queries) | 393,864 | 2,358,441 |
| Mean Latency (`elapsed_sec`) | 16.75s | 71.60s |
| Median (P50) Latency | 13.81s | 44.28s |
| P95 / Max Latency | 33.83s / 134.94s | 226.58s / 482.61s |
| Total Wall-Clock Time | 279.1 min | 1193.4 min |
| Acc% per 1k Tokens | 200.32 | 36.89 |
| Any Truncated (`n_truncated>0`) | 2.70% | 5.50% |
| CoT Truncated (`n_cot_truncated>0`) | 0.00% | 4.80% |
| Loop Guard Abort (%) | 0.00% | 0.00% |

## 3. Paired Head-to-Head Comparison (McNemar's Test)
### `DUALREF` vs. Baseline `DIRECT` (Paired N = 1000)
- **Wins (`0 -> 1`):** `122` | **Losses (`1 -> 0`):** `41` | **Net Gain:** `+81` (`+8.10%`)
- **Both Correct:** `748` | **Both Wrong:** `89`
- **McNemar's Test:** $\chi^2 = 39.2638$, $p = 0.000000$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 1000 | 78.90% | 87.00% | +8.10% | 122 | 41 | +81 |

## 4. Method Internals: Stop Reasons, Routing & Auxiliary Modules
### Method: `DIRECT`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SINGLE_SAMPLE` | 973 | 97.3% | 81.09% | - | 0.00 | 375 | 15.25s |
| `NO_ANSWER` | 27 | 2.7% | 0.00% | - | 0.00 | 1080 | 70.64s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 1000 | 78.90% | 78.90% | - | 0.00 | 394 | 16.75s |

### Method: `DUALREF`
- **Auxiliary Modules:** PAL Exec Success = `2035/2089` (97.4%), Debug Fixed = `203/257` (79.0%)
- **Critic & Vetoes:** Critic Calls = `699`, Parse Fails = `0`, Local Vetoes = `178` (17.80%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CROSS_VALIDATED` | 835 | 83.5% | 93.29% | 96.65% | 3.71 | 1921 | 58.36s |
| `MARGIN` | 106 | 10.6% | 70.75% | 90.57% | 4.25 | 2694 | 75.84s |
| `KMAX` | 58 | 5.8% | 27.59% | 72.41% | 9.00 | 7971 | 253.33s |
| `PAL_DOMINANT` | 1 | 0.1% | 0.00% | 100.00% | 9.00 | 6439 | 142.03s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `PAL_FIRST` | 551 | 55.1% | 86.57% | 95.64% | 2076 | 61.16s |
| `COT_FIRST` | 449 | 44.9% | 87.53% | 93.32% | 2705 | 84.42s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 1000 | 87.00% | 87.00% | 94.60% | 4.08 | 2358 | 71.60s |

## 5. Anomaly & Error Audit Summary
- **Total flagged records:** `170` (full details exported to `anomalies_audit.csv`)
| Anomaly Tag | Count |
|---|---:|
| `ORACLE_LOSS` | 76 |
| `TOKEN_ESTIMATED_FALLBACK` | 73 |
| `FALSE_CROSS_VALIDATION` | 56 |
| `REGRESSION_VS_BASELINE` | 41 |
| `EXTRACTION_OR_VOTE_POLLUTION` | 27 |
| `COMPUTE_LATENCY_OUTLIER` | 22 |
