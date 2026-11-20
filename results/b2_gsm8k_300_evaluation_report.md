# Experimental Evaluation & Anomaly Audit Report
- **Generated at:** 2026-11-20 15:49:34
- **Input files:** `Data\stage3_batch2_gsm8k_test_1000_300.jsonl`
- **SymPy symbolic equivalence engine:** Enabled

## 1. Data Integrity & Coverage Audit
- **Total valid JSON records loaded:** `1800`
- **Malformed JSON lines:** `0`
- **Duplicate `(method, seed, id)` records:** `0`
- **Records missing core schema fields:** `0`
- **Internal field logic mismatches:** `0`

| Method / Seed | Records | Unique IDs | Missing MATH-500 IDs | Sample Missing IDs |
|---|---:|---:|---:|---|
| DUALREF (seed=0) | 300 | 300 | 0 | - |
| DUALREF_NOCRIT (seed=0) | 300 | 300 | 0 | - |
| PAL_ONLY (seed=0) | 300 | 300 | 0 | - |
| PAL_POOL12 (seed=42) | 300 | 300 | 0 | - |
| SC_POOL12 (seed=42) | 300 | 300 | 0 | - |
| V5_COT (seed=0) | 300 | 300 | 0 | - |

## 2. Overall Performance, Oracle & Compute Efficiency by Method
| Metric | DUALREF | DUALREF_NOCRIT | PAL_ONLY | PAL_POOL12 | SC_POOL12 | V5_COT |
|---|---:|---:|---:|---:|---:|---:|
| Sample Count (`N`) | 300 | 300 | 300 | 300 | 300 | 300 |
| **Raw Accuracy (%)** | **88.67%** (266/300) | **85.00%** (255/300) | **70.00%** (210/300) | **85.00%** (255/300) | **89.67%** (269/300) | **83.67%** (251/300) |
| 95% Wilson CI | [84.58%, 91.78%] | [80.52%, 88.60%] | [64.59%, 74.91%] | [80.52%, 88.60%] | [85.71%, 92.62%] | [79.06%, 87.42%] |
| Suspected False Negatives | 0 | 0 | 0 | 0 | 0 | 0 |
| **Adjusted Accuracy (%)** | **88.67%** | **85.00%** | **70.00%** | **85.00%** | **89.67%** | **83.67%** |
| Oracle Any Correct (%) | 96.00% | 95.00% | 82.00% | - | - | 90.33% |
| Oracle Gap (Oracle - Raw) | 7.33% | 10.00% | 12.00% | - | - | 6.67% |
| PAL Any Correct (%) | 89.67% | 84.33% | 82.00% | - | - | 0.00% |
| Cross-Validated Rate (%) | 87.00% | 80.33% | 0.00% | - | - | 0.00% |
| Acc when Cross-Validated | 93.49% | 92.95% | - | - | - | - |
| Acc when NOT Cross-Validated | 56.41% | 52.54% | 70.00% | 85.00% | 89.67% | 83.67% |
| Mean Samples / Query | 4.20 | 4.27 | 5.71 | - | - | 5.79 |
| Mean LLM Calls / Query | 5.51 | 5.19 | 9.56 | 13.35 | 14.10 | 9.80 |
| Mean Total Tokens | 2464.9 | 2334.6 | 5140.2 | 6456.8 | 5180.7 | 4458.1 |
| Median (P50) Total Tokens | 1456.5 | 1550.5 | 4700.0 | 5943.0 | 4749.0 | 3715.5 |
| P95 Total Tokens | 7089.9 | 6407.1 | 9795.0 | 9696.0 | 8376.0 | 8658.4 |
| Total Tokens (All Queries) | 739,480 | 700,386 | 1,542,046 | 1,937,053 | 1,554,204 | 1,337,419 |
| Mean Latency (`elapsed_sec`) | 53.88s | 43.01s | 12.26s | 91.33s | 149.89s | 14.18s |
| Median (P50) Latency | 27.12s | 25.07s | 10.44s | 80.57s | 126.63s | 10.39s |
| P95 / Max Latency | 182.80s / 415.75s | 141.18s / 360.51s | 27.18s / 41.05s | 172.26s / 242.08s | 321.63s / 605.10s | 33.03s / 83.72s |
| Total Wall-Clock Time | 269.4 min | 215.0 min | 61.3 min | 456.6 min | 749.5 min | 70.9 min |
| Acc% per 1k Tokens | 35.97 | 36.41 | 13.62 | 13.16 | 17.31 | 18.77 |
| Any Truncated (`n_truncated>0`) | 5.67% | 3.67% | 4.67% | 1.67% | 11.33% | 16.00% |
| CoT Truncated (`n_cot_truncated>0`) | 5.33% | 2.67% | 0.00% | 0.00% | 0.00% | 6.33% |
| Loop Guard Abort (%) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## 3. Paired Head-to-Head Comparison (McNemar's Test)
### `DUALREF_NOCRIT` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `15` | **Losses (`1 -> 0`):** `26` | **Net Gain:** `-11` (`-3.67%`)
- **Both Correct:** `240` | **Both Wrong:** `19`
- **McNemar's Test:** $\chi^2 = 2.4390$, $p = 0.118350$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 88.67% | 85.00% | -3.67% | 15 | 26 | -11 |

### `PAL_ONLY` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `11` | **Losses (`1 -> 0`):** `67` | **Net Gain:** `-56` (`-18.67%`)
- **Both Correct:** `199` | **Both Wrong:** `23`
- **McNemar's Test:** $\chi^2 = 38.7821$, $p = 0.000000$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 88.67% | 70.00% | -18.67% | 11 | 67 | -56 |

### `PAL_POOL12` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `14` | **Losses (`1 -> 0`):** `25` | **Net Gain:** `-11` (`-3.67%`)
- **Both Correct:** `241` | **Both Wrong:** `20`
- **McNemar's Test:** $\chi^2 = 2.5641$, $p = 0.109315$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 88.67% | 85.00% | -3.67% | 14 | 25 | -11 |

### `SC_POOL12` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `20` | **Losses (`1 -> 0`):** `17` | **Net Gain:** `+3` (`+1.00%`)
- **Both Correct:** `249` | **Both Wrong:** `14`
- **McNemar's Test:** $\chi^2 = 0.1081$, $p = 0.742308$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 88.67% | 89.67% | +1.00% | 20 | 17 | +3 |

### `V5_COT` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `17` | **Losses (`1 -> 0`):** `32` | **Net Gain:** `-15` (`-5.00%`)
- **Both Correct:** `234` | **Both Wrong:** `17`
- **McNemar's Test:** $\chi^2 = 4.0000$, $p = 0.045500$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 88.67% | 83.67% | -5.00% | 17 | 32 | -15 |

## 4. Method Internals: Stop Reasons, Routing & Auxiliary Modules
### Method: `DUALREF`
- **Auxiliary Modules:** PAL Exec Success = `633/647` (97.8%), Debug Fixed = `71/85` (83.5%)
- **Critic & Vetoes:** Critic Calls = `220`, Parse Fails = `0`, Local Vetoes = `55` (18.33%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CROSS_VALIDATED` | 261 | 87.0% | 93.49% | 97.32% | 3.85 | 2118 | 44.55s |
| `MARGIN` | 22 | 7.3% | 68.18% | 90.91% | 4.73 | 3002 | 67.20s |
| `KMAX` | 15 | 5.0% | 33.33% | 80.00% | 9.00 | 7274 | 185.92s |
| `PAL_DOMINANT` | 2 | 0.7% | 100.00% | 100.00% | 8.00 | 5734 | 135.67s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `PAL_FIRST` | 167 | 55.7% | 89.82% | 96.41% | 2216 | 46.67s |
| `COT_FIRST` | 133 | 44.3% | 87.22% | 95.49% | 2777 | 62.94s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 88.67% | 88.67% | 96.00% | 4.20 | 2465 | 53.88s |

### Method: `DUALREF_NOCRIT`
- **Auxiliary Modules:** PAL Exec Success = `613/657` (93.3%), Debug Fixed = `84/128` (65.6%)
- **Critic & Vetoes:** Critic Calls = `0`, Parse Fails = `0`, Local Vetoes = `44` (14.67%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CROSS_VALIDATED` | 241 | 80.3% | 92.95% | 96.27% | 3.76 | 1913 | 36.56s |
| `MARGIN` | 33 | 11.0% | 81.82% | 100.00% | 4.27 | 2373 | 42.41s |
| `KMAX` | 26 | 8.7% | 15.38% | 76.92% | 9.00 | 6192 | 103.56s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `PAL_FIRST` | 167 | 55.7% | 86.83% | 95.81% | 2166 | 41.18s |
| `COT_FIRST` | 133 | 44.3% | 82.71% | 93.98% | 2546 | 45.30s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 85.00% | 85.00% | 95.00% | 4.27 | 2335 | 43.01s |

### Method: `PAL_ONLY`
- **Auxiliary Modules:** PAL Exec Success = `1038/1714` (60.6%), Debug Fixed = `478/1154` (41.4%)
- **Critic & Vetoes:** Critic Calls = `0`, Parse Fails = `0`, Local Vetoes = `0` (0.00%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MARGIN` | 186 | 62.0% | 78.49% | 88.17% | 4.91 | 4387 | 10.26s |
| `KMAX` | 59 | 19.7% | 28.81% | 55.93% | 9.00 | 8370 | 20.80s |
| `PAL_DOMINANT` | 52 | 17.3% | 90.38% | 94.23% | 4.65 | 3908 | 9.25s |
| `NO_ANSWER` | 3 | 1.0% | 0.00% | 0.00% | 9.00 | 9668 | 20.28s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `PAL_ONLY` | 300 | 100.0% | 70.00% | 82.00% | 5140 | 12.26s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 70.00% | 70.00% | 82.00% | 5.71 | 5140 | 12.26s |

### Method: `PAL_POOL12`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `POOL_COMPLETE` | 300 | 100.0% | 85.00% | - | 0.00 | 6457 | 91.33s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 85.00% | 85.00% | - | 0.00 | 6457 | 91.33s |

### Method: `SC_POOL12`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `POOL_COMPLETE` | 300 | 100.0% | 89.67% | - | 0.00 | 5181 | 149.89s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 89.67% | 89.67% | - | 0.00 | 5181 | 149.89s |

### Method: `V5_COT`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MARGIN` | 214 | 71.3% | 98.60% | 98.60% | 4.50 | 3174 | 8.87s |
| `KMAX` | 82 | 27.3% | 48.78% | 70.73% | 9.00 | 7684 | 27.85s |
| `NO_ANSWER` | 4 | 1.3% | 0.00% | 50.00% | 9.00 | 6993 | 18.20s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `COT_ONLY` | 300 | 100.0% | 83.67% | 90.33% | 4458 | 14.18s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math Word Problem | 300 | 83.67% | 83.67% | 90.33% | 5.79 | 4458 | 14.18s |

## 5. Anomaly & Error Audit Summary
- **Total flagged records:** `250` (full details exported to `anomalies_audit.csv`)
| Anomaly Tag | Count |
|---|---:|
| `REGRESSION_VS_BASELINE` | 125 |
| `ORACLE_LOSS` | 108 |
| `TOKEN_ESTIMATED_FALLBACK` | 89 |
| `FALSE_CROSS_VALIDATION` | 34 |
| `EXTRACTION_OR_VOTE_POLLUTION` | 17 |
| `COMPUTE_LATENCY_OUTLIER` | 9 |
