# Experimental Evaluation & Anomaly Audit Report
- **Generated at:** 2026-11-20 22:35:23
- **Input files:** `Data\stage3_batch3_custom40.jsonl`
- **SymPy symbolic equivalence engine:** Enabled

## 1. Data Integrity & Coverage Audit
- **Total valid JSON records loaded:** `200`
- **Malformed JSON lines:** `0`
- **Duplicate `(method, seed, id)` records:** `0`
- **Records missing core schema fields:** `0`
- **Internal field logic mismatches:** `0`

| Method / Seed | Records | Unique IDs | Missing MATH-500 IDs | Sample Missing IDs |
|---|---:|---:|---:|---|
| DIRECT (seed=0) | 40 | 40 | 0 | - |
| DUALREF (seed=0) | 40 | 40 | 0 | - |
| DUALREF_NOCRIT (seed=0) | 40 | 40 | 0 | - |
| PAL@5 (seed=0) | 40 | 40 | 0 | - |
| SC@5 (seed=0) | 40 | 40 | 0 | - |

## 2. Overall Performance, Oracle & Compute Efficiency by Method
| Metric | DIRECT | DUALREF | DUALREF_NOCRIT | PAL@5 | SC@5 |
|---|---:|---:|---:|---:|---:|
| Sample Count (`N`) | 40 | 40 | 40 | 40 | 40 |
| **Raw Accuracy (%)** | **32.50%** (13/40) | **70.00%** (28/40) | **75.00%** (30/40) | **70.00%** (28/40) | **32.50%** (13/40) |
| 95% Wilson CI | [20.08%, 47.98%] | [54.57%, 81.93%] | [59.81%, 85.81%] | [54.57%, 81.93%] | [20.08%, 47.98%] |
| Suspected False Negatives | 0 | 0 | 0 | 0 | 0 |
| **Adjusted Accuracy (%)** | **32.50%** | **70.00%** | **75.00%** | **70.00%** | **32.50%** |
| Oracle Any Correct (%) | - | 85.00% | 87.50% | - | - |
| Oracle Gap (Oracle - Raw) | - | 15.00% | 12.50% | - | - |
| PAL Any Correct (%) | - | 70.00% | 80.00% | - | - |
| Cross-Validated Rate (%) | - | 40.00% | 35.00% | - | - |
| Acc when Cross-Validated | - | 81.25% | 78.57% | - | - |
| Acc when NOT Cross-Validated | 32.50% | 62.50% | 73.08% | 70.00% | 32.50% |
| Mean Samples / Query | - | 5.40 | 5.45 | - | - |
| Mean LLM Calls / Query | 1.05 | 7.83 | 6.38 | 6.12 | 5.67 |
| Mean Total Tokens | 577.9 | 3837.7 | 3275.8 | 2953.3 | 3020.1 |
| Median (P50) Total Tokens | 474.0 | 3060.0 | 2290.0 | 2554.0 | 2873.5 |
| P95 Total Tokens | 1209.1 | 8208.7 | 7492.5 | 4975.9 | 4878.9 |
| Total Tokens (All Queries) | 23,116 | 153,509 | 131,033 | 118,134 | 120,803 |
| Mean Latency (`elapsed_sec`) | 38.73s | 103.13s | 104.21s | 47.56s | 105.53s |
| Median (P50) Latency | 25.67s | 68.45s | 80.45s | 41.49s | 97.89s |
| P95 / Max Latency | 111.84s / 185.53s | 310.91s / 356.31s | 280.34s / 406.76s | 88.68s / 142.32s | 226.49s / 268.41s |
| Total Wall-Clock Time | 25.8 min | 68.8 min | 69.5 min | 31.7 min | 70.4 min |
| Acc% per 1k Tokens | 56.24 | 18.24 | 22.89 | 23.70 | 10.76 |
| Any Truncated (`n_truncated>0`) | 22.50% | 15.00% | 30.00% | 0.00% | 0.00% |
| CoT Truncated (`n_cot_truncated>0`) | 0.00% | 15.00% | 30.00% | 0.00% | 0.00% |
| Loop Guard Abort (%) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## 3. Paired Head-to-Head Comparison (McNemar's Test)
### `DUALREF` vs. Baseline `DIRECT` (Paired N = 40)
- **Wins (`0 -> 1`):** `17` | **Losses (`1 -> 0`):** `2` | **Net Gain:** `+15` (`+37.50%`)
- **Both Correct:** `11` | **Both Wrong:** `10`
- **McNemar's Test:** $\chi^2 = 10.3158$, $p = 0.000729$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 0.00% | 50.00% | +50.00% | 2 | 0 | +2 |
| Calculus | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Calculus / Sequence | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Combinatorics | 9 | 22.22% | 66.67% | +44.44% | 4 | 0 | +4 |
| Geometry | 1 | 100.00% | 0.00% | -100.00% | 0 | 1 | -1 |
| Geometry / Algebra | 1 | 100.00% | 0.00% | -100.00% | 0 | 1 | -1 |
| Linear Algebra | 2 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Logarithms | 2 | 50.00% | 50.00% | +0.00% | 0 | 0 | +0 |
| Number Theory | 10 | 30.00% | 90.00% | +60.00% | 6 | 0 | +6 |
| Polynomials | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Probability | 4 | 25.00% | 25.00% | +0.00% | 0 | 0 | +0 |
| Sequences | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Series | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Trigonometry | 2 | 0.00% | 100.00% | +100.00% | 2 | 0 | +2 |

### `DUALREF_NOCRIT` vs. Baseline `DIRECT` (Paired N = 40)
- **Wins (`0 -> 1`):** `18` | **Losses (`1 -> 0`):** `1` | **Net Gain:** `+17` (`+42.50%`)
- **Both Correct:** `12` | **Both Wrong:** `9`
- **McNemar's Test:** $\chi^2 = 13.4737$, $p = 0.000076$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 0.00% | 75.00% | +75.00% | 3 | 0 | +3 |
| Calculus | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Calculus / Sequence | 1 | 0.00% | 0.00% | +0.00% | 0 | 0 | +0 |
| Combinatorics | 9 | 22.22% | 55.56% | +33.33% | 4 | 1 | +3 |
| Geometry | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Geometry / Algebra | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Linear Algebra | 2 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Logarithms | 2 | 50.00% | 50.00% | +0.00% | 0 | 0 | +0 |
| Number Theory | 10 | 30.00% | 100.00% | +70.00% | 7 | 0 | +7 |
| Polynomials | 1 | 0.00% | 0.00% | +0.00% | 0 | 0 | +0 |
| Probability | 4 | 25.00% | 50.00% | +25.00% | 1 | 0 | +1 |
| Sequences | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Series | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Trigonometry | 2 | 0.00% | 100.00% | +100.00% | 2 | 0 | +2 |

### `PAL@5` vs. Baseline `DIRECT` (Paired N = 40)
- **Wins (`0 -> 1`):** `18` | **Losses (`1 -> 0`):** `3` | **Net Gain:** `+15` (`+37.50%`)
- **Both Correct:** `10` | **Both Wrong:** `9`
- **McNemar's Test:** $\chi^2 = 9.3333$, $p = 0.001490$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 0.00% | 75.00% | +75.00% | 3 | 0 | +3 |
| Calculus | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Calculus / Sequence | 1 | 0.00% | 0.00% | +0.00% | 0 | 0 | +0 |
| Combinatorics | 9 | 22.22% | 55.56% | +33.33% | 4 | 1 | +3 |
| Geometry | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Geometry / Algebra | 1 | 100.00% | 0.00% | -100.00% | 0 | 1 | -1 |
| Linear Algebra | 2 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Logarithms | 2 | 50.00% | 0.00% | -50.00% | 0 | 1 | -1 |
| Number Theory | 10 | 30.00% | 90.00% | +60.00% | 6 | 0 | +6 |
| Polynomials | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Probability | 4 | 25.00% | 50.00% | +25.00% | 1 | 0 | +1 |
| Sequences | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Series | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Trigonometry | 2 | 0.00% | 100.00% | +100.00% | 2 | 0 | +2 |

### `SC@5` vs. Baseline `DIRECT` (Paired N = 40)
- **Wins (`0 -> 1`):** `4` | **Losses (`1 -> 0`):** `4` | **Net Gain:** `+0` (`+0.00%`)
- **Both Correct:** `9` | **Both Wrong:** `23`
- **McNemar's Test:** $\chi^2 = 0.0000$, $p = 1.000000$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 0.00% | 25.00% | +25.00% | 1 | 0 | +1 |
| Calculus | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Calculus / Sequence | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Combinatorics | 9 | 22.22% | 11.11% | -11.11% | 0 | 1 | -1 |
| Geometry | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Geometry / Algebra | 1 | 100.00% | 0.00% | -100.00% | 0 | 1 | -1 |
| Linear Algebra | 2 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Logarithms | 2 | 50.00% | 50.00% | +0.00% | 0 | 0 | +0 |
| Number Theory | 10 | 30.00% | 10.00% | -20.00% | 0 | 2 | -2 |
| Polynomials | 1 | 0.00% | 0.00% | +0.00% | 0 | 0 | +0 |
| Probability | 4 | 25.00% | 25.00% | +0.00% | 0 | 0 | +0 |
| Sequences | 1 | 100.00% | 100.00% | +0.00% | 0 | 0 | +0 |
| Series | 1 | 0.00% | 100.00% | +100.00% | 1 | 0 | +1 |
| Trigonometry | 2 | 0.00% | 50.00% | +50.00% | 1 | 0 | +1 |

## 4. Method Internals: Stop Reasons, Routing & Auxiliary Modules
### Method: `DIRECT`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SINGLE_SAMPLE` | 30 | 75.0% | 43.33% | - | 0.00 | 452 | 24.89s |
| `NO_ANSWER` | 10 | 25.0% | 0.00% | - | 0.00 | 956 | 80.27s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 0.00% | 0.00% | - | 0.00 | 1166 | 106.98s |
| Calculus | 1 | 100.00% | 100.00% | - | 0.00 | 316 | 12.64s |
| Calculus / Sequence | 1 | 0.00% | 0.00% | - | 0.00 | 867 | 76.90s |
| Combinatorics | 9 | 22.22% | 22.22% | - | 0.00 | 573 | 35.13s |
| Geometry | 1 | 100.00% | 100.00% | - | 0.00 | 481 | 28.08s |
| Geometry / Algebra | 1 | 100.00% | 100.00% | - | 0.00 | 691 | 43.10s |
| Linear Algebra | 2 | 100.00% | 100.00% | - | 0.00 | 383 | 20.24s |
| Logarithms | 2 | 50.00% | 50.00% | - | 0.00 | 350 | 8.91s |
| Number Theory | 10 | 30.00% | 30.00% | - | 0.00 | 502 | 37.81s |
| Polynomials | 1 | 0.00% | 0.00% | - | 0.00 | 486 | 26.03s |
| Probability | 4 | 25.00% | 25.00% | - | 0.00 | 674 | 32.15s |
| Sequences | 1 | 100.00% | 100.00% | - | 0.00 | 295 | 10.66s |
| Series | 1 | 0.00% | 0.00% | - | 0.00 | 325 | 12.54s |
| Trigonometry | 2 | 0.00% | 0.00% | - | 0.00 | 324 | 15.13s |

### Method: `DUALREF`
- **Auxiliary Modules:** PAL Exec Success = `103/117` (88.0%), Debug Fixed = `10/24` (41.7%)
- **Critic & Vetoes:** Critic Calls = `60`, Parse Fails = `0`, Local Vetoes = `15` (37.50%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MARGIN` | 18 | 45.0% | 77.78% | 83.33% | 4.67 | 3215 | 81.92s |
| `CROSS_VALIDATED` | 16 | 40.0% | 81.25% | 87.50% | 4.88 | 2895 | 73.62s |
| `KMAX` | 6 | 15.0% | 16.67% | 83.33% | 9.00 | 8218 | 245.48s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `PAL_FIRST` | 29 | 72.5% | 72.41% | 82.76% | 3567 | 91.66s |
| `COT_FIRST` | 11 | 27.5% | 63.64% | 90.91% | 4551 | 133.36s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 50.00% | 50.00% | 75.00% | 7.00 | 6523 | 186.22s |
| Calculus | 1 | 100.00% | 100.00% | 100.00% | 7.00 | 2583 | 56.09s |
| Calculus / Sequence | 1 | 100.00% | 100.00% | 100.00% | 3.00 | 1781 | 36.71s |
| Combinatorics | 9 | 66.67% | 66.67% | 88.89% | 6.11 | 4113 | 111.08s |
| Geometry | 1 | 0.00% | 0.00% | 100.00% | 7.00 | 3552 | 82.12s |
| Geometry / Algebra | 1 | 0.00% | 0.00% | 100.00% | 9.00 | 7170 | 169.41s |
| Linear Algebra | 2 | 100.00% | 100.00% | 100.00% | 6.00 | 3786 | 92.12s |
| Logarithms | 2 | 50.00% | 50.00% | 50.00% | 6.00 | 4456 | 183.58s |
| Number Theory | 10 | 90.00% | 90.00% | 100.00% | 4.20 | 2786 | 74.30s |
| Polynomials | 1 | 100.00% | 100.00% | 100.00% | 3.00 | 2501 | 47.17s |
| Probability | 4 | 25.00% | 25.00% | 25.00% | 6.00 | 5107 | 123.61s |
| Sequences | 1 | 100.00% | 100.00% | 100.00% | 3.00 | 1062 | 18.32s |
| Series | 1 | 100.00% | 100.00% | 100.00% | 3.00 | 2069 | 47.06s |
| Trigonometry | 2 | 100.00% | 100.00% | 100.00% | 4.00 | 2454 | 67.43s |

### Method: `DUALREF_NOCRIT`
- **Auxiliary Modules:** PAL Exec Success = `98/118` (83.1%), Debug Fixed = `7/27` (25.9%)
- **Critic & Vetoes:** Critic Calls = `0`, Parse Fails = `0`, Local Vetoes = `16` (40.00%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MARGIN` | 20 | 50.0% | 85.00% | 95.00% | 5.20 | 3055 | 97.04s |
| `CROSS_VALIDATED` | 14 | 35.0% | 78.57% | 92.86% | 4.29 | 2148 | 63.94s |
| `KMAX` | 6 | 15.0% | 33.33% | 50.00% | 9.00 | 6642 | 222.05s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `PAL_FIRST` | 29 | 72.5% | 68.97% | 86.21% | 3447 | 110.18s |
| `COT_FIRST` | 11 | 27.5% | 90.91% | 90.91% | 2824 | 88.45s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 75.00% | 75.00% | 75.00% | 6.50 | 5065 | 183.37s |
| Calculus | 1 | 100.00% | 100.00% | 100.00% | 5.00 | 1868 | 40.76s |
| Calculus / Sequence | 1 | 0.00% | 0.00% | 100.00% | 9.00 | 4759 | 141.25s |
| Combinatorics | 9 | 55.56% | 55.56% | 77.78% | 5.89 | 3776 | 117.85s |
| Geometry | 1 | 100.00% | 100.00% | 100.00% | 9.00 | 4176 | 106.91s |
| Geometry / Algebra | 1 | 100.00% | 100.00% | 100.00% | 3.00 | 1786 | 67.13s |
| Linear Algebra | 2 | 100.00% | 100.00% | 100.00% | 6.00 | 2464 | 48.51s |
| Logarithms | 2 | 50.00% | 50.00% | 50.00% | 4.00 | 1450 | 29.31s |
| Number Theory | 10 | 100.00% | 100.00% | 100.00% | 4.00 | 2457 | 89.46s |
| Polynomials | 1 | 0.00% | 0.00% | 100.00% | 5.00 | 3998 | 183.46s |
| Probability | 4 | 50.00% | 50.00% | 75.00% | 7.50 | 5174 | 144.60s |
| Sequences | 1 | 100.00% | 100.00% | 100.00% | 3.00 | 1028 | 17.08s |
| Series | 1 | 100.00% | 100.00% | 100.00% | 5.00 | 1842 | 35.32s |
| Trigonometry | 2 | 100.00% | 100.00% | 100.00% | 5.00 | 2120 | 76.78s |

### Method: `PAL@5`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `K_SAMPLES_VOTE` | 40 | 100.0% | 70.00% | - | 0.00 | 2953 | 47.56s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 75.00% | 75.00% | - | 0.00 | 3178 | 58.51s |
| Calculus | 1 | 100.00% | 100.00% | - | 0.00 | 2164 | 29.88s |
| Calculus / Sequence | 1 | 0.00% | 0.00% | - | 0.00 | 2717 | 44.44s |
| Combinatorics | 9 | 55.56% | 55.56% | - | 0.00 | 2604 | 38.72s |
| Geometry | 1 | 100.00% | 100.00% | - | 0.00 | 4235 | 70.36s |
| Geometry / Algebra | 1 | 0.00% | 0.00% | - | 0.00 | 4886 | 87.61s |
| Linear Algebra | 2 | 100.00% | 100.00% | - | 0.00 | 4524 | 74.27s |
| Logarithms | 2 | 0.00% | 0.00% | - | 0.00 | 1934 | 18.56s |
| Number Theory | 10 | 90.00% | 90.00% | - | 0.00 | 2830 | 43.49s |
| Polynomials | 1 | 100.00% | 100.00% | - | 0.00 | 3442 | 57.38s |
| Probability | 4 | 50.00% | 50.00% | - | 0.00 | 3588 | 68.15s |
| Sequences | 1 | 100.00% | 100.00% | - | 0.00 | 2256 | 34.83s |
| Series | 1 | 100.00% | 100.00% | - | 0.00 | 2548 | 63.57s |
| Trigonometry | 2 | 100.00% | 100.00% | - | 0.00 | 2086 | 19.33s |

### Method: `SC@5`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `K_SAMPLES_VOTE` | 40 | 100.0% | 32.50% | - | 0.00 | 3020 | 105.53s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 4 | 25.00% | 25.00% | - | 0.00 | 4071 | 166.40s |
| Calculus | 1 | 100.00% | 100.00% | - | 0.00 | 1950 | 49.89s |
| Calculus / Sequence | 1 | 100.00% | 100.00% | - | 0.00 | 2830 | 94.00s |
| Combinatorics | 9 | 11.11% | 11.11% | - | 0.00 | 3442 | 113.91s |
| Geometry | 1 | 100.00% | 100.00% | - | 0.00 | 3117 | 83.82s |
| Geometry / Algebra | 1 | 0.00% | 0.00% | - | 0.00 | 3109 | 117.62s |
| Linear Algebra | 2 | 100.00% | 100.00% | - | 0.00 | 1722 | 37.34s |
| Logarithms | 2 | 50.00% | 50.00% | - | 0.00 | 2326 | 81.96s |
| Number Theory | 10 | 10.00% | 10.00% | - | 0.00 | 3006 | 112.70s |
| Polynomials | 1 | 0.00% | 0.00% | - | 0.00 | 2400 | 76.47s |
| Probability | 4 | 25.00% | 25.00% | - | 0.00 | 3202 | 99.03s |
| Sequences | 1 | 100.00% | 100.00% | - | 0.00 | 1500 | 34.76s |
| Series | 1 | 100.00% | 100.00% | - | 0.00 | 1792 | 46.83s |
| Trigonometry | 2 | 50.00% | 50.00% | - | 0.00 | 2940 | 132.75s |

## 5. Anomaly & Error Audit Summary
- **Total flagged records:** `55` (full details exported to `anomalies_audit.csv`)
| Anomaly Tag | Count |
|---|---:|
| `TOKEN_ESTIMATED_FALLBACK` | 35 |
| `EXTRACTION_OR_VOTE_POLLUTION` | 11 |
| `ORACLE_LOSS` | 11 |
| `REGRESSION_VS_BASELINE` | 10 |
| `FALSE_CROSS_VALIDATION` | 6 |
