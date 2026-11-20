# Experimental Evaluation & Anomaly Audit Report
- **Generated at:** 2026-11-19 23:08:12
- **Input files:** `Data\stage3_batch2_math500_with_subject_300.jsonl`
- **SymPy symbolic equivalence engine:** Enabled

## 1. Data Integrity & Coverage Audit
- **Total valid JSON records loaded:** `1800`
- **Malformed JSON lines:** `0`
- **Duplicate `(method, seed, id)` records:** `0`
- **Records missing core schema fields:** `0`
- **Internal field logic mismatches:** `0`

| Method / Seed | Records | Unique IDs | Missing MATH-500 IDs | Sample Missing IDs |
|---|---:|---:|---:|---|
| DUALREF (seed=0) | 300 | 300 | 200 | MATH500_0001, MATH500_0002, MATH500_0005, MATH500_0007, MATH500_0009 |
| DUALREF_NOCRIT (seed=0) | 300 | 300 | 200 | MATH500_0001, MATH500_0002, MATH500_0005, MATH500_0007, MATH500_0009 |
| PAL_ONLY (seed=0) | 300 | 300 | 200 | MATH500_0001, MATH500_0002, MATH500_0005, MATH500_0007, MATH500_0009 |
| PAL_POOL12 (seed=42) | 300 | 300 | 200 | MATH500_0001, MATH500_0002, MATH500_0005, MATH500_0007, MATH500_0009 |
| SC_POOL12 (seed=42) | 300 | 300 | 200 | MATH500_0001, MATH500_0002, MATH500_0005, MATH500_0007, MATH500_0009 |
| V5_COT (seed=0) | 300 | 300 | 200 | MATH500_0001, MATH500_0002, MATH500_0005, MATH500_0007, MATH500_0009 |

## 2. Overall Performance, Oracle & Compute Efficiency by Method
| Metric | DUALREF | DUALREF_NOCRIT | PAL_ONLY | PAL_POOL12 | SC_POOL12 | V5_COT |
|---|---:|---:|---:|---:|---:|---:|
| Sample Count (`N`) | 300 | 300 | 300 | 300 | 300 | 300 |
| **Raw Accuracy (%)** | **50.67%** (152/300) | **49.00%** (147/300) | **44.67%** (134/300) | **51.00%** (153/300) | **48.00%** (144/300) | **44.00%** (132/300) |
| 95% Wilson CI | [45.04%, 56.28%] | [43.39%, 54.63%] | [39.14%, 50.32%] | [45.37%, 56.61%] | [42.41%, 53.64%] | [38.49%, 49.66%] |
| Suspected False Negatives | 4 | 5 | 7 | 5 | 0 | 2 |
| **Adjusted Accuracy (%)** | **52.00%** | **50.67%** | **47.00%** | **52.67%** | **48.00%** | **44.67%** |
| Oracle Any Correct (%) | 66.00% | 64.67% | 57.67% | - | - | 59.67% |
| Oracle Gap (Oracle - Raw) | 15.33% | 15.67% | 13.00% | - | - | 15.67% |
| PAL Any Correct (%) | 51.33% | 49.00% | 57.67% | - | - | 0.00% |
| Cross-Validated Rate (%) | 35.00% | 34.00% | 0.00% | - | - | 0.00% |
| Acc when Cross-Validated | 84.76% | 81.37% | - | - | - | - |
| Acc when NOT Cross-Validated | 32.31% | 32.32% | 44.67% | 51.00% | 48.00% | 44.00% |
| Mean Samples / Query | 6.84 | 6.73 | 5.71 | - | - | 7.39 |
| Mean LLM Calls / Query | 11.41 | 9.83 | 8.53 | 17.59 | 15.23 | 10.40 |
| Mean Total Tokens | 7522.2 | 6399.1 | 5143.5 | 10241.6 | 10928.2 | 7466.1 |
| Median (P50) Total Tokens | 6997.0 | 6156.5 | 4078.0 | 9592.5 | 9910.5 | 7325.0 |
| P95 Total Tokens | 15117.2 | 12739.7 | 12042.3 | 16735.2 | 18508.4 | 14422.5 |
| Total Tokens (All Queries) | 2,256,649 | 1,919,723 | 1,543,047 | 3,072,474 | 3,278,456 | 2,239,836 |
| Mean Latency (`elapsed_sec`) | 41.89s | 38.82s | 21.80s | 22.03s | 48.49s | 58.97s |
| Median (P50) Latency | 33.11s | 30.82s | 13.85s | 18.50s | 39.48s | 45.63s |
| P95 / Max Latency | 103.76s / 158.20s | 100.46s / 149.55s | 62.20s / 111.43s | 47.40s / 73.56s | 106.90s / 151.05s | 153.70s / 194.43s |
| Total Wall-Clock Time | 209.5 min | 194.1 min | 109.0 min | 110.2 min | 242.5 min | 294.9 min |
| Acc% per 1k Tokens | 6.74 | 7.66 | 8.68 | 4.98 | 4.39 | 5.89 |
| Any Truncated (`n_truncated>0`) | 41.00% | 33.33% | 5.33% | 11.00% | 51.00% | 46.00% |
| CoT Truncated (`n_cot_truncated>0`) | 33.00% | 27.67% | 0.00% | 0.00% | 0.00% | 40.00% |
| Loop Guard Abort (%) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## 3. Paired Head-to-Head Comparison (McNemar's Test)
### `DUALREF_NOCRIT` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `20` | **Losses (`1 -> 0`):** `25` | **Net Gain:** `-5` (`-1.67%`)
- **Both Correct:** `127` | **Both Wrong:** `128`
- **McNemar's Test:** $\chi^2 = 0.3556$, $p = 0.550985$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 60.81% | 55.41% | -5.41% | 4 | 8 | -4 |
| Counting & Probability | 23 | 52.17% | 43.48% | -8.70% | 0 | 2 | -2 |
| Geometry | 25 | 36.00% | 44.00% | +8.00% | 4 | 2 | +2 |
| Intermediate Algebra | 58 | 27.59% | 29.31% | +1.72% | 4 | 3 | +1 |
| Number Theory | 37 | 64.86% | 62.16% | -2.70% | 4 | 5 | -1 |
| Prealgebra | 49 | 65.31% | 71.43% | +6.12% | 4 | 1 | +3 |
| Precalculus | 34 | 41.18% | 29.41% | -11.76% | 0 | 4 | -4 |

### `PAL_ONLY` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `22` | **Losses (`1 -> 0`):** `40` | **Net Gain:** `-18` (`-6.00%`)
- **Both Correct:** `112` | **Both Wrong:** `126`
- **McNemar's Test:** $\chi^2 = 4.6613$, $p = 0.030850$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 60.81% | 58.11% | -2.70% | 5 | 7 | -2 |
| Counting & Probability | 23 | 52.17% | 39.13% | -13.04% | 1 | 4 | -3 |
| Geometry | 25 | 36.00% | 32.00% | -4.00% | 3 | 4 | -1 |
| Intermediate Algebra | 58 | 27.59% | 29.31% | +1.72% | 6 | 5 | +1 |
| Number Theory | 37 | 64.86% | 48.65% | -16.22% | 2 | 8 | -6 |
| Prealgebra | 49 | 65.31% | 59.18% | -6.12% | 4 | 7 | -3 |
| Precalculus | 34 | 41.18% | 29.41% | -11.76% | 1 | 5 | -4 |

### `PAL_POOL12` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `28` | **Losses (`1 -> 0`):** `27` | **Net Gain:** `+1` (`+0.33%`)
- **Both Correct:** `125` | **Both Wrong:** `120`
- **McNemar's Test:** $\chi^2 = 0.0000$, $p = 1.000000$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 60.81% | 66.22% | +5.41% | 10 | 6 | +4 |
| Counting & Probability | 23 | 52.17% | 47.83% | -4.35% | 0 | 1 | -1 |
| Geometry | 25 | 36.00% | 52.00% | +16.00% | 6 | 2 | +4 |
| Intermediate Algebra | 58 | 27.59% | 31.03% | +3.45% | 5 | 3 | +2 |
| Number Theory | 37 | 64.86% | 56.76% | -8.11% | 1 | 4 | -3 |
| Prealgebra | 49 | 65.31% | 63.27% | -2.04% | 5 | 6 | -1 |
| Precalculus | 34 | 41.18% | 29.41% | -11.76% | 1 | 5 | -4 |

### `SC_POOL12` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `33` | **Losses (`1 -> 0`):** `41` | **Net Gain:** `-8` (`-2.67%`)
- **Both Correct:** `111` | **Both Wrong:** `115`
- **McNemar's Test:** $\chi^2 = 0.6622$, $p = 0.415798$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 60.81% | 62.16% | +1.35% | 10 | 9 | +1 |
| Counting & Probability | 23 | 52.17% | 30.43% | -21.74% | 0 | 5 | -5 |
| Geometry | 25 | 36.00% | 44.00% | +8.00% | 4 | 2 | +2 |
| Intermediate Algebra | 58 | 27.59% | 25.86% | -1.72% | 4 | 5 | -1 |
| Number Theory | 37 | 64.86% | 54.05% | -10.81% | 6 | 10 | -4 |
| Prealgebra | 49 | 65.31% | 71.43% | +6.12% | 7 | 4 | +3 |
| Precalculus | 34 | 41.18% | 29.41% | -11.76% | 2 | 6 | -4 |

### `V5_COT` vs. Baseline `DUALREF` (Paired N = 300)
- **Wins (`0 -> 1`):** `33` | **Losses (`1 -> 0`):** `53` | **Net Gain:** `-20` (`-6.67%`)
- **Both Correct:** `99` | **Both Wrong:** `115`
- **McNemar's Test:** $\chi^2 = 4.1977$, $p = 0.040479$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 60.81% | 62.16% | +1.35% | 9 | 8 | +1 |
| Counting & Probability | 23 | 52.17% | 26.09% | -26.09% | 0 | 6 | -6 |
| Geometry | 25 | 36.00% | 40.00% | +4.00% | 4 | 3 | +1 |
| Intermediate Algebra | 58 | 27.59% | 22.41% | -5.17% | 5 | 8 | -3 |
| Number Theory | 37 | 64.86% | 37.84% | -27.03% | 5 | 15 | -10 |
| Prealgebra | 49 | 65.31% | 63.27% | -2.04% | 6 | 7 | -1 |
| Precalculus | 34 | 41.18% | 35.29% | -5.88% | 4 | 6 | -2 |

## 4. Method Internals: Stop Reasons, Routing & Auxiliary Modules
### Method: `DUALREF`
- **Auxiliary Modules:** PAL Exec Success = `675/972` (69.4%), Debug Fixed = `228/525` (43.4%)
- **Critic & Vetoes:** Critic Calls = `411`, Parse Fails = `0`, Local Vetoes = `154` (51.33%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `KMAX` | 113 | 37.7% | 12.39% | 38.94% | 9.00 | 11341 | 67.54s |
| `CROSS_VALIDATED` | 105 | 35.0% | 84.76% | 91.43% | 5.25 | 4478 | 21.71s |
| `MARGIN` | 81 | 27.0% | 60.49% | 71.60% | 5.86 | 6017 | 31.74s |
| `NO_ANSWER` | 1 | 0.3% | 0.00% | 0.00% | 9.00 | 17556 | 84.35s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `COT_FIRST` | 204 | 68.0% | 48.53% | 62.75% | 7913 | 45.76s |
| `PAL_FIRST` | 96 | 32.0% | 55.21% | 72.92% | 6692 | 33.68s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 60.81% | 60.81% | 72.97% | 6.14 | 5930 | 29.53s |
| Counting & Probability | 23 | 52.17% | 52.17% | 65.22% | 6.65 | 6680 | 29.98s |
| Geometry | 25 | 36.00% | 36.00% | 56.00% | 7.40 | 8463 | 39.86s |
| Intermediate Algebra | 58 | 27.59% | 27.59% | 50.00% | 8.21 | 10839 | 74.62s |
| Number Theory | 37 | 64.86% | 75.68% | 78.38% | 6.57 | 5981 | 36.75s |
| Prealgebra | 49 | 65.31% | 65.31% | 81.63% | 5.86 | 5512 | 24.20s |
| Precalculus | 34 | 41.18% | 41.18% | 50.00% | 7.47 | 9781 | 53.59s |

### Method: `DUALREF_NOCRIT`
- **Auxiliary Modules:** PAL Exec Success = `671/955` (70.3%), Debug Fixed = `216/500` (43.2%)
- **Critic & Vetoes:** Critic Calls = `0`, Parse Fails = `0`, Local Vetoes = `165` (55.00%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `KMAX` | 108 | 36.0% | 11.11% | 40.74% | 9.00 | 9808 | 64.12s |
| `CROSS_VALIDATED` | 102 | 34.0% | 81.37% | 87.25% | 4.98 | 3714 | 18.16s |
| `MARGIN` | 87 | 29.0% | 59.77% | 70.11% | 5.87 | 5215 | 31.22s |
| `NO_ANSWER` | 2 | 0.7% | 0.00% | 0.00% | 9.00 | 11326 | 60.70s |
| `PAL_DOMINANT` | 1 | 0.3% | 0.00% | 0.00% | 9.00 | 5328 | 30.57s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `COT_FIRST` | 204 | 68.0% | 47.55% | 61.76% | 6446 | 41.77s |
| `PAL_FIRST` | 96 | 32.0% | 52.08% | 70.83% | 6299 | 32.54s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 55.41% | 55.41% | 70.27% | 5.89 | 4834 | 27.34s |
| Counting & Probability | 23 | 43.48% | 43.48% | 60.87% | 6.83 | 6551 | 31.46s |
| Geometry | 25 | 44.00% | 44.00% | 52.00% | 7.00 | 7007 | 38.50s |
| Intermediate Algebra | 58 | 29.31% | 29.31% | 44.83% | 7.62 | 8373 | 64.62s |
| Number Theory | 37 | 62.16% | 72.97% | 78.38% | 6.51 | 5300 | 32.17s |
| Prealgebra | 49 | 71.43% | 71.43% | 83.67% | 5.98 | 5021 | 20.78s |
| Precalculus | 34 | 29.41% | 32.35% | 55.88% | 8.06 | 9070 | 58.22s |

### Method: `PAL_ONLY`
- **Auxiliary Modules:** PAL Exec Success = `1214/1712` (70.9%), Debug Fixed = `348/846` (41.1%)
- **Critic & Vetoes:** Critic Calls = `0`, Parse Fails = `0`, Local Vetoes = `0` (0.00%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MARGIN` | 171 | 57.0% | 50.29% | 60.82% | 5.02 | 4256 | 17.25s |
| `KMAX` | 63 | 21.0% | 12.70% | 36.51% | 9.00 | 9379 | 42.86s |
| `PAL_DOMINANT` | 62 | 20.7% | 64.52% | 74.19% | 4.03 | 2836 | 11.16s |
| `NO_ANSWER` | 4 | 1.3% | 0.00% | 0.00% | 9.00 | 12158 | 49.42s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `PAL_ONLY` | 300 | 100.0% | 44.67% | 57.67% | 5143 | 21.80s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 58.11% | 58.11% | 75.68% | 5.19 | 4181 | 15.97s |
| Counting & Probability | 23 | 39.13% | 39.13% | 43.48% | 5.87 | 5118 | 19.56s |
| Geometry | 25 | 32.00% | 32.00% | 52.00% | 6.36 | 6491 | 24.00s |
| Intermediate Algebra | 58 | 29.31% | 29.31% | 39.66% | 6.45 | 6178 | 33.72s |
| Number Theory | 37 | 48.65% | 64.86% | 59.46% | 5.11 | 4010 | 18.23s |
| Prealgebra | 49 | 59.18% | 59.18% | 73.47% | 4.84 | 3880 | 12.34s |
| Precalculus | 34 | 29.41% | 32.35% | 38.24% | 6.88 | 7554 | 31.55s |

### Method: `PAL_POOL12`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `POOL_COMPLETE` | 296 | 98.7% | 51.69% | - | 0.00 | 10160 | 21.85s |
| `NO_ANSWER` | 4 | 1.3% | 0.00% | - | 0.00 | 16288 | 35.62s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 66.22% | 66.22% | - | 0.00 | 9070 | 17.75s |
| Counting & Probability | 23 | 47.83% | 47.83% | - | 0.00 | 9735 | 19.87s |
| Geometry | 25 | 52.00% | 52.00% | - | 0.00 | 11891 | 25.25s |
| Intermediate Algebra | 58 | 31.03% | 31.03% | - | 0.00 | 10733 | 28.76s |
| Number Theory | 37 | 56.76% | 70.27% | - | 0.00 | 9376 | 21.49s |
| Prealgebra | 49 | 63.27% | 63.27% | - | 0.00 | 9820 | 16.95s |
| Precalculus | 34 | 29.41% | 29.41% | - | 0.00 | 12633 | 26.89s |

### Method: `SC_POOL12`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `POOL_COMPLETE` | 300 | 100.0% | 48.00% | - | 0.00 | 10928 | 48.49s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 62.16% | 62.16% | - | 0.00 | 9151 | 37.20s |
| Counting & Probability | 23 | 30.43% | 30.43% | - | 0.00 | 10308 | 37.26s |
| Geometry | 25 | 44.00% | 44.00% | - | 0.00 | 11842 | 48.57s |
| Intermediate Algebra | 58 | 25.86% | 25.86% | - | 0.00 | 14542 | 77.96s |
| Number Theory | 37 | 54.05% | 54.05% | - | 0.00 | 9488 | 42.93s |
| Prealgebra | 49 | 71.43% | 71.43% | - | 0.00 | 8507 | 28.23s |
| Precalculus | 34 | 29.41% | 29.41% | - | 0.00 | 13434 | 65.60s |

### Method: `V5_COT`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `KMAX` | 174 | 58.0% | 20.69% | 43.10% | 9.00 | 10125 | 85.04s |
| `MARGIN` | 116 | 38.7% | 82.76% | 87.07% | 4.83 | 3510 | 21.81s |
| `NO_ANSWER` | 10 | 3.3% | 0.00% | 30.00% | 9.00 | 7094 | 36.45s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `COT_ONLY` | 300 | 100.0% | 44.00% | 59.67% | 7466 | 58.97s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 74 | 62.16% | 62.16% | 75.68% | 6.81 | 5827 | 41.38s |
| Counting & Probability | 23 | 26.09% | 30.43% | 56.52% | 7.52 | 7800 | 53.33s |
| Geometry | 25 | 40.00% | 40.00% | 56.00% | 7.96 | 8026 | 58.15s |
| Intermediate Algebra | 58 | 22.41% | 22.41% | 31.03% | 8.07 | 10151 | 97.49s |
| Number Theory | 37 | 37.84% | 37.84% | 54.05% | 7.92 | 6909 | 53.35s |
| Prealgebra | 49 | 63.27% | 63.27% | 89.80% | 6.31 | 5077 | 26.91s |
| Precalculus | 34 | 35.29% | 38.24% | 41.18% | 7.94 | 9867 | 88.30s |

## 5. Anomaly & Error Audit Summary
- **Total flagged records:** `665` (full details exported to `anomalies_audit.csv`)
| Anomaly Tag | Count |
|---|---:|
| `TOKEN_ESTIMATED_FALLBACK` | 432 |
| `ORACLE_LOSS` | 179 |
| `REGRESSION_VS_BASELINE` | 118 |
| `EXTRACTION_OR_VOTE_POLLUTION` | 73 |
| `FALSE_CROSS_VALIDATION` | 35 |
| `FALSE_NEGATIVE_SUSPECT` | 23 |
| `COMPUTE_LATENCY_OUTLIER` | 1 |

### Sample Suspected False Negatives (Grader Normalization Bugs)
| Method | ID | Category | Ground Truth | Predicted Answer | Reason |
|---|---|---|---|---|---|
| `PAL_POOL12` | `MATH500_0030` | Number Theory | `52_8` | `52` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_8)) |
| `PAL_POOL12` | `MATH500_0137` | Number Theory | `204_5` | `204` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_5)) |
| `PAL_POOL12` | `MATH500_0176` | Number Theory | `.0000672` | `21/312500` | Grader mismatch (NUMERIC_FRACTION_EQUIV) |
| `PAL_POOL12` | `MATH500_0217` | Number Theory | `11,\! 111,\! 111,\! 100` | `11111111100` | Grader mismatch (NUMERIC_FRACTION_EQUIV) |
| `PAL_POOL12` | `MATH500_0338` | Number Theory | `4343_6` | `4343` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_6)) |
| `DUALREF` | `MATH500_0030` | Number Theory | `52_8` | `52` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_8)); Oracle loss caused by grader formatting mismatch |
| `DUALREF` | `MATH500_0070` | Number Theory | `40_9` | `40` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_9)); Oracle loss caused by grader formatting mismatch |
| `DUALREF` | `MATH500_0176` | Number Theory | `.0000672` | `21/312500` | Grader mismatch (NUMERIC_FRACTION_EQUIV) |
| `DUALREF` | `MATH500_0338` | Number Theory | `4343_6` | `4343` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_6)) |
| `DUALREF_NOCRIT` | `MATH500_0028` | Precalculus | `-2 + 7i` | `(-2+7j)` | Grader mismatch (COMPLEX_J_VS_I_MATCH); Oracle loss caused by grader formatting mismatch |
| `DUALREF_NOCRIT` | `MATH500_0030` | Number Theory | `52_8` | `52` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_8)); Oracle loss caused by grader formatting mismatch |
| `DUALREF_NOCRIT` | `MATH500_0070` | Number Theory | `40_9` | `40` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_9)); Oracle loss caused by grader formatting mismatch |
| `DUALREF_NOCRIT` | `MATH500_0176` | Number Theory | `.0000672` | `21/312500` | Grader mismatch (NUMERIC_FRACTION_EQUIV) |
| `DUALREF_NOCRIT` | `MATH500_0217` | Number Theory | `11,\! 111,\! 111,\! 100` | `11111111100` | Grader mismatch (NUMERIC_FRACTION_EQUIV); token_estimated_calls=2 |
| `V5_COT` | `MATH500_0198` | Counting & Probability | `10,\!080` | `\frac{10080}{1}` | Grader mismatch (NUMERIC_FRACTION_EQUIV) |
