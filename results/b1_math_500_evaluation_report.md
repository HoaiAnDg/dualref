# Experimental Evaluation & Anomaly Audit Report
- **Generated at:** 2026-11-18 22:31:06
- **Input files:** `stage3_batch1_math500_with_subject.jsonl`
- **SymPy symbolic equivalence engine:** Enabled

## 1. Data Integrity & Coverage Audit
- **Total valid JSON records loaded:** `1000`
- **Malformed JSON lines:** `0`
- **Duplicate `(method, seed, id)` records:** `0`
- **Records missing core schema fields:** `0`
- **Internal field logic mismatches:** `0`

| Method / Seed | Records | Unique IDs | Missing MATH-500 IDs | Sample Missing IDs |
|---|---:|---:|---:|---|
| DIRECT (seed=0) | 500 | 500 | 0 | - |
| DUALREF (seed=0) | 500 | 500 | 0 | - |

## 2. Overall Performance, Oracle & Compute Efficiency by Method
| Metric | DIRECT | DUALREF |
|---|---:|---:|
| Sample Count (`N`) | 500 | 500 |
| **Raw Accuracy (%)** | **27.60%** (138/500) | **50.60%** (253/500) |
| 95% Wilson CI | [23.86%, 31.68%] | [46.23%, 54.96%] |
| Suspected False Negatives | 1 | 10 |
| **Adjusted Accuracy (%)** | **27.80%** | **52.60%** |
| Oracle Any Correct (%) | - | 66.00% |
| Oracle Gap (Oracle - Raw) | - | 15.40% |
| PAL Any Correct (%) | - | 52.80% |
| Cross-Validated Rate (%) | - | 32.40% |
| Acc when Cross-Validated | - | 85.19% |
| Acc when NOT Cross-Validated | 27.60% | 34.02% |
| Mean Samples / Query | - | 6.81 |
| Mean LLM Calls / Query | 1.23 | 11.42 |
| Mean Total Tokens | 844.2 | 7418.3 |
| Median (P50) Total Tokens | 720.5 | 6875.0 |
| P95 Total Tokens | 1747.7 | 14698.8 |
| Total Tokens (All Queries) | 422,114 | 3,709,139 |
| Mean Latency (`elapsed_sec`) | 7.21s | 40.39s |
| Median (P50) Latency | 4.73s | 31.48s |
| P95 / Max Latency | 21.17s / 30.63s | 100.84s / 168.45s |
| Total Wall-Clock Time | 60.1 min | 336.6 min |
| Acc% per 1k Tokens | 32.69 | 6.82 |
| Any Truncated (`n_truncated>0`) | 16.20% | 34.80% |
| CoT Truncated (`n_cot_truncated>0`) | 0.00% | 28.60% |
| Loop Guard Abort (%) | 0.00% | 0.00% |

## 3. Paired Head-to-Head Comparison (McNemar's Test)
### `DUALREF` vs. Baseline `DIRECT` (Paired N = 500)
- **Wins (`0 -> 1`):** `150` | **Losses (`1 -> 0`):** `35` | **Net Gain:** `+115` (`+23.00%`)
- **Both Correct:** `103` | **Both Wrong:** `212`
- **McNemar's Test:** $\chi^2 = 70.2486$, $p = 0.000000$

| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 124 | 40.32% | 69.35% | +29.03% | 43 | 7 | +36 |
| Counting & Probability | 38 | 26.32% | 44.74% | +18.42% | 13 | 6 | +7 |
| Geometry | 41 | 24.39% | 43.90% | +19.51% | 11 | 3 | +8 |
| Intermediate Algebra | 97 | 11.34% | 29.90% | +18.56% | 22 | 4 | +18 |
| Number Theory | 62 | 25.81% | 56.45% | +30.65% | 26 | 7 | +19 |
| Prealgebra | 82 | 41.46% | 63.41% | +21.95% | 24 | 6 | +18 |
| Precalculus | 56 | 12.50% | 28.57% | +16.07% | 11 | 2 | +9 |

## 4. Method Internals: Stop Reasons, Routing & Auxiliary Modules
### Method: `DIRECT`

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SINGLE_SAMPLE` | 312 | 62.4% | 44.23% | - | 0.00 | 663 | 5.57s |
| `NO_ANSWER` | 188 | 37.6% | 0.00% | - | 0.00 | 1146 | 9.93s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 124 | 40.32% | 40.32% | - | 0.00 | 717 | 5.47s |
| Counting & Probability | 38 | 26.32% | 26.32% | - | 0.00 | 813 | 6.24s |
| Geometry | 41 | 24.39% | 24.39% | - | 0.00 | 819 | 6.78s |
| Intermediate Algebra | 97 | 11.34% | 11.34% | - | 0.00 | 1058 | 10.94s |
| Number Theory | 62 | 25.81% | 25.81% | - | 0.00 | 753 | 6.48s |
| Prealgebra | 82 | 41.46% | 41.46% | - | 0.00 | 666 | 4.26s |
| Precalculus | 56 | 12.50% | 14.29% | - | 0.00 | 1158 | 10.69s |

### Method: `DUALREF`
- **Auxiliary Modules:** PAL Exec Success = `1137/1610` (70.6%), Debug Fixed = `415/888` (46.7%)
- **Critic & Vetoes:** Critic Calls = `688`, Parse Fails = `0`, Local Vetoes = `259` (51.80%)

| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `KMAX` | 199 | 39.8% | 14.07% | 42.21% | 9.00 | 11380 | 66.46s |
| `CROSS_VALIDATED` | 162 | 32.4% | 85.19% | 88.89% | 4.88 | 4110 | 18.65s |
| `MARGIN` | 138 | 27.6% | 63.04% | 73.91% | 5.90 | 5588 | 28.33s |
| `PAL_DOMINANT` | 1 | 0.2% | 0.00% | 0.00% | 9.00 | 7481 | 38.96s |

| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|
| `COT_FIRST` | 342 | 68.4% | 49.12% | 63.74% | 7884 | 44.35s |
| `PAL_FIRST` | 158 | 31.6% | 53.80% | 70.89% | 6411 | 31.82s |

| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algebra | 124 | 69.35% | 69.35% | 79.84% | 6.32 | 5881 | 29.47s |
| Counting & Probability | 38 | 44.74% | 44.74% | 63.16% | 6.37 | 6784 | 32.86s |
| Geometry | 41 | 43.90% | 43.90% | 63.41% | 6.95 | 7544 | 37.01s |
| Intermediate Algebra | 97 | 29.90% | 29.90% | 43.30% | 7.70 | 10113 | 67.27s |
| Number Theory | 62 | 56.45% | 67.74% | 75.81% | 6.42 | 5504 | 29.51s |
| Prealgebra | 82 | 63.41% | 63.41% | 82.93% | 6.10 | 5844 | 24.03s |
| Precalculus | 56 | 28.57% | 33.93% | 42.86% | 8.00 | 10918 | 61.60s |

## 5. Anomaly & Error Audit Summary
- **Total flagged records:** `410` (full details exported to `anomalies_audit.csv`)
| Anomaly Tag | Count |
|---|---:|
| `EXTRACTION_OR_VOTE_POLLUTION` | 203 |
| `TOKEN_ESTIMATED_FALLBACK` | 199 |
| `ORACLE_LOSS` | 77 |
| `REGRESSION_VS_BASELINE` | 35 |
| `FALSE_CROSS_VALIDATION` | 24 |
| `FALSE_NEGATIVE_SUSPECT` | 11 |

### Sample Suspected False Negatives (Grader Normalization Bugs)
| Method | ID | Category | Ground Truth | Predicted Answer | Reason |
|---|---|---|---|---|---|
| `DIRECT` | `MATH500_0298` | Precalculus | `\begin{pmatrix} -18 \\ -49 \\ 96 \end{pmatrix}` | `\begin{pmatrix}-18\\-49\\96\end{pmatrix}` | Grader mismatch (LATEX_WHITESPACE_WRAPPER_MATCH) |
| `DUALREF` | `MATH500_0028` | Precalculus | `-2 + 7i` | `(-2+7j)` | Grader mismatch (COMPLEX_J_VS_I_MATCH); Oracle loss caused by grader formatting mismatch |
| `DUALREF` | `MATH500_0030` | Number Theory | `52_8` | `52` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_8)); Baseline solved correctly, this method failed |
| `DUALREF` | `MATH500_0070` | Number Theory | `40_9` | `40` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_9)); Oracle loss caused by grader formatting mismatch; Baseline solved correctly, this method failed |
| `DUALREF` | `MATH500_0071` | Number Theory | `2516_8` | `2516` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_8)) |
| `DUALREF` | `MATH500_0074` | Precalculus | `\cot x` | `1/tan(x)` | Grader mismatch (TRIG_OR_ALGEBRAIC_CANONICAL_MATCH) |
| `DUALREF` | `MATH500_0099` | Precalculus | `\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}` | `\begin{pmatrix}-\frac{1}{3}\\ \frac{2}{3}\\ \frac{5}{3}\end{pmatrix}` | Grader mismatch (MATRIX_VECTOR_FORMAT_MATCH) |
| `DUALREF` | `MATH500_0127` | Number Theory | `4210_{5}` | `4210` | Grader mismatch (MISSING_BASE_SUBSCRIPT (_5)) |
| `DUALREF` | `MATH500_0176` | Number Theory | `.0000672` | `21/312500` | Grader mismatch (NUMERIC_FRACTION_EQUIV) |
| `DUALREF` | `MATH500_0217` | Number Theory | `11,\! 111,\! 111,\! 100` | `11111111100` | Grader mismatch (NUMERIC_FRACTION_EQUIV); token_estimated_calls=1 |
| `DUALREF` | `MATH500_0433` | Number Theory | `.35625` | `0.35625` | Grader mismatch (NUMERIC_FRACTION_EQUIV); Cross-validated consensus reached on incorrect answer |
