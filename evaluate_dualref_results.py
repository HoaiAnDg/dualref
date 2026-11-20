#!/usr/bin/env python3
"""
Comprehensive Experimental Evaluation & Anomaly Audit Script for JSONL Outputs
(Designed for MATH-500 / DualRef / Direct / CoT / PAL experimental runs)

Usage:
    # Automatically outputs to: eval_report/<jsonl_filename>/
    python evaluate_dualref_results.py path/to/results.jsonl

    # Outputs to: custom_root/<jsonl_filename>/
    python evaluate_dualref_results.py path/to/results.jsonl --out-dir custom_root

    # Merge multiple JSONL files into: eval_report/run1__run2/
    python evaluate_dualref_results.py run1.jsonl run2.jsonl --baseline DIRECT
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False


# ==============================================================================
# 1. UTILITY & STATISTICAL HELPERS
# ==============================================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return float(int(val)) if isinstance(val, bool) else default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, bool):
        return int(val)
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0, "sum": 0.0, "mean": 0.0, "std": 0.0,
            "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0
        }
    n = len(values)
    mean_v = sum(values) / n
    std_v = statistics.stdev(values) if n > 1 else 0.0
    return {
        "count": n,
        "sum": sum(values),
        "mean": mean_v,
        "std": std_v,
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = p + (z * z) / (2.0 * total)
    adj = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
    low = max(0.0, (center - adj) / denom)
    high = min(1.0, (center + adj) / denom)
    return (low, high)


def mcnemar_test(wins: int, losses: int) -> Dict[str, float]:
    """McNemar's test with continuity correction and exact binomial p-value for small b+c."""
    b, c = wins, losses
    n = b + c
    if n == 0:
        return {"b_wins": b, "c_losses": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = ((abs(b - c) - 1.0) ** 2) / n if abs(b - c) >= 1 else 0.0
    if n <= 25:
        k = min(b, c)
        prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        p_val = min(1.0, 2.0 * prob)
    else:
        # Survival function of chi2(df=1) equals erfc(sqrt(chi2 / 2))
        p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b_wins": b, "c_losses": c, "chi2": chi2, "p_value": p_val}


def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip("._") or "unnamed_run"


def resolve_output_dir(
    base_out_dir: str,
    raw_inputs: List[str],
    jsonl_paths: List[str],
    no_subdir: bool = False,
    add_timestamp: bool = False,
) -> Path:
    """
    Build a dedicated output folder named after the input JSONL file(s)
    under `base_out_dir` so evaluations of different JSONL files never overwrite each other.
    """
    base = Path(base_out_dir)
    if no_subdir:
        out_dir = base
    else:
        if len(jsonl_paths) == 1:
            run_slug = sanitize_folder_name(Path(jsonl_paths[0]).stem)
        elif len(raw_inputs) == 1 and Path(raw_inputs[0]).is_dir():
            dir_slug = sanitize_folder_name(Path(raw_inputs[0]).resolve().name)
            run_slug = f"{dir_slug}__merged_{len(jsonl_paths)}files"
        else:
            stems = [sanitize_folder_name(Path(p).stem) for p in jsonl_paths]
            if len(stems) <= 3:
                run_slug = "__".join(stems)
            else:
                run_slug = f"{stems[0]}__{stems[1]}__plus_{len(stems) - 2}_more"

        # Avoid duplicating folder name if user already passed `--out-dir eval_report/<run_slug>`
        if base.name == run_slug:
            out_dir = base
        else:
            out_dir = base / run_slug

    if add_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir.parent / f"{out_dir.name}_{ts}"

    return out_dir


# ==============================================================================
# 2. MATHEMATICAL NORMALIZATION & FALSE-NEGATIVE DETECTOR
# ==============================================================================

def strip_latex_wrappers(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Remove outer $ or \( \)
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    # Unwrap \boxed{...}
    m = re.match(r"^\\boxed\{(.*)\}$", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    # Normalize fraction commands and sizing commands
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "").replace(r"\:", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{\\circ\}|\^\\circ|°", "", s)
    return s.strip()


def normalize_simple_fraction_or_number(s: str) -> Optional[Fraction]:
    s = strip_latex_wrappers(s)
    # Remove spaces and outer parentheses if simple scalar
    s_clean = s.replace(" ", "")
    if s_clean.startswith("(") and s_clean.endswith(")") and "," not in s_clean:
        s_clean = s_clean[1:-1]
    # Remove thousands commas like 1,234 -> 1234 (only if single number)
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", s_clean):
        s_clean = s_clean.replace(",", "")
    # Match \frac{a}{b} or -\frac{a}{b}
    m = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", s_clean)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        num = int(m.group(2))
        den = int(m.group(3))
        if den != 0:
            return Fraction(sign * num, den)
    # Match a/b
    m2 = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", s_clean)
    if m2:
        num = int(m2.group(1))
        den = int(m2.group(2))
        if den != 0:
            return Fraction(num, den)
    # Match integer or decimal
    if re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", s_clean):
        try:
            return Fraction(s_clean)
        except Exception:
            return None
    return None


def parse_latex_matrix(s: str) -> Optional[List[List[str]]]:
    s = strip_latex_wrappers(s)
    m = re.fullmatch(
        r"\\begin\{(?:p|b|v|V|small)?matrix\}(.*?)\\end\{(?:p|b|v|V|small)?matrix\}",
        s,
        flags=re.DOTALL,
    )
    if not m:
        return None
    body = m.group(1).strip()
    rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
    matrix = []
    for r in rows:
        cols = [c.strip() for c in r.split("&")]
        matrix.append(cols)
    return matrix


def parse_complex_number(s: str) -> Optional[complex]:
    s = strip_latex_wrappers(s).replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    # Convert math 'i' to python 'j'
    s_py = re.sub(r"(?<![a-zA-Z])i(?![a-zA-Z])", "j", s)
    # Handle cases like 'j' or '-j' or '+j'
    s_py = re.sub(r"(^|[+-])j\b", r"\g<1>1j", s_py)
    if "j" not in s_py:
        return None
    try:
        return complex(s_py)
    except Exception:
        return None


def canonicalize_trig_and_algebra(s: str) -> str:
    s = strip_latex_wrappers(s).lower()
    s = re.sub(r"\s+", "", s)
    # Strip outer parens
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    # Normalize LaTeX functions to plain names
    for fn in ["sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "ln", "log", "sqrt"]:
        s = s.replace("\\" + fn, fn)
    s = s.replace("\\pi", "pi").replace("\\theta", "theta").replace("\\infty", "inf")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # Reciprocal trig equivalences
    s = re.sub(r"1/tan\(([^()]+)\)", r"cot(\1)", s)
    s = re.sub(r"1/sin\(([^()]+)\)", r"csc(\1)", s)
    s = re.sub(r"1/cos\(([^()]+)\)", r"sec(\1)", s)
    s = re.sub(r"1/cot\(([^()]+)\)", r"tan(\1)", s)
    # Normalize cotx vs cot(x)
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc)([a-z])\b", r"\1(\2)", s)
    # Normalize \frac{a}{b} -> (a)/(b)
    s = re.sub(r"frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    return s


def check_false_negative_suspect(pred: Any, gt: Any) -> Tuple[bool, str]:
    """
    Checks if a record marked correct==0 actually has a predicted_answer
    mathematically equivalent to ground_truth (formatting/notation mismatch).
    """
    if pred is None or gt is None:
        return False, ""
    p_str = str(pred).strip()
    g_str = str(gt).strip()
    if not p_str or not g_str:
        return False, ""

    # 1. Exact after stripping LaTeX wrappers & whitespace
    p_clean = re.sub(r"\s+", "", strip_latex_wrappers(p_str))
    g_clean = re.sub(r"\s+", "", strip_latex_wrappers(g_str))
    if p_clean and p_clean == g_clean:
        return True, "LATEX_WHITESPACE_WRAPPER_MATCH"

    # 2. Exact numeric / fraction equivalence (e.g., -\frac{1}{3} vs -1/3, 0.5 vs 1/2)
    p_frac = normalize_simple_fraction_or_number(p_str)
    g_frac = normalize_simple_fraction_or_number(g_str)
    if p_frac is not None and g_frac is not None and p_frac == g_frac:
        return True, "NUMERIC_FRACTION_EQUIV"

    # 3. Matrix / Vector LaTeX element-wise equivalence (e.g., MATH500_0099)
    p_mat = parse_latex_matrix(p_str)
    g_mat = parse_latex_matrix(g_str)
    if p_mat is not None and g_mat is not None:
        if len(p_mat) == len(g_mat) and all(len(r1) == len(r2) for r1, r2 in zip(p_mat, g_mat)):
            all_match = True
            for r1, r2 in zip(p_mat, g_mat):
                for c1, c2 in zip(r1, r2):
                    f1 = normalize_simple_fraction_or_number(c1)
                    f2 = normalize_simple_fraction_or_number(c2)
                    if f1 is not None and f2 is not None:
                        if f1 != f2:
                            all_match = False
                    elif canonicalize_trig_and_algebra(c1) != canonicalize_trig_and_algebra(c2):
                        all_match = False
            if all_match:
                return True, "MATRIX_VECTOR_FORMAT_MATCH"

    # 4. Complex number notation mismatch: Python 'j' vs math 'i' (e.g., MATH500_0028)
    p_cplx = parse_complex_number(p_str)
    g_cplx = parse_complex_number(g_str)
    if p_cplx is not None and g_cplx is not None:
        if abs(p_cplx.real - g_cplx.real) < 1e-9 and abs(p_cplx.imag - g_cplx.imag) < 1e-9:
            return True, "COMPLEX_J_VS_I_MATCH"

    # 5. Base-subscript stripped by PAL/SymPy (e.g., '52' vs '52_8', '40' vs '40_9')
    m_base = re.fullmatch(r"([0-9A-Za-z]+)_(?:\{?(\d+)\}?)", g_clean)
    if m_base:
        digits_part = m_base.group(1)
        if p_clean == digits_part:
            return True, f"MISSING_BASE_SUBSCRIPT (_{m_base.group(2)})"

    # 6. Canonical algebraic / trigonometric identity match (e.g., 1/tan(x) vs \cot x)
    p_alg = canonicalize_trig_and_algebra(p_str)
    g_alg = canonicalize_trig_and_algebra(g_str)
    if p_alg and p_alg == g_alg:
        return True, "TRIG_OR_ALGEBRAIC_CANONICAL_MATCH"

    # 7. Optional SymPy symbolic equivalence check
    if HAS_SYMPY:
        try:
            e1 = parse_latex(strip_latex_wrappers(p_str))
            e2 = parse_latex(strip_latex_wrappers(g_str))
            if sp.simplify(e1 - e2) == 0:
                return True, "SYMPY_SYMBOLIC_EQUIV"
        except Exception:
            pass

    return False, ""


# ==============================================================================
# 3. VOTE DISTRIBUTION & EXTRACTION POLLUTION INSPECTOR
# ==============================================================================

POLLUTION_PATTERNS = [
    (r"import\s+(?:numpy|sympy|math|scipy|os|sys|re)", "CODE_IMPORT_LEAK"),
    (r"\bdef\s+\w+\s*\(", "CODE_DEF_LEAK"),
    (r"\bprint\s*\(", "CODE_PRINT_LEAK"),
    (r"['\"]name['\"]\s*:", "TOOL_JSON_LEAK"),
    (r"solve_equation|execute_python|python_interpreter", "TOOL_NAME_LEAK"),
    (r"\\boxed\s*\(", "MALFORMED_BOXED_PAREN"),
    (r"(?:^|:)\s*(?:nan|none|null|inf|-inf|error|traceback)\s*$", "NAN_OR_ERROR_CANDIDATE"),
    (r"thefinalansweris|finalansweris|theansweris", "UNSTRIPPED_PROSE_PREFIX"),
]


def inspect_extraction_pollution(rec: Dict[str, Any]) -> List[str]:
    flags = []
    pred = rec.get("predicted_answer")
    if pred is None or str(pred).strip() == "":
        flags.append("EMPTY_OR_NULL_PREDICTION")
    else:
        p_str = str(pred).strip()
        if len(p_str) > 100:
            flags.append(f"OVERLONG_PREDICTION (len={len(p_str)})")
        if p_str.count("{") != p_str.count("}"):
            flags.append("UNBALANCED_BRACES_IN_PRED")
        for pat, label in POLLUTION_PATTERNS:
            if re.search(pat, p_str, flags=re.IGNORECASE):
                flags.append(f"PRED_{label}")

    vote_dist = rec.get("vote_distribution")
    if isinstance(vote_dist, dict):
        for cand in vote_dist.keys():
            c_str = str(cand)
            # Strip prefix like 'T:' or 'P:'
            body = c_str[2:] if len(c_str) >= 2 and c_str[1] == ":" else c_str
            if len(body) > 90:
                flags.append("VOTE_CANDIDATE_OVERLONG")
            for pat, label in POLLUTION_PATTERNS:
                if re.search(pat, body, flags=re.IGNORECASE):
                    flags.append(f"VOTE_{label}")
    return sorted(set(flags))


# ==============================================================================
# 4. DATA LOADING & INTEGRITY CHECKS
# ==============================================================================

def load_jsonl_files(paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = []
    parse_errors = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            parse_errors.append({"file": str(p), "line": 0, "error": "File not found"})
            continue
        with p.open("r", encoding="utf-8-sig") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj["_source_file"] = p.name
                    obj["_source_line"] = line_no
                    records.append(obj)
                except json.JSONDecodeError as e:
                    parse_errors.append({
                        "file": p.name,
                        "line": line_no,
                        "error": f"JSONDecodeError: {e.msg} at col {e.colno}",
                        "snippet": line[:120]
                    })
    return records, parse_errors


def audit_data_integrity(records: List[Dict[str, Any]], parse_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method_seed = defaultdict(list)
    duplicates = []
    seen_keys = set()
    missing_fields_rows = []
    logic_mismatches = []

    required_fields = ["id", "method", "ground_truth", "predicted_answer", "correct"]

    for r in records:
        method = str(r.get("method", "UNKNOWN"))
        seed = r.get("seed", 0)
        qid = str(r.get("id", ""))
        key = (method, seed, qid)

        if key in seen_keys:
            duplicates.append({"method": method, "seed": seed, "id": qid, "line": r.get("_source_line")})
        seen_keys.add(key)
        by_method_seed[(method, seed)].append(r)

        missing = [f for f in required_fields if f not in r]
        if missing:
            missing_fields_rows.append({"id": qid, "method": method, "seed": seed, "missing": missing})

        # Logical invariant checks
        row_issues = []
        if "n_samples" in r and "n_cot" in r and "n_pal" in r:
            ns = safe_int(r["n_samples"])
            nc = safe_int(r["n_cot"])
            np_ = safe_int(r["n_pal"])
            if ns != nc + np_:
                row_issues.append(f"n_samples({ns}) != n_cot({nc}) + n_pal({np_})")

        if "total_tokens" in r and "prompt_tokens" in r and "completion_tokens" in r:
            tt = safe_int(r["total_tokens"])
            pt = safe_int(r["prompt_tokens"])
            ct = safe_int(r["completion_tokens"])
            if tt != pt + ct:
                row_issues.append(f"total_tokens({tt}) != prompt({pt}) + completion({ct})")

        if "oracle_any_correct" in r:
            corr = safe_int(r.get("correct"))
            ora = safe_int(r.get("oracle_any_correct"))
            if corr == 1 and ora == 0:
                row_issues.append("correct==1 but oracle_any_correct==0")

        if "pal_exec_ok" in r and "n_pal" in r:
            if safe_int(r["pal_exec_ok"]) > safe_int(r["n_pal"]):
                row_issues.append(f"pal_exec_ok({r['pal_exec_ok']}) > n_pal({r['n_pal']})")

        if "debug_fixed" in r and "debug_calls" in r:
            if safe_int(r["debug_fixed"]) > safe_int(r["debug_calls"]):
                row_issues.append(f"debug_fixed({r['debug_fixed']}) > debug_calls({r['debug_calls']})")

        if "n_cot_truncated" in r and "n_truncated" in r:
            if safe_int(r["n_cot_truncated"]) > safe_int(r["n_truncated"]):
                row_issues.append(f"n_cot_truncated({r['n_cot_truncated']}) > n_truncated({r['n_truncated']})")

        if row_issues:
            logic_mismatches.append({
                "id": qid, "method": method, "seed": seed, "issues": "; ".join(row_issues)
            })

    # Expected MATH500 ID coverage check per (method, seed)
    expected_ids = {f"MATH500_{i:04d}" for i in range(500)}
    coverage_report = {}
    for (method, seed), grp in sorted(by_method_seed.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        present_ids = {str(x.get("id", "")) for x in grp}
        math500_like = any(x.startswith("MATH500_") for x in present_ids)
        missing_500 = sorted(expected_ids - present_ids) if math500_like else []
        coverage_report[f"{method} (seed={seed})"] = {
            "total_records": len(grp),
            "unique_ids": len(present_ids),
            "missing_math500_count": len(missing_500),
            "missing_math500_sample": missing_500[:15],
        }

    return {
        "total_records_loaded": len(records),
        "json_parse_errors": parse_errors,
        "duplicate_keys_count": len(duplicates),
        "duplicates_sample": duplicates[:20],
        "missing_fields_count": len(missing_fields_rows),
        "logic_mismatches_count": len(logic_mismatches),
        "logic_mismatches": logic_mismatches,
        "coverage_by_method_seed": coverage_report,
    }


# ==============================================================================
# 5. METHOD-LEVEL & CATEGORY-LEVEL AGGREGATION
# ==============================================================================

def analyze_method(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    correct_list = [safe_int(r.get("correct")) for r in records]
    n_correct = sum(correct_list)
    acc = n_correct / n
    ci_low, ci_high = wilson_ci(n_correct, n)

    # False negative scan
    fn_suspects = []
    for r in records:
        if safe_int(r.get("correct")) == 0:
            is_fn, reason = check_false_negative_suspect(r.get("predicted_answer"), r.get("ground_truth"))
            if is_fn:
                fn_suspects.append((r.get("id"), reason))
    adj_correct = n_correct + len(fn_suspects)
    adj_acc = adj_correct / n

    # Oracle & Cross-validation metrics
    has_oracle = any("oracle_any_correct" in r for r in records)
    oracle_acc = sum(safe_int(r.get("oracle_any_correct")) for r in records) / n if has_oracle else None
    oracle_gap = (oracle_acc - acc) if oracle_acc is not None else None

    has_pal_any = any("pal_any_correct" in r for r in records)
    pal_any_acc = sum(safe_int(r.get("pal_any_correct")) for r in records) / n if has_pal_any else None

    has_cv = any("cross_validated" in r for r in records)
    cv_rows = [r for r in records if safe_int(r.get("cross_validated")) == 1]
    non_cv_rows = [r for r in records if safe_int(r.get("cross_validated")) == 0]
    cv_rate = len(cv_rows) / n if has_cv else None
    cv_acc = (sum(safe_int(r.get("correct")) for r in cv_rows) / len(cv_rows)) if cv_rows else None
    non_cv_acc = (sum(safe_int(r.get("correct")) for r in non_cv_rows) / len(non_cv_rows)) if non_cv_rows else None

    # Compute & Latency metrics
    llm_calls = summarize_numeric([safe_float(r.get("llm_calls")) for r in records if "llm_calls" in r])
    elapsed = summarize_numeric([safe_float(r.get("elapsed_sec")) for r in records if "elapsed_sec" in r])
    prompt_tok = summarize_numeric([safe_float(r.get("prompt_tokens")) for r in records if "prompt_tokens" in r])
    comp_tok = summarize_numeric([safe_float(r.get("completion_tokens")) for r in records if "completion_tokens" in r])
    total_tok = summarize_numeric([safe_float(r.get("total_tokens")) for r in records if "total_tokens" in r])
    gen_tok = summarize_numeric([safe_float(r.get("tokens_generator_model")) for r in records if "tokens_generator_model" in r])
    other_tok = summarize_numeric([safe_float(r.get("tokens_other_model")) for r in records if "tokens_other_model" in r])
    n_samples_stat = summarize_numeric([safe_float(r.get("n_samples")) for r in records if "n_samples" in r])

    # Efficiency ratios
    acc_per_1k_tok = (acc * 100.0) / (total_tok["mean"] / 1000.0) if total_tok["mean"] > 0 else 0.0
    acc_per_sec = (acc * 100.0) / elapsed["mean"] if elapsed["mean"] > 0 else 0.0

    # Stop reason breakdown
    stop_groups = defaultdict(list)
    for r in records:
        sr = str(r.get("stop_reason", "N/A"))
        stop_groups[sr].append(r)

    stop_breakdown = {}
    for sr, grp in sorted(stop_groups.items(), key=lambda x: -len(x[1])):
        gn = len(grp)
        g_acc = sum(safe_int(x.get("correct")) for x in grp) / gn
        g_ora = (sum(safe_int(x.get("oracle_any_correct")) for x in grp) / gn) if has_oracle else None
        g_tok = sum(safe_float(x.get("total_tokens")) for x in grp) / gn
        g_sec = sum(safe_float(x.get("elapsed_sec")) for x in grp) / gn
        g_samp = sum(safe_float(x.get("n_samples")) for x in grp) / gn
        stop_breakdown[sr] = {
            "count": gn,
            "share_pct": (gn / n) * 100.0,
            "accuracy_pct": g_acc * 100.0,
            "oracle_acc_pct": (g_ora * 100.0) if g_ora is not None else None,
            "mean_samples": g_samp,
            "mean_tokens": g_tok,
            "mean_sec": g_sec,
        }

    # Truncation & Guard diagnostics
    trunc_rate = sum(1 for r in records if safe_int(r.get("n_truncated")) > 0) / n
    cot_trunc_rate = sum(1 for r in records if safe_int(r.get("n_cot_truncated")) > 0) / n
    tok_est_rate = sum(1 for r in records if safe_int(r.get("token_estimated_calls")) > 0) / n
    loop_guard_rate = sum(1 for r in records if safe_int(r.get("loop_guard_abort")) > 0) / n

    # Route breakdown
    route_groups = defaultdict(list)
    for r in records:
        if "route" in r and r["route"] is not None:
            route_groups[str(r["route"])].append(r)
    route_breakdown = {}
    for rt, grp in sorted(route_groups.items(), key=lambda x: -len(x[1])):
        gn = len(grp)
        route_breakdown[rt] = {
            "count": gn,
            "share_pct": (gn / n) * 100.0,
            "accuracy_pct": (sum(safe_int(x.get("correct")) for x in grp) / gn) * 100.0,
            "oracle_acc_pct": (sum(safe_int(x.get("oracle_any_correct")) for x in grp) / gn) * 100.0 if has_oracle else None,
            "mean_tokens": sum(safe_float(x.get("total_tokens")) for x in grp) / gn,
            "mean_sec": sum(safe_float(x.get("elapsed_sec")) for x in grp) / gn,
        }

    # PAL, Debugger & Critic diagnostics
    tot_pal = sum(safe_int(r.get("n_pal")) for r in records)
    tot_pal_ok = sum(safe_int(r.get("pal_exec_ok")) for r in records)
    tot_dbg_calls = sum(safe_int(r.get("debug_calls")) for r in records)
    tot_dbg_fixed = sum(safe_int(r.get("debug_fixed")) for r in records)
    tot_critic_calls = sum(safe_int(r.get("critic_calls")) for r in records)
    tot_critic_fail = sum(safe_int(r.get("critic_parse_fail")) for r in records)
    tot_vetoes = sum(safe_int(r.get("local_vetoes")) for r in records)

    # Category breakdown
    cat_groups = defaultdict(list)
    for r in records:
        cat_groups[str(r.get("category", "Unknown"))].append(r)
    category_breakdown = {}
    for cat, grp in sorted(cat_groups.items()):
        gn = len(grp)
        c_corr = sum(safe_int(x.get("correct")) for x in grp)
        c_fn = sum(
            1 for x in grp
            if safe_int(x.get("correct")) == 0
            and check_false_negative_suspect(x.get("predicted_answer"), x.get("ground_truth"))[0]
        )
        category_breakdown[cat] = {
            "count": gn,
            "accuracy_pct": (c_corr / gn) * 100.0,
            "adj_accuracy_pct": ((c_corr + c_fn) / gn) * 100.0,
            "oracle_acc_pct": (sum(safe_int(x.get("oracle_any_correct")) for x in grp) / gn) * 100.0 if has_oracle else None,
            "mean_samples": sum(safe_float(x.get("n_samples")) for x in grp) / gn,
            "mean_tokens": sum(safe_float(x.get("total_tokens")) for x in grp) / gn,
            "mean_sec": sum(safe_float(x.get("elapsed_sec")) for x in grp) / gn,
        }

    # Per-seed breakdown
    seed_groups = defaultdict(list)
    for r in records:
        seed_groups[r.get("seed", 0)].append(r)
    seed_breakdown = {}
    for sd, grp in sorted(seed_groups.items(), key=lambda x: str(x[0])):
        gn = len(grp)
        seed_breakdown[str(sd)] = {
            "count": gn,
            "accuracy_pct": (sum(safe_int(x.get("correct")) for x in grp) / gn) * 100.0,
            "oracle_acc_pct": (sum(safe_int(x.get("oracle_any_correct")) for x in grp) / gn) * 100.0 if has_oracle else None,
            "mean_tokens": sum(safe_float(x.get("total_tokens")) for x in grp) / gn,
            "mean_sec": sum(safe_float(x.get("elapsed_sec")) for x in grp) / gn,
        }

    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy_pct": acc * 100.0,
        "ci95_low_pct": ci_low * 100.0,
        "ci95_high_pct": ci_high * 100.0,
        "false_negative_suspects_count": len(fn_suspects),
        "adjusted_accuracy_pct": adj_acc * 100.0,
        "oracle_accuracy_pct": (oracle_acc * 100.0) if oracle_acc is not None else None,
        "oracle_gap_pct": (oracle_gap * 100.0) if oracle_gap is not None else None,
        "pal_any_accuracy_pct": (pal_any_acc * 100.0) if pal_any_acc is not None else None,
        "cross_validated_rate_pct": (cv_rate * 100.0) if cv_rate is not None else None,
        "accuracy_when_cv_pct": (cv_acc * 100.0) if cv_acc is not None else None,
        "accuracy_when_not_cv_pct": (non_cv_acc * 100.0) if non_cv_acc is not None else None,
        "llm_calls": llm_calls,
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "total_tokens": total_tok,
        "tokens_generator_model": gen_tok,
        "tokens_other_model": other_tok,
        "n_samples": n_samples_stat,
        "acc_pct_per_1k_tokens": acc_per_1k_tok,
        "acc_pct_per_sec": acc_per_sec,
        "truncation_diagnostics": {
            "any_truncated_pct": trunc_rate * 100.0,
            "cot_truncated_pct": cot_trunc_rate * 100.0,
            "token_estimated_pct": tok_est_rate * 100.0,
            "loop_guard_abort_pct": loop_guard_rate * 100.0,
        },
        "auxiliary_modules": {
            "total_pal_calls": tot_pal,
            "pal_exec_ok": tot_pal_ok,
            "pal_exec_success_rate_pct": (tot_pal_ok / tot_pal * 100.0) if tot_pal > 0 else None,
            "debug_calls": tot_dbg_calls,
            "debug_fixed": tot_dbg_fixed,
            "debug_fix_rate_pct": (tot_dbg_fixed / tot_dbg_calls * 100.0) if tot_dbg_calls > 0 else None,
            "critic_calls": tot_critic_calls,
            "critic_parse_fail": tot_critic_fail,
            "critic_parse_fail_rate_pct": (tot_critic_fail / tot_critic_calls * 100.0) if tot_critic_calls > 0 else None,
            "local_vetoes": tot_vetoes,
            "local_veto_rate_pct": (tot_vetoes / n * 100.0),
        },
        "stop_reason_breakdown": stop_breakdown,
        "route_breakdown": route_breakdown,
        "category_breakdown": category_breakdown,
        "seed_breakdown": seed_breakdown,
    }


# ==============================================================================
# 6. PAIRED HEAD-TO-HEAD COMPARISON & ANOMALY COLLECTION
# ==============================================================================

def compare_paired_methods(
    base_records: List[Dict[str, Any]],
    target_records: List[Dict[str, Any]],
    base_name: str,
    target_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base_map = {(r.get("seed", 0), str(r.get("id"))): r for r in base_records}
    target_map = {(r.get("seed", 0), str(r.get("id"))): r for r in target_records}
    common_keys = sorted(set(base_map.keys()) & set(target_map.keys()), key=lambda x: (str(x[0]), x[1]))
    if not common_keys and base_map and target_map:
        base_map = {(0, str(r.get('id'))): r for r in base_records}
        target_map = {(0, str(r.get('id'))): r for r in target_records}
        common_keys = sorted(set(base_map.keys()) & set(target_map.keys()), key=lambda x: (str(x[0]), x[1]))

    wins = losses = both_ok = both_bad = 0
    cat_stats = defaultdict(lambda: {"n": 0, "wins": 0, "losses": 0, "base_ok": 0, "targ_ok": 0})
    paired_rows = []

    for key in common_keys:
        b = base_map[key]
        t = target_map[key]
        b_ok = safe_int(b.get("correct"))
        t_ok = safe_int(t.get("correct"))
        cat = str(t.get("category") or b.get("category") or "Unknown")

        if b_ok == 0 and t_ok == 1:
            status = "WIN"
            wins += 1
        elif b_ok == 1 and t_ok == 0:
            status = "LOSS"
            losses += 1
        elif b_ok == 1 and t_ok == 1:
            status = "BOTH_CORRECT"
            both_ok += 1
        else:
            status = "BOTH_WRONG"
            both_bad += 1

        cat_stats[cat]["n"] += 1
        cat_stats[cat]["base_ok"] += b_ok
        cat_stats[cat]["targ_ok"] += t_ok
        if status == "WIN":
            cat_stats[cat]["wins"] += 1
        elif status == "LOSS":
            cat_stats[cat]["losses"] += 1

        paired_rows.append({
            "seed": key[0],
            "id": key[1],
            "category": cat,
            "status": status,
            f"{base_name}_correct": b_ok,
            f"{target_name}_correct": t_ok,
            f"{base_name}_pred": b.get("predicted_answer"),
            f"{target_name}_pred": t.get("predicted_answer"),
            "ground_truth": t.get("ground_truth"),
            "target_stop_reason": t.get("stop_reason"),
            "target_oracle_any": t.get("oracle_any_correct"),
            "delta_tokens": safe_float(t.get("total_tokens")) - safe_float(b.get("total_tokens")),
            "delta_sec": round(safe_float(t.get("elapsed_sec")) - safe_float(b.get("elapsed_sec")), 3),
        })

    n_paired = len(common_keys)
    mcn = mcnemar_test(wins, losses)
    cat_summary = {}
    for cat, st in sorted(cat_stats.items()):
        cn = st["n"]
        cat_summary[cat] = {
            "n": cn,
            "base_acc_pct": (st["base_ok"] / cn * 100.0) if cn else 0.0,
            "target_acc_pct": (st["targ_ok"] / cn * 100.0) if cn else 0.0,
            "diff_pct": ((st["targ_ok"] - st["base_ok"]) / cn * 100.0) if cn else 0.0,
            "wins": st["wins"],
            "losses": st["losses"],
            "net_gain": st["wins"] - st["losses"],
        }

    summary = {
        "baseline": base_name,
        "target": target_name,
        "paired_count": n_paired,
        "wins_0_to_1": wins,
        "losses_1_to_0": losses,
        "net_gain": wins - losses,
        "net_gain_pct": ((wins - losses) / n_paired * 100.0) if n_paired else 0.0,
        "both_correct": both_ok,
        "both_wrong": both_bad,
        "mcnemar": mcn,
        "by_category": cat_summary,
    }
    return summary, paired_rows


def collect_all_anomalies(
    records: List[Dict[str, Any]],
    paired_losses_keys: set,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    # Compute dynamic latency & token thresholds per method (p95 * 1.75 or > 60s)
    by_method = defaultdict(list)
    for r in records:
        by_method[str(r.get("method", "UNKNOWN"))].append(r)

    thresholds = {}
    for m, grp in by_method.items():
        secs = [safe_float(x.get("elapsed_sec")) for x in grp if "elapsed_sec" in x]
        toks = [safe_float(x.get("total_tokens")) for x in grp if "total_tokens" in x]
        thresholds[m] = {
            "sec": max(60.0, percentile(secs, 95) * 1.75) if secs else 60.0,
            "tok": max(12000.0, percentile(toks, 95) * 1.75) if toks else 12000.0,
        }

    anomaly_rows = []
    counts = Counter()

    for r in records:
        m = str(r.get("method", "UNKNOWN"))
        sd = r.get("seed", 0)
        qid = str(r.get("id", ""))
        corr = safe_int(r.get("correct"))
        ora = safe_int(r.get("oracle_any_correct"))
        cv = safe_int(r.get("cross_validated"))
        sec = safe_float(r.get("elapsed_sec"))
        tok = safe_float(r.get("total_tokens"))
        sr = str(r.get("stop_reason", ""))

        tags = []
        details = []

        # 1. Suspected False Negative
        if corr == 0:
            is_fn, fn_reason = check_false_negative_suspect(r.get("predicted_answer"), r.get("ground_truth"))
            if is_fn:
                tags.append("FALSE_NEGATIVE_SUSPECT")
                details.append(f"Grader mismatch ({fn_reason})")

        # 2. Extraction / Vote distribution pollution
        poll_flags = inspect_extraction_pollution(r)
        if poll_flags:
            tags.append("EXTRACTION_OR_VOTE_POLLUTION")
            details.extend(poll_flags)

        # 3. Oracle Loss (correct answer existed in candidate pool, but voting/critic failed)
        if "oracle_any_correct" in r and ora == 1 and corr == 0:
            tags.append("ORACLE_LOSS")
            if "FALSE_NEGATIVE_SUSPECT" in tags:
                details.append("Oracle loss caused by grader formatting mismatch")
            elif safe_int(r.get("local_vetoes")) > 0:
                details.append("Oracle loss with local_vetoes > 0")
            else:
                details.append(f"Aggregation/voting selected wrong candidate (stop={sr})")

        # 4. False Cross-Validation (both CoT and PAL agreed on wrong answer)
        if "cross_validated" in r and cv == 1 and corr == 0:
            tags.append("FALSE_CROSS_VALIDATION")
            details.append("Cross-validated consensus reached on incorrect answer")

        # 5. Regression vs Baseline
        if (m, sd, qid) in paired_losses_keys:
            tags.append("REGRESSION_VS_BASELINE")
            details.append("Baseline solved correctly, this method failed")

        # 6. Critic parse fail or token estimation fallback
        if safe_int(r.get("critic_parse_fail")) > 0:
            tags.append("CRITIC_PARSE_FAIL")
            details.append(f"critic_parse_fail={r.get('critic_parse_fail')}")
        if safe_int(r.get("token_estimated_calls")) > 0:
            tags.append("TOKEN_ESTIMATED_FALLBACK")
            details.append(f"token_estimated_calls={r.get('token_estimated_calls')}")
        if safe_int(r.get("loop_guard_abort")) > 0:
            tags.append("LOOP_GUARD_ABORT")
            details.append("Triggered loop_guard_abort")

        # 7. Extreme latency or token outlier
        if sec > thresholds[m]["sec"] or tok > thresholds[m]["tok"]:
            tags.append("COMPUTE_LATENCY_OUTLIER")
            details.append(f"sec={sec:.1f}s, tokens={int(tok)}, trunc={r.get('n_truncated', 0)}, stop={sr}")

        if tags:
            for t in tags:
                counts[t] += 1
            anomaly_rows.append({
                "id": qid,
                "method": m,
                "seed": sd,
                "category": r.get("category", ""),
                "anomaly_tags": "|".join(tags),
                "anomaly_details": "; ".join(details),
                "correct": corr,
                "oracle_any_correct": r.get("oracle_any_correct", ""),
                "cross_validated": r.get("cross_validated", ""),
                "stop_reason": sr,
                "ground_truth": r.get("ground_truth"),
                "predicted_answer": r.get("predicted_answer"),
                "n_samples": r.get("n_samples", ""),
                "n_truncated": r.get("n_truncated", ""),
                "elapsed_sec": round(sec, 2),
                "total_tokens": int(tok),
                "vote_distribution": json.dumps(r.get("vote_distribution"), ensure_ascii=False)
                if "vote_distribution" in r else "",
            })

    return anomaly_rows, dict(counts)


# ==============================================================================
# 7. MARKDOWN REPORT GENERATOR
# ==============================================================================

def render_markdown_report(
    input_files: List[str],
    integrity: Dict[str, Any],
    method_summaries: Dict[str, Dict[str, Any]],
    paired_summaries: List[Dict[str, Any]],
    anomaly_counts: Dict[str, int],
    anomaly_rows: List[Dict[str, Any]],
) -> str:
    lines = []
    lines.append("# Experimental Evaluation & Anomaly Audit Report")
    lines.append(f"- **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Input files:** `{', '.join(input_files)}`")
    lines.append(f"- **SymPy symbolic equivalence engine:** {'Enabled' if HAS_SYMPY else 'Not installed (using built-in exact algebraic/matrix/complex normalizers)'}")
    lines.append("")

    # Section 1: Data Integrity
    lines.append("## 1. Data Integrity & Coverage Audit")
    lines.append(f"- **Total valid JSON records loaded:** `{integrity['total_records_loaded']}`")
    lines.append(f"- **Malformed JSON lines:** `{len(integrity['json_parse_errors'])}`")
    lines.append(f"- **Duplicate `(method, seed, id)` records:** `{integrity['duplicate_keys_count']}`")
    lines.append(f"- **Records missing core schema fields:** `{integrity['missing_fields_count']}`")
    lines.append(f"- **Internal field logic mismatches:** `{integrity['logic_mismatches_count']}`")
    lines.append("")
    lines.append("| Method / Seed | Records | Unique IDs | Missing MATH-500 IDs | Sample Missing IDs |")
    lines.append("|---|---:|---:|---:|---|")
    for k, v in integrity["coverage_by_method_seed"].items():
        sample_str = ", ".join(v["missing_math500_sample"][:5]) if v["missing_math500_sample"] else "-"
        lines.append(f"| {k} | {v['total_records']} | {v['unique_ids']} | {v['missing_math500_count']} | {sample_str} |")
    lines.append("")

    if integrity["logic_mismatches"]:
        lines.append("### Internal Logic Mismatches")
        for lm in integrity["logic_mismatches"][:20]:
            lines.append(f"- `{lm['method']}` (seed={lm['seed']}) `{lm['id']}`: {lm['issues']}")
        lines.append("")

    # Section 2: Overall Performance & Cost Table
    lines.append("## 2. Overall Performance, Oracle & Compute Efficiency by Method")
    lines.append("| Metric | " + " | ".join(method_summaries.keys()) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(method_summaries)) + "|")

    def row_fmt(label: str, extractor) -> str:
        vals = [extractor(ms) for ms in method_summaries.values()]
        return f"| {label} | " + " | ".join(vals) + " |"

    lines.append(row_fmt("Sample Count (`N`)", lambda m: f"{m['n']}"))
    lines.append(row_fmt("**Raw Accuracy (%)**", lambda m: f"**{m['accuracy_pct']:.2f}%** ({m['n_correct']}/{m['n']})"))
    lines.append(row_fmt("95% Wilson CI", lambda m: f"[{m['ci95_low_pct']:.2f}%, {m['ci95_high_pct']:.2f}%]"))
    lines.append(row_fmt("Suspected False Negatives", lambda m: f"{m['false_negative_suspects_count']}"))
    lines.append(row_fmt("**Adjusted Accuracy (%)**", lambda m: f"**{m['adjusted_accuracy_pct']:.2f}%**"))
    lines.append(row_fmt("Oracle Any Correct (%)", lambda m: f"{m['oracle_accuracy_pct']:.2f}%" if m['oracle_accuracy_pct'] is not None else "-"))
    lines.append(row_fmt("Oracle Gap (Oracle - Raw)", lambda m: f"{m['oracle_gap_pct']:.2f}%" if m['oracle_gap_pct'] is not None else "-"))
    lines.append(row_fmt("PAL Any Correct (%)", lambda m: f"{m['pal_any_accuracy_pct']:.2f}%" if m['pal_any_accuracy_pct'] is not None else "-"))
    lines.append(row_fmt("Cross-Validated Rate (%)", lambda m: f"{m['cross_validated_rate_pct']:.2f}%" if m['cross_validated_rate_pct'] is not None else "-"))
    lines.append(row_fmt("Acc when Cross-Validated", lambda m: f"{m['accuracy_when_cv_pct']:.2f}%" if m['accuracy_when_cv_pct'] is not None else "-"))
    lines.append(row_fmt("Acc when NOT Cross-Validated", lambda m: f"{m['accuracy_when_not_cv_pct']:.2f}%" if m['accuracy_when_not_cv_pct'] is not None else "-"))
    lines.append(row_fmt("Mean Samples / Query", lambda m: f"{m['n_samples']['mean']:.2f}" if m['n_samples']['count'] else "-"))
    lines.append(row_fmt("Mean LLM Calls / Query", lambda m: f"{m['llm_calls']['mean']:.2f}" if m['llm_calls']['count'] else "-"))
    lines.append(row_fmt("Mean Total Tokens", lambda m: f"{m['total_tokens']['mean']:.1f}"))
    lines.append(row_fmt("Median (P50) Total Tokens", lambda m: f"{m['total_tokens']['p50']:.1f}"))
    lines.append(row_fmt("P95 Total Tokens", lambda m: f"{m['total_tokens']['p95']:.1f}"))
    lines.append(row_fmt("Total Tokens (All Queries)", lambda m: f"{int(m['total_tokens']['sum']):,}"))
    lines.append(row_fmt("Mean Latency (`elapsed_sec`)", lambda m: f"{m['elapsed_sec']['mean']:.2f}s"))
    lines.append(row_fmt("Median (P50) Latency", lambda m: f"{m['elapsed_sec']['p50']:.2f}s"))
    lines.append(row_fmt("P95 / Max Latency", lambda m: f"{m['elapsed_sec']['p95']:.2f}s / {m['elapsed_sec']['max']:.2f}s"))
    lines.append(row_fmt("Total Wall-Clock Time", lambda m: f"{m['elapsed_sec']['sum']/60.0:.1f} min"))
    lines.append(row_fmt("Acc% per 1k Tokens", lambda m: f"{m['acc_pct_per_1k_tokens']:.2f}"))
    lines.append(row_fmt("Any Truncated (`n_truncated>0`)", lambda m: f"{m['truncation_diagnostics']['any_truncated_pct']:.2f}%"))
    lines.append(row_fmt("CoT Truncated (`n_cot_truncated>0`)", lambda m: f"{m['truncation_diagnostics']['cot_truncated_pct']:.2f}%"))
    lines.append(row_fmt("Loop Guard Abort (%)", lambda m: f"{m['truncation_diagnostics']['loop_guard_abort_pct']:.2f}%"))
    lines.append("")

    # Section 3: Paired Head-to-Head Comparisons
    if paired_summaries:
        lines.append("## 3. Paired Head-to-Head Comparison (McNemar's Test)")
        for ps in paired_summaries:
            mcn = ps["mcnemar"]
            lines.append(f"### `{ps['target']}` vs. Baseline `{ps['baseline']}` (Paired N = {ps['paired_count']})")
            lines.append(f"- **Wins (`0 -> 1`):** `{ps['wins_0_to_1']}` | **Losses (`1 -> 0`):** `{ps['losses_1_to_0']}` | **Net Gain:** `{ps['net_gain']:+d}` (`{ps['net_gain_pct']:+.2f}%`)")
            lines.append(f"- **Both Correct:** `{ps['both_correct']}` | **Both Wrong:** `{ps['both_wrong']}`")
            lines.append(f"- **McNemar's Test:** $\\chi^2 = {mcn['chi2']:.4f}$, $p = {mcn['p_value']:.6f}$")
            lines.append("")
            lines.append("| Category | N | Baseline Acc | Target Acc | Diff | Wins | Losses | Net |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for cat, cst in ps["by_category"].items():
                lines.append(
                    f"| {cat} | {cst['n']} | {cst['base_acc_pct']:.2f}% | {cst['target_acc_pct']:.2f}% | "
                    f"{cst['diff_pct']:+.2f}% | {cst['wins']} | {cst['losses']} | {cst['net_gain']:+d} |"
                )
            lines.append("")

    # Section 4: Deep-Dive per Method (Stop Reasons, Routes, Aux Modules, Categories)
    lines.append("## 4. Method Internals: Stop Reasons, Routing & Auxiliary Modules")
    for m_name, ms in method_summaries.items():
        lines.append(f"### Method: `{m_name}`")
        aux = ms["auxiliary_modules"]
        if aux["total_pal_calls"] > 0 or aux["critic_calls"] > 0:
            lines.append("- **Auxiliary Modules:** "
                         f"PAL Exec Success = `{aux['pal_exec_ok']}/{aux['total_pal_calls']}` "
                         f"({aux['pal_exec_success_rate_pct']:.1f}%), "
                         f"Debug Fixed = `{aux['debug_fixed']}/{aux['debug_calls']}` "
                         f"({aux['debug_fix_rate_pct']:.1f}%)" if aux['debug_calls'] else
                         f"PAL Exec Success = `{aux['pal_exec_ok']}/{aux['total_pal_calls']}` "
                         f"({aux['pal_exec_success_rate_pct']:.1f}%), Debug Calls = `0`"
                         )
            lines.append(f"- **Critic & Vetoes:** Critic Calls = `{aux['critic_calls']}`, "
                         f"Parse Fails = `{aux['critic_parse_fail']}`, Local Vetoes = `{aux['local_vetoes']}` "
                         f"({aux['local_veto_rate_pct']:.2f}%)")
        lines.append("")
        lines.append("| Stop Reason | Count | Share | Accuracy | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for sr, sinfo in ms["stop_reason_breakdown"].items():
            ora_s = f"{sinfo['oracle_acc_pct']:.2f}%" if sinfo["oracle_acc_pct"] is not None else "-"
            lines.append(
                f"| `{sr}` | {sinfo['count']} | {sinfo['share_pct']:.1f}% | {sinfo['accuracy_pct']:.2f}% | "
                f"{ora_s} | {sinfo['mean_samples']:.2f} | {sinfo['mean_tokens']:.0f} | {sinfo['mean_sec']:.2f}s |"
            )
        lines.append("")

        if ms["route_breakdown"]:
            lines.append("| Route | Count | Share | Accuracy | Oracle Acc | Mean Tokens | Mean Sec |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for rt, rinfo in ms["route_breakdown"].items():
                ora_s = f"{rinfo['oracle_acc_pct']:.2f}%" if rinfo["oracle_acc_pct"] is not None else "-"
                lines.append(
                    f"| `{rt}` | {rinfo['count']} | {rinfo['share_pct']:.1f}% | {rinfo['accuracy_pct']:.2f}% | "
                    f"{ora_s} | {rinfo['mean_tokens']:.0f} | {rinfo['mean_sec']:.2f}s |"
                )
            lines.append("")

        lines.append("| Category | N | Raw Acc | Adjusted Acc | Oracle Acc | Mean Samples | Mean Tokens | Mean Sec |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cat, cinfo in ms["category_breakdown"].items():
            ora_s = f"{cinfo['oracle_acc_pct']:.2f}%" if cinfo["oracle_acc_pct"] is not None else "-"
            lines.append(
                f"| {cat} | {cinfo['count']} | {cinfo['accuracy_pct']:.2f}% | {cinfo['adj_accuracy_pct']:.2f}% | "
                f"{ora_s} | {cinfo['mean_samples']:.2f} | {cinfo['mean_tokens']:.0f} | {cinfo['mean_sec']:.2f}s |"
            )
        lines.append("")

    # Section 5: Anomalies Summary
    lines.append("## 5. Anomaly & Error Audit Summary")
    lines.append(f"- **Total flagged records:** `{len(anomaly_rows)}` (full details exported to `anomalies_audit.csv`)")
    if anomaly_counts:
        lines.append("| Anomaly Tag | Count |")
        lines.append("|---|---:|")
        for tag, cnt in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{tag}` | {cnt} |")
    lines.append("")

    # Show top sample false negatives directly in report
    fn_samples = [r for r in anomaly_rows if "FALSE_NEGATIVE_SUSPECT" in r["anomaly_tags"]][:15]
    if fn_samples:
        lines.append("### Sample Suspected False Negatives (Grader Normalization Bugs)")
        lines.append("| Method | ID | Category | Ground Truth | Predicted Answer | Reason |")
        lines.append("|---|---|---|---|---|---|")
        for r in fn_samples:
            gt_esc = str(r["ground_truth"]).replace("\n", " ").replace("|", "\\|")
            pr_esc = str(r["predicted_answer"]).replace("\n", " ").replace("|", "\\|")
            lines.append(f"| `{r['method']}` | `{r['id']}` | {r['category']} | `{gt_esc}` | `{pr_esc}` | {r['anomaly_details']} |")
        lines.append("")

    return "\n".join(lines)


# ==============================================================================
# 8. MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation & Anomaly Audit for JSONL Experiment Outputs"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to .jsonl file(s) or directory containing .jsonl files",
    )
    parser.add_argument(
        "--baseline",
        default="DIRECT",
        help="Baseline method name for paired comparison (default: DIRECT)",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_report",
        help="Root output directory; a subfolder named after the JSONL file is created inside (default: eval_report)",
    )
    parser.add_argument(
        "--no-subdir",
        action="store_true",
        help="Write reports directly into --out-dir without creating a JSONL-named subfolder",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append a YYYYMMDD_HHMMSS timestamp suffix to the output folder name",
    )
    args = parser.parse_args()

    # Expand directories if passed
    jsonl_paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            jsonl_paths.extend(str(x) for x in sorted(p.glob("*.jsonl")))
        else:
            jsonl_paths.append(str(p))

    if not jsonl_paths:
        raise SystemExit("No .jsonl files found in input paths.")

    records, parse_errors = load_jsonl_files(jsonl_paths)
    integrity = audit_data_integrity(records, parse_errors)

    by_method = defaultdict(list)
    for r in records:
        by_method[str(r.get("method", "UNKNOWN"))].append(r)

    # Order methods with baseline first if present
    method_names = sorted(by_method.keys())
    if args.baseline in method_names:
        method_names.remove(args.baseline)
        method_names = [args.baseline] + method_names

    method_summaries = {m: analyze_method(by_method[m]) for m in method_names}

    # Paired comparisons against baseline
    paired_summaries = []
    all_paired_rows = []
    paired_losses_keys = set()

    base_name = args.baseline if args.baseline in by_method else (method_names[0] if method_names else None)
    if base_name and len(method_names) > 1:
        for target_name in method_names:
            if target_name == base_name:
                continue
            ps, prows = compare_paired_methods(
                by_method[base_name], by_method[target_name], base_name, target_name
            )
            paired_summaries.append(ps)
            all_paired_rows.extend(prows)
            for pr in prows:
                if pr["status"] == "LOSS":
                    paired_losses_keys.add((target_name, pr["seed"], pr["id"]))

    anomaly_rows, anomaly_counts = collect_all_anomalies(records, paired_losses_keys)

    out_dir = resolve_output_dir(
        base_out_dir=args.out_dir,
        raw_inputs=args.inputs,
        jsonl_paths=jsonl_paths,
        no_subdir=args.no_subdir,
        add_timestamp=args.timestamp,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write Markdown Report
    md_text = render_markdown_report(
        jsonl_paths, integrity, method_summaries, paired_summaries, anomaly_counts, anomaly_rows
    )
    md_path = out_dir / "evaluation_report.md"
    md_path.write_text(md_text, encoding="utf-8")

    # 2. Write Structured JSON Summary
    json_summary = {
        "generated_at": datetime.now().isoformat(),
        "input_files": jsonl_paths,
        "output_dir": str(out_dir.resolve()),
        "integrity": integrity,
        "methods": method_summaries,
        "paired_comparisons": paired_summaries,
        "anomaly_counts": anomaly_counts,
    }
    json_path = out_dir / "summary_metrics.json"
    json_path.write_text(json.dumps(json_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Write Anomalies Audit CSV
    anom_csv_path = out_dir / "anomalies_audit.csv"
    if anomaly_rows:
        with anom_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(anomaly_rows[0].keys()))
            writer.writeheader()
            writer.writerows(anomaly_rows)
    else:
        anom_csv_path.write_text("id,method,seed,anomaly_tags\n", encoding="utf-8-sig")

    # 4. Write Paired Comparison CSV
    paired_csv_path = out_dir / "paired_comparison.csv"
    if all_paired_rows:
        with paired_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            fnames = list(dict.fromkeys(k for r in all_paired_rows for k in r.keys()))
            writer = csv.DictWriter(f, fieldnames=fnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_paired_rows)

    print(f"[OK] Loaded {len(records)} records from {len(jsonl_paths)} file(s).")
    for m, ms in method_summaries.items():
        print(
            f"  - {m}: N={ms['n']}, RawAcc={ms['accuracy_pct']:.2f}%, "
            f"AdjAcc={ms['adjusted_accuracy_pct']:.2f}%, "
            f"MeanTokens={ms['total_tokens']['mean']:.1f}, MeanSec={ms['elapsed_sec']['mean']:.2f}s"
        )
    print(f"[OK] Flagged {len(anomaly_rows)} anomalies across {len(anomaly_counts)} categories.")
    print(f"[OK] Reports saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()