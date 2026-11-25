#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_stage3.py -- Script điều phối thực nghiệm Giai đoạn 3 cho DualRef.

Tự động nạp `evaluation_patches.py` đè lên `evaluation.py` và hỗ trợ 4 chế độ chạy:
  1. --regrade-gsm8k : Chấm lại result_gsm8k_raw.csv bằng parser & chuẩn hóa LaTeX mới.
  2. --batch 1       : Đánh giá toàn bộ tập dữ liệu (DIRECT vs DUALREF) trên MATH-500 hoặc --dataset.
  3. --batch 2       : Baseline cùng ngân sách (SC@1..12, PAL@1..12) & Ablation trên tập phân tầng (mặc định 300 bài).
  4. --batch 3       : Chạy lại tập chẩn đoán 40 bài (M01-M40) với parser mới & full log.

Ví dụ chạy:
  python run_stage3.py --regrade-gsm8k
  python run_stage3.py --batch 1 --critic critic
  python run_stage3.py --batch 1 --dataset gsm8k_test_1000.jsonl --critic critic
  python run_stage3.py --batch 2 --critic critic
  python run_stage3.py --batch 2 --dataset gsm8k_test_1000.jsonl --sample-n 300 --critic critic
  python run_stage3.py --batch 3 --critic critic --limit 5
"""

import os
import sys
import json
import math
import time
import random
import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Bảo đảm in UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Nạp evaluation.py và áp dụng bản vá evaluation_patches.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import evaluation as ev  # noqa: E402


def load_patched_evaluation(patch_filenames=("evaluation_patches.py", "evaluation_patches_speed.py")):
    for fn in patch_filenames:
        p = fn if os.path.exists(fn) else os.path.join(SCRIPT_DIR, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy tệp bản vá: {p}")
        with open(p, "r", encoding="utf-8") as f:
            exec(compile(f.read(), p, "exec"), ev.__dict__)
    return ev


load_patched_evaluation()


# ======================================================================
# 1. NẠP DỮ LIỆU ĐA NGUỒN (MATH-500 MẶC ĐỊNH HOẶC --dataset BẤT KỲ)
# ======================================================================

def _normalize_loaded_rows(rows: List[Dict[str, Any]], default_prefix: str = "P") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, raw in enumerate(rows, 1):
        r = raw["question"] if isinstance(raw.get("question"), dict) else raw
        prob = str(r.get("problem") or r.get("question") or "").strip()
        gt = str(
            r.get("ground_truth")
            or r.get("ground_truth_answer")
            or r.get("answer")
            or raw.get("answer")
            or ""
        ).strip()
        cat = str(r.get("category") or r.get("subject") or raw.get("subject") or "").strip()
        uid = str(r.get("id") or r.get("unique_id") or raw.get("unique_id") or f"{default_prefix}{i:04d}").strip()
        if prob and gt:
            out.append({
                "id": uid,
                "category": cat,
                "problem": prob,
                "ground_truth": gt,
            })
    return out


def load_target_dataset(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], str, bool]:
    """
    Trả về (items, dataset_tag, is_default_math500):
      - Nếu không truyền --dataset: nạp MATH-500 chuẩn qua ev.load_math500(args.math500_cache)
        và giữ nguyên tag 'math500' để tương thích 100% với các file checkpoint/CSV cũ.
      - Nếu truyền --dataset <path>: nạp trực tiếp tệp .jsonl / .json / .csv bất kỳ
        (ví dụ: gsm8k_test_1000.jsonl) và dùng tên tệp (không đuôi) làm dataset_tag.
    """
    if not getattr(args, "dataset", None):
        items = ev.load_math500(cache_path=args.math500_cache)
        if os.path.exists(args.math500_cache):
            with open(args.math500_cache, "r", encoding="utf-8") as f:
                raw_lines = [json.loads(ln) for ln in f if ln.strip()]
            for it, raw in zip(items, raw_lines):
                if it.get("category") in ("", "unknown"):
                    q = raw["question"] if isinstance(raw.get("question"), dict) else raw
                    it["category"] = str(q.get("subject") or raw.get("subject") or "unknown").strip()
        return items, "math500", True

    ds_path = args.dataset
    if not os.path.exists(ds_path):
        alt_path = os.path.join(SCRIPT_DIR, ds_path)
        if os.path.exists(alt_path):
            ds_path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {args.dataset}")

    tag = os.path.splitext(os.path.basename(ds_path))[0]
    if ds_path.endswith(".csv"):
        df = pd.read_csv(ds_path, dtype=str).fillna("")
        rows = df.to_dict("records")
    elif ds_path.endswith(".jsonl"):
        with open(ds_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(ds_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

    items = _normalize_loaded_rows(rows, default_prefix=tag.upper())
    if not items:
        raise ValueError(f"Tệp dữ liệu rỗng hoặc thiếu cột problem/ground_truth: {ds_path}")
    print(f"[Dataset Loaded] {os.path.abspath(ds_path)} -> {len(items)} bài (tag={tag})")
    return items, tag, False


def stratified_sample_items(
    all_items: List[Dict[str, str]],
    n: int = 300,
    seed: int = 42,
    save_ids_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Lấy mẫu phân tầng đúng `n` bài theo `category` với `seed` cố định."""
    if n >= len(all_items):
        chosen = list(all_items)
    elif hasattr(ev, "stratified_sample_math500"):
        chosen = ev.stratified_sample_math500(all_items, n=n, seed=seed, save_ids_path=None)
    else:
        rng = random.Random(seed)
        by_cat: Dict[str, List[Dict[str, str]]] = {}
        for it in all_items:
            by_cat.setdefault(it.get("category") or "general", []).append(it)
        total = len(all_items)
        alloc: Dict[str, int] = {}
        remainders: List[Tuple[float, str]] = []
        for cat, group in sorted(by_cat.items()):
            exact = len(group) * n / total
            k = int(math.floor(exact))
            alloc[cat] = k
            remainders.append((exact - k, cat))
        for _, cat in sorted(remainders, reverse=True)[: n - sum(alloc.values())]:
            alloc[cat] += 1
        chosen = []
        for cat, group in sorted(by_cat.items()):
            pool = list(group)
            rng.shuffle(pool)
            chosen.extend(pool[: alloc[cat]])
        id_order = {it["id"]: idx for idx, it in enumerate(all_items)}
        chosen.sort(key=lambda x: id_order[x["id"]])

    if save_ids_path:
        with open(save_ids_path, "w", encoding="utf-8") as f:
            for it in chosen:
                f.write(it["id"] + "\n")
    return chosen


# ======================================================================
# 2. THỐNG KÊ GHÉP CẶP: BOOTSTRAP 95% CI & MCNEMAR EXACT P-VALUE
# ======================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap_ci(
    acc_a: np.ndarray,
    acc_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Khoảng tin cậy bootstrap 95% cho hiệu độ chính xác (acc_a - acc_b)."""
    n = len(acc_a)
    if n == 0:
        return 0.0, 0.0, 0.0
    diff = acc_a.astype(float) - acc_b.astype(float)
    point = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, float(lo), float(hi)


def mcnemar_exact_pvalue(acc_a: np.ndarray, acc_b: np.ndarray) -> Tuple[int, int, float]:
    """Kiểm định McNemar chính xác hai phía (nhị thức) trên các cặp bất đồng (b = A đúng B sai, c = A sai B đúng)."""
    b = int(np.sum((acc_a == 1) & (acc_b == 0)))
    c = int(np.sum((acc_a == 0) & (acc_b == 1)))
    m = b + c
    if m == 0:
        return b, c, 1.0
    k = min(b, c)
    # P(X <= k) với X ~ Binomial(m, 0.5), tính ổn định số học bằng log-gamma
    log_half_m = -m * math.log(2.0)
    log_fact_m = math.lgamma(m + 1)
    tail = 0.0
    for i in range(k + 1):
        log_comb = log_fact_m - math.lgamma(i + 1) - math.lgamma(m - i + 1)
        tail += math.exp(log_comb + log_half_m)
    p_val = min(1.0, 2.0 * tail)
    return b, c, p_val


def _is_nan_pred(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in ("none", "nan", "null")


def summarize_methods(df: pd.DataFrame, method_order: Optional[List[str]] = None) -> pd.DataFrame:
    """Tổng hợp độ chính xác, tỷ lệ NaN, độ chính xác có điều kiện, trung bình token & số lượt gọi."""
    if df.empty:
        return pd.DataFrame()
    if method_order is None:
        method_order = list(dict.fromkeys(df["method"].tolist()))

    rows = []
    for m in method_order:
        g = df[df["method"] == m]
        if g.empty:
            continue
        n = len(g)
        correct_arr = g["correct"].astype(int).to_numpy()
        k = int(correct_arr.sum())
        lo, hi = wilson_ci(k, n)

        nan_mask = g["predicted_answer"].apply(_is_nan_pred).to_numpy()
        n_nan = int(nan_mask.sum())
        n_valid = n - n_nan
        cond_acc = (100.0 * k / n_valid) if n_valid > 0 else float("nan")

        calls_mean = float(g["llm_calls"].astype(float).mean()) if "llm_calls" in g else float("nan")
        pt_mean = float(g["prompt_tokens"].astype(float).mean()) if "prompt_tokens" in g else float("nan")
        ct_mean = float(g["completion_tokens"].astype(float).mean()) if "completion_tokens" in g else float("nan")
        tt_mean = float(g["total_tokens"].astype(float).mean()) if "total_tokens" in g else float("nan")
        sec_mean = float(g["elapsed_sec"].astype(float).mean()) if "elapsed_sec" in g else float("nan")
        trunc_cnt = int(g["n_truncated"].astype(float).sum()) if "n_truncated" in g else 0

        rows.append({
            "method": m,
            "n": n,
            "correct": k,
            "acc_pct": round(100.0 * k / n, 2),
            "ci95_low_pct": round(100.0 * lo, 2),
            "ci95_high_pct": round(100.0 * hi, 2),
            "nan_count": n_nan,
            "nan_pct": round(100.0 * n_nan / n, 2),
            "cond_acc_pct": round(cond_acc, 2) if not math.isnan(cond_acc) else float("nan"),
            "mean_llm_calls": round(calls_mean, 2),
            "mean_prompt_tokens": round(pt_mean, 1),
            "mean_completion_tokens": round(ct_mean, 1),
            "mean_total_tokens": round(tt_mean, 1),
            "mean_elapsed_sec": round(sec_mean, 2),
            "truncated_calls": trunc_cnt,
        })
    return pd.DataFrame(rows)


def print_paired_comparison(df: pd.DataFrame, target_method: str, baselines: List[str]) -> pd.DataFrame:
    """In bảng đối chiếu ghép cặp giữa `target_method` và các `baselines`."""
    t_df = df[df["method"] == target_method].drop_duplicates("id").set_index("id")
    if t_df.empty:
        return pd.DataFrame()

    comp_rows = []
    print(f"\nĐối chiếu ghép cặp ({target_method} vs Baselines):")
    print(f"{'Comparison':<28}{'N':>5}{'Win/Loss/Tie':>15}{'Delta%':>9}{'95% Boot CI':>18}{'McNemar p':>12}")
    for b in baselines:
        if b == target_method:
            continue
        b_df = df[df["method"] == b].drop_duplicates("id").set_index("id")
        common_ids = [pid for pid in t_df.index if pid in b_df.index]
        if not common_ids:
            continue
        a_arr = t_df.loc[common_ids, "correct"].astype(int).to_numpy()
        b_arr = b_df.loc[common_ids, "correct"].astype(int).to_numpy()
        wins, losses, p_val = mcnemar_exact_pvalue(a_arr, b_arr)
        ties = len(common_ids) - wins - losses
        delta, lo, hi = paired_bootstrap_ci(a_arr, b_arr, n_boot=10000, seed=42)
        label = f"{target_method} vs {b}"
        wlt = f"+{wins}/-{losses}/={ties}"
        ci_str = f"[{100*lo:+.2f}, {100*hi:+.2f}]"
        print(f"{label:<28}{len(common_ids):>5}{wlt:>15}{100*delta:>+9.2f}{ci_str:>18}{p_val:>12.4g}")
        comp_rows.append({
            "target": target_method,
            "baseline": b,
            "n_paired": len(common_ids),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "delta_pct": round(100.0 * delta, 2),
            "boot_ci95_low_pct": round(100.0 * lo, 2),
            "boot_ci95_high_pct": round(100.0 * hi, 2),
            "mcnemar_p_exact": p_val,
        })
    return pd.DataFrame(comp_rows)


# ======================================================================
# 3. TIỆN ÍCH CHECKPOINT & KIỂM TRA KẾT NỐI LM STUDIO
# ======================================================================

def load_checkpoint_jsonl(path: str, resume: bool = True) -> List[Dict[str, Any]]:
    if not resume:
        if os.path.exists(path):
            os.remove(path)
        return []
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def verify_llm_server(base_url: str, generator: str, critic: str) -> None:
    ev.BASE_URL = base_url
    ev.GENERATOR_MODEL = generator
    ev.CRITIC_MODEL = critic
    ev.VERIFIER_MODEL = critic
    ev._client = None
    try:
        models = [m.id for m in ev.get_client().models.list().data]
        print(f"[LM Studio ONLINE] Endpoint={base_url} | Models={models}")
    except Exception as e:
        raise RuntimeError(f"Không thể kết nối tới LM Studio tại {base_url}: {e}") from e


# ======================================================================
# 4. CHẾ ĐỘ 1: --regrade-gsm8k
# ======================================================================

def find_gsm8k_csv(custom_path: Optional[str] = None) -> str:
    candidates = [
        custom_path,
        "result_gsm8k_raw.csv",
        os.path.join(SCRIPT_DIR, "result_gsm8k_raw.csv"),
        os.path.join(SCRIPT_DIR, "report", "old_runs", "result_gsm8k_raw.csv"),
        os.path.join(SCRIPT_DIR, "..", "result_gsm8k", "result_gsm8k_raw.csv"),
        os.path.join("..", "result_gsm8k", "result_gsm8k_raw.csv"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Không tìm thấy result_gsm8k_raw.csv (đã tìm trong thư mục hiện tại, report/old_runs/ và ../result_gsm8k/)."
    )


def run_regrade_gsm8k(csv_path: Optional[str] = None) -> None:
    path = find_gsm8k_csv(csv_path)
    print("=" * 96)
    print(f"REGRADE GSM8K: {os.path.abspath(path)}")
    print("=" * 96)

    df = pd.read_csv(path, dtype=str).fillna("")
    old_correct = df["correct"].astype(int).tolist()
    old_preds = df["predicted_answer"].tolist()

    new_preds: List[Optional[str]] = []
    new_correct: List[int] = []
    source_tag: List[str] = []

    for _, row in df.iterrows():
        raw_pred = row.get("predicted_answer", "")
        p = ev.clean_final_answer(raw_pred)
        src = "cleaned_pred"
        if p is None and row.get("raw_output", "").strip():
            p = ev.extract_final_answer(row["raw_output"])
            if p is not None:
                src = "reextracted_raw_output"
        if p is None and row.get("solution_explanation", "").strip():
            p = ev.extract_final_answer(row["solution_explanation"])
            if p is not None:
                src = "reextracted_solution"
        if p is None:
            src = "none"

        gt = row.get("ground_truth", "")
        c = int(ev.evaluate_answer_match(p, gt))
        new_preds.append(p)
        new_correct.append(c)
        source_tag.append(src)

    df["old_predicted_answer"] = old_preds
    df["old_correct"] = old_correct
    df["predicted_answer"] = [p if p is not None else "" for p in new_preds]
    df["correct"] = new_correct
    df["regrade_source"] = source_tag

    out_csv = "stage3_gsm8k_regraded.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'Method':<12}{'N':>6}{'Old Acc%':>11}{'New Acc%':>11}{'Gain%':>9}{'NaN%':>9}{'Cond Acc%':>12}")
    summary_rows = []
    for m, g in df.groupby("method", sort=False):
        n = len(g)
        oc = int(g["old_correct"].astype(int).sum())
        nc = int(g["correct"].astype(int).sum())
        n_nan = int((g["predicted_answer"].str.strip() == "").sum())
        n_valid = n - n_nan
        cond = (100.0 * nc / n_valid) if n_valid > 0 else float("nan")
        print(
            f"{m:<12}{n:>6}{100*oc/n:>11.2f}{100*nc/n:>11.2f}"
            f"{100*(nc-oc)/n:>+9.2f}{100*n_nan/n:>9.2f}{cond:>12.2f}"
        )
        summary_rows.append({
            "method": m,
            "n": n,
            "old_correct": oc,
            "old_acc_pct": round(100.0 * oc / n, 2),
            "new_correct": nc,
            "new_acc_pct": round(100.0 * nc / n, 2),
            "gain_pct": round(100.0 * (nc - oc) / n, 2),
            "nan_count": n_nan,
            "nan_pct": round(100.0 * n_nan / n, 2),
            "cond_acc_pct": round(cond, 2),
        })

    pd.DataFrame(summary_rows).to_csv("stage3_gsm8k_regraded_summary.csv", index=False, encoding="utf-8-sig")
    if "V5" in set(df["method"]) and "DIRECT" in set(df["method"]):
        print_paired_comparison(df, "V5", ["DIRECT"])
    print(f"\nĐã lưu: {os.path.abspath(out_csv)}")


# ======================================================================
# 5. CHẾ ĐỘ 2: --batch 1 (FULL DATASET: DIRECT vs DUALREF)
# ======================================================================

def run_batch1(args: argparse.Namespace) -> None:
    items, tag, _ = load_target_dataset(args)
    if args.limit:
        items = items[: args.limit]

    ckpt_path = f"stage3_batch1_{tag}.jsonl"
    records = load_checkpoint_jsonl(ckpt_path, resume=not args.no_resume)
    done = {(r["id"], r["method"], int(r.get("seed", args.seed))) for r in records}

    methods = ["DIRECT", "DUALREF"]
    total = len(items) * len(methods)
    print("=" * 96)
    print(f"BATCH 1 -- {tag.upper()} FULL ({len(items)} bài x {len(methods)} cấu hình = {total} lượt)")
    print(f"Checkpoint: {os.path.abspath(ckpt_path)} (đã xong {len(done)}/{total})")
    print("=" * 96)

    step_idx = 0
    for method in methods:
        for item in items:
            step_idx += 1
            key = (item["id"], method, args.seed)
            if key in done:
                continue
            if method == "DIRECT":
                r = ev.run_logged(ev.run_direct, item, seed=args.seed)
            else:
                r = ev.run_logged(ev.run_v5, item, method="DUALREF", seed=args.seed)
            ev.append_jsonl_checkpoint(ckpt_path, r)
            records.append(r)
            done.add(key)
            cat_str = f" ({item['category']})" if item.get("category") else ""
            print(
                f"[{step_idx}/{total}] {method:<10} {item['id']}{cat_str} | "
                f"pred={r['predicted_answer']} | GT={r['ground_truth']} | "
                f"correct={r['correct']} | calls={r['llm_calls']} | tok={r['total_tokens']} | "
                f"{r.get('stop_reason', '')}"
            )

    df = pd.DataFrame(records)
    summary_df = summarize_methods(df, method_order=methods)
    print("\n" + "=" * 96)
    print(f"KẾT QUẢ BATCH 1 -- {tag.upper()}")
    print("=" * 96)
    print(summary_df.to_string(index=False))

    paired_df = print_paired_comparison(df, "DUALREF", ["DIRECT"])
    sum_csv = f"stage3_batch1_{tag}_summary.csv"
    pair_csv = f"stage3_batch1_{tag}_paired.csv"
    cat_csv = f"stage3_batch1_{tag}_by_category.csv"

    summary_df.to_csv(sum_csv, index=False, encoding="utf-8-sig")
    if not paired_df.empty:
        paired_df.to_csv(pair_csv, index=False, encoding="utf-8-sig")

    # Thống kê theo từng chủ đề (Subject/Category) nếu tập dữ liệu có phân loại
    has_cats = df["category"].astype(str).str.strip().ne("").any()
    if has_cats:
        cat_rows = []
        for (cat, m), g in df.groupby(["category", "method"], sort=False):
            n = len(g)
            k = int(g["correct"].astype(int).sum())
            cat_rows.append({
                "category": cat or "general",
                "method": m,
                "n": n,
                "correct": k,
                "acc_pct": round(100.0 * k / n, 2),
                "mean_tokens": round(float(g["total_tokens"].astype(float).mean()), 1),
            })
        cat_df = pd.DataFrame(cat_rows)
        if not cat_df.empty:
            cat_df.to_csv(cat_csv, index=False, encoding="utf-8-sig")
        print(f"\nĐã xuất: {sum_csv} & {cat_csv}")
    else:
        print(f"\nĐã xuất: {sum_csv}")


# ======================================================================
# 6. CHẾ ĐỘ 3: --batch 2 (CURVES & ABLATION TRÊN TẬP PHÂN TẦNG)
# ======================================================================

def load_or_create_stratified_subset(
    all_items: List[Dict[str, str]],
    tag: str,
    sample_n: int = 300,
    seed: int = 42,
) -> List[Dict[str, str]]:
    target_n = min(sample_n, len(all_items))
    ids_path = f"{tag}_stratified_{target_n}_ids.txt"
    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            wanted = [ln.strip() for ln in f if ln.strip()]
        by_id = {it["id"]: it for it in all_items}
        chosen = [by_id[pid] for pid in wanted if pid in by_id]
        if len(chosen) == target_n:
            return chosen
    return stratified_sample_items(all_items, n=target_n, seed=seed, save_ids_path=ids_path)


def run_batch2(args: argparse.Namespace) -> None:
    all_items, tag, is_default_math500 = load_target_dataset(args)
    items = load_or_create_stratified_subset(all_items, tag=tag, sample_n=args.sample_n, seed=42)
    if args.limit:
        items = items[: args.limit]

    # Giữ nguyên tên file xuất cũ cho MATH-500 mặc định; dùng hậu tố _{tag} cho dataset ngoài
    if is_default_math500 and args.sample_n == 300:
        ckpt_path = "stage3_batch2_math300.jsonl"
        curves_csv = "stage3_batch2_curves.csv"
        ablation_csv = "stage3_batch2_ablation.csv"
        paired_csv = "stage3_batch2_paired.csv"
    else:
        ckpt_path = f"stage3_batch2_{tag}_{len(items)}.jsonl"
        curves_csv = f"stage3_batch2_{tag}_curves.csv"
        ablation_csv = f"stage3_batch2_{tag}_ablation.csv"
        paired_csv = f"stage3_batch2_{tag}_paired.csv"

    records = load_checkpoint_jsonl(ckpt_path, resume=not args.no_resume)
    done = {(r["id"], r["method"]) for r in records}

    # 6 tác vụ thực thi trực tiếp (SC_POOL12 & PAL_POOL12 cho phép rút offline k=1,3,5,7,9,12)
    jobs = [
        ("SC_POOL12", lambda it: ev.run_sc_pool(it, max_k=12, seed=42)),
        ("PAL_POOL12", lambda it: ev.run_pal_pool(it, max_k=12, seed=42)),
        ("DUALREF", lambda it: ev.run_logged(ev.run_v5, it, method="DUALREF", seed=args.seed)),
        ("DUALREF_NOCRIT", lambda it: ev.run_logged(ev.run_v5, it, method="DUALREF_NOCRIT", seed=args.seed)),
        ("V5_COT", lambda it: ev.run_logged(ev.run_v5, it, method="V5_COT", seed=args.seed)),
        ("PAL_ONLY", lambda it: ev.run_logged(ev.run_v5, it, method="PAL_ONLY", seed=args.seed)),
    ]

    total = len(items) * len(jobs)
    print("=" * 96)
    print(f"BATCH 2 -- {tag.upper()}-{len(items)} STRATIFIED ({len(items)} bài x {len(jobs)} tác vụ = {total} lượt)")
    print(f"Checkpoint: {os.path.abspath(ckpt_path)} (đã xong {len(done)}/{total})")
    print("=" * 96)

    step_idx = 0
    for m_name, runner in jobs:
        for item in items:
            step_idx += 1
            if (item["id"], m_name) in done:
                continue
            r = runner(item)
            ev.append_jsonl_checkpoint(ckpt_path, r)
            records.append(r)
            done.add((item["id"], m_name))
            print(
                f"[{step_idx}/{total}] {m_name:<15} {item['id']} | pred={r['predicted_answer']} | "
                f"GT={r['ground_truth']} | correct={r['correct']} | calls={r['llm_calls']} | "
                f"tok={r['total_tokens']}"
            )

    # Trích xuất offline các mốc k in {1, 3, 5, 7, 9, 12} từ SC_POOL12 và PAL_POOL12
    ks = (1, 3, 5, 7, 9, 12)
    curve_records: List[Dict[str, Any]] = []
    for r in records:
        if r.get("method") in ("SC_POOL12", "PAL_POOL12"):
            curve_records.extend(ev.eval_pool_at_k(r, ks=ks))

    curve_df_raw = pd.DataFrame(curve_records)
    curve_order = [f"SC@{k}" for k in ks] + [f"PAL@{k}" for k in ks]
    curves_summary = summarize_methods(curve_df_raw, method_order=curve_order)

    # Gộp các biến thể ablation + SC@12 + PAL@12 để so sánh cùng trần ngân sách
    ablation_raw = pd.concat(
        [
            pd.DataFrame([r for r in records if r.get("method") in ("DUALREF", "DUALREF_NOCRIT", "V5_COT", "PAL_ONLY")]),
            curve_df_raw[curve_df_raw["method"].isin(["SC@12", "PAL@12"])],
        ],
        ignore_index=True,
    )
    ablation_order = ["DUALREF", "DUALREF_NOCRIT", "V5_COT", "PAL_ONLY", "SC@12", "PAL@12"]
    ablation_summary = summarize_methods(ablation_raw, method_order=ablation_order)

    print("\n" + "=" * 96)
    print(f"ĐƯỜNG CONG ĐỘ CHÍNH XÁC -- TOKEN -- LƯỢT GỌI ({tag.upper()}: SC@1..12 & PAL@1..12)")
    print("=" * 96)
    print(curves_summary.to_string(index=False))

    print("\n" + "=" * 96)
    print(f"BẢNG ABLATION & CÙNG NGÂN SÁCH ({tag.upper()}-{len(items)})")
    print("=" * 96)
    print(ablation_summary.to_string(index=False))

    paired_df = print_paired_comparison(
        ablation_raw,
        "DUALREF",
        ["DUALREF_NOCRIT", "V5_COT", "PAL_ONLY", "SC@12", "PAL@12"],
    )

    curves_summary.to_csv(curves_csv, index=False, encoding="utf-8-sig")
    ablation_summary.to_csv(ablation_csv, index=False, encoding="utf-8-sig")
    if not paired_df.empty:
        paired_df.to_csv(paired_csv, index=False, encoding="utf-8-sig")
    print(f"\nĐã xuất: {curves_csv} & {ablation_csv}")


# ======================================================================
# 7. CHẾ ĐỘ 4: --batch 3 (TẬP CHẨN ĐOÁN 40 BÀI M01-M40)
# ======================================================================

def load_custom_40() -> List[Dict[str, str]]:
    if hasattr(ev, "DATASET_40") and len(ev.DATASET_40) >= 40:
        return list(ev.DATASET_40[:40])
    if hasattr(ev, "DATASET_20") and len(ev.DATASET_20) >= 40:
        return list(ev.DATASET_20[:40])
    fallback_json = os.path.join(SCRIPT_DIR, "..", "dualref-repo", "data", "custom_40_problems.json")
    if os.path.exists(fallback_json):
        return ev.load_dataset(fallback_json)[:40]
    raise RuntimeError("Không tìm thấy tập 40 bài M01-M40 trong evaluation.py hoặc custom_40_problems.json.")


def run_batch3_custom40(args: argparse.Namespace) -> None:
    items = load_custom_40()
    if args.limit:
        items = items[: args.limit]

    ckpt_path = "stage3_batch3_custom40.jsonl"
    records = load_checkpoint_jsonl(ckpt_path, resume=not args.no_resume)
    done = {(r["id"], r["method"], int(r.get("seed", args.seed))) for r in records}

    jobs = [
        ("DIRECT", lambda it: ev.run_logged(ev.run_direct, it, seed=args.seed)),
        ("SC@5", lambda it: ev.run_self_consistency(it, k=5, seed=args.seed)),
        ("PAL@5", lambda it: ev.run_pal(it, k=5, seed=args.seed, fallback_direct=False)),
        ("DUALREF_NOCRIT", lambda it: ev.run_logged(ev.run_v5, it, method="DUALREF_NOCRIT", seed=args.seed)),
        ("DUALREF", lambda it: ev.run_logged(ev.run_v5, it, method="DUALREF", seed=args.seed)),
    ]

    total = len(items) * len(jobs)
    print("=" * 96)
    print(f"BATCH 3 -- CUSTOM 40 DIAGNOSTIC ({len(items)} bài x {len(jobs)} cấu hình = {total} lượt)")
    print(f"Checkpoint: {os.path.abspath(ckpt_path)} (đã xong {len(done)}/{total})")
    print("=" * 96)

    step_idx = 0
    for m_name, runner in jobs:
        for item in items:
            step_idx += 1
            key = (item["id"], m_name, args.seed)
            if key in done:
                continue
            r = runner(item)
            ev.append_jsonl_checkpoint(ckpt_path, r)
            records.append(r)
            done.add(key)
            print(
                f"[{step_idx}/{total}] {m_name:<15} {item['id']} | pred={r['predicted_answer']} | "
                f"GT={r['ground_truth']} | correct={r['correct']} | calls={r['llm_calls']} | "
                f"tok={r['total_tokens']} | {r.get('stop_reason', '')}"
            )

    df = pd.DataFrame(records)
    method_order = ["DIRECT", "SC@5", "PAL@5", "DUALREF_NOCRIT", "DUALREF"]
    summary_df = summarize_methods(df, method_order=method_order)

    print("\n" + "=" * 96)
    print("KẾT QUẢ BATCH 3 -- TẬP CHẨN ĐOÁN 40 BÀI (M01-M40)")
    print("=" * 96)
    print(summary_df.to_string(index=False))

    dual_df = df[df["method"] == "DUALREF"]
    if not dual_df.empty and "stop_reason" in dual_df.columns:
        print("\nPhân bố stop_reason của DUALREF:")
        for sr, cnt in dual_df["stop_reason"].value_counts().items():
            print(f"  - {sr:<22}: {cnt}/{len(dual_df)} ({100.0*cnt/len(dual_df):.1f}%)")

    paired_df = print_paired_comparison(df, "DUALREF", ["DUALREF_NOCRIT", "SC@5", "PAL@5", "DIRECT"])
    summary_df.to_csv("stage3_batch3_custom40_summary.csv", index=False, encoding="utf-8-sig")
    if not paired_df.empty:
        paired_df.to_csv("stage3_batch3_custom40_paired.csv", index=False, encoding="utf-8-sig")
    print("\nĐã xuất: stage3_batch3_custom40_summary.csv")


# ======================================================================
# 8. CLI ENTRYPOINT
# ======================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DualRef Stage 3 Experiment Runner (MATH-500 & Custom Datasets)")
    ap.add_argument("--regrade-gsm8k", action="store_true", help="Chấm lại result_gsm8k_raw.csv bằng parser mới")
    ap.add_argument("--gsm8k-csv", default=None, help="Đường dẫn tùy chọn tới result_gsm8k_raw.csv")
    ap.add_argument("--batch", type=int, choices=[1, 2, 3], default=None,
                    help="1: Full dataset (DIRECT vs DUALREF) | 2: Stratified curves & ablation | 3: Custom-40")
    ap.add_argument("--dataset", default=None,
                    help="Đường dẫn tệp dữ liệu (.jsonl/.json/.csv), vd: gsm8k_test_1000.jsonl (bỏ trống = MATH-500)")
    ap.add_argument("--math500-cache", default="math500_test.jsonl", help="File cache MATH-500 JSONL")
    ap.add_argument("--sample-n", type=int, default=300,
                    help="Số bài lấy mẫu phân tầng cho Batch 2 (mặc định: 300)")
    ap.add_argument("--limit", type=int, default=None, help="Giới hạn N bài đầu (dùng để chạy thử nhanh)")
    ap.add_argument("--seed", type=int, default=0, help="Seed mặc định cho Direct/DualRef")
    ap.add_argument("--no-resume", action="store_true", help="Xóa checkpoint cũ và chạy lại từ đầu")
    ap.add_argument("--base-url", default=ev.BASE_URL, help="Endpoint LM Studio (mặc định http://127.0.0.1:1234/v1)")
    ap.add_argument("--generator", default=ev.GENERATOR_MODEL, help="Tên model Generator")
    ap.add_argument("--critic", default="dualref_critic_q4_k_m_v2", help="API Identifier của Critic trên LM Studio")
    ap.add_argument("--workers", type=int, default=1, help="Số mẫu sinh song song (1 = tuần tự)")
    ap.add_argument("--no-loop-guard", action="store_true")
    ap.add_argument("--gen-max-tokens", type=int, default=2048)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.regrade_gsm8k and args.batch is None:
        print("Vui lòng chọn --regrade-gsm8k hoặc --batch {1,2,3}. Chạy --help để xem chi tiết.")
        sys.exit(1)

    if args.regrade_gsm8k:
        run_regrade_gsm8k(args.gsm8k_csv)

    if args.batch is not None:
        verify_llm_server(args.base_url, args.generator, args.critic)
        ev.CRITIC_MODEL = args.critic
        ev.VERIFIER_MODEL = args.critic
        ev.PARALLEL_WORKERS = args.workers
        ev.LOOP_GUARD = not args.no_loop_guard
        ev.DIRECT_MAX_TOKENS = args.gen_max_tokens
        if args.batch == 1:
            run_batch1(args)
        elif args.batch == 2:
            run_batch2(args)
        elif args.batch == 3:
            run_batch3_custom40(args)


if __name__ == "__main__":
    main()