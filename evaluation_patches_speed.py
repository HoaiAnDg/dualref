# -*- coding: utf-8 -*-
"""
evaluation_patches_speed.py -- Bản vá TỐC ĐỘ + sửa lỗi cho DualRef (nạp SAU evaluation_patches.py).

KHÔNG đổi: K0=3, bước mở rộng 3->5->7->9, K_max=9, luật dừng (CROSS_VALIDATED / PAL_CROSS_CONFIRMED /
PAL_DOMINANT / MARGIN), trọng số bỏ phiếu, ngưỡng Critic, nhiệt độ lấy mẫu, prompt, số lần debug PAL.

Đổi (đều áp dụng ĐỒNG NHẤT cho mọi phương pháp dùng chung _direct_once / call_llm):
  1. DIRECT_MAX_TOKENS 3072 -> 2048 (một mẫu CoT bị lặp không còn đốt 3072 token).
  2. Loop-guard: stream + ngắt sớm khi văn bản lặp vô hạn (tính là "truncated").
  3. Không gọi extract_answer_llm trên đầu ra bị cắt cụt (tránh đáp án rác + 1 lượt gọi thừa).
  4. clean_final_answer không còn nuốt "**" (lũy thừa) trong biểu thức -> sửa key PAL/CoT lệch nhau.
  5. Sinh song song các mẫu độc lập trong cùng một vòng (ThreadPool) -- cần LM Studio cho phép
     nhiều request đồng thời; kết quả/seed/thứ tự mẫu giữ nguyên.
  6. (BATCH 2) Song song hóa _collect_sc_samples / _collect_pal_samples (SC_POOL12, PAL_POOL12):
     - 12 mẫu chạy bằng ThreadPoolExecutor(max_workers = min(--workers, 4 slot LM Studio, k));
       kết quả luôn xếp đúng chỉ số mẫu 0..11 (không theo thứ tự luồng xong trước) => SC@k/PAL@k
       (k = 1,3,5,7,9,12) rút offline từ samples[:k] giữ nguyên ý nghĩa.
     - llm_calls / prompt / completion / total_tokens / truncated_calls của TỪNG mẫu được đo bằng
       bộ đếm theo luồng (thread-local), không dùng hiệu số bộ đếm toàn cục (sẽ lẫn giữa các luồng).
     - Seed, nhiệt độ, prompt, max_tokens, self-debug PAL, schema trace_json: GIỮ NGUYÊN
       (chỉ THÊM trường "truncated_calls" cho mỗi mẫu).
     - workers <= 1 (hoặc GEN_STREAMING=False): chạy tuần tự y như bản cũ.
     - PAL: giới hạn số subprocess Python/SymPy chạy đồng thời (semaphore) để timeout 8s không
       bị "ăn" bởi tranh chấp CPU; stdin=DEVNULL; mỗi lần chạy có thư mục tạm riêng.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from collections.abc import Callable
    from openai import APIConnectionError, APITimeoutError, InternalServerError

    LLM_CALLS: int
    LLM_STATS: Counter[str]
    TOKEN_STATS: Dict[str, int]
    DEFAULT_GEN_STOP: List[str]
    GENERATOR_MODEL: str
    MAX_EMPTY_RETRIES: int
    DIRECT_SYSTEM_PROMPT: str
    DIRECT_SYSTEM_PROMPT_EN: str
    LAST_FINISH_REASON: Optional[str]
    EXTRACT_FALLBACK: bool
    SC_TEMPERATURE: float
    V5_SPECS: Dict[str, bool]
    PAL_CONFIRM_REQUIRES_VALID_COT: bool
    V5_MARGIN: float
    PAL_TIMEOUT: int
    _FORBIDDEN_ANSWERS: set[str]

    get_client: Callable[..., Any]
    _record_tokens: Callable[..., Any]
    call_llm: Callable[..., str]
    strip_function_call_hallucination: Callable[[str], str]
    strip_tool_json: Callable[[str], str]
    extract_final_answer: Callable[[str], Optional[str]]
    extract_answer_llm: Callable[[str, str], Optional[str]]
    safe_sympy_check_equalities: Callable[[str], Tuple[str, str]]
    answer_key: Callable[[Optional[str]], Optional[str]]
    extract_boxed_balanced: Callable[[str], Optional[str]]
    _v5_sample_pal: Callable[..., Dict[str, Any]]
    _v5_route: Callable[[str], str]
    _v5_tally: Callable[..., Tuple[Any, Any, Any, Any, Any]]
    _critic_whole: Callable[..., Dict[str, Any]]
    format_solution_explanation: Callable[..., str]
    evaluate_answer_match: Callable[[Optional[str], str], bool]
    _pal_static_check: Callable[[str], Optional[str]]
    token_snapshot: Callable[[], Dict[str, int]]
    token_delta: Callable[[Dict[str, int]], Dict[str, int]]
    run_sc_pool: Callable[..., Dict[str, Any]]
    run_pal_pool: Callable[..., Dict[str, Any]]

# ----------------------------------------------------------------------
# CẤU HÌNH
# ----------------------------------------------------------------------
DIRECT_MAX_TOKENS = 2048          # áp dụng cho DIRECT, SC@k, nhánh CoT của DualRef
GEN_STREAMING = True              # gọi model sinh bằng stream (bắt buộc cho loop-guard)
LOOP_GUARD = True
LOOP_GUARD_MIN_CHARS = 800        # chỉ kiểm tra khi đã sinh >= 800 ký tự
LOOP_GUARD_UNIT = 80             # đoạn cuối 80 ký tự ...
LOOP_GUARD_REPEATS = 3            # ... xuất hiện >= 3 lần => coi là lặp (đã thử trên log: 1/1 ca lặp, 0/8 CoT bình thường)
LOOP_GUARD_STEP = 200             # kiểm tra mỗi 200 ký tự mới
SKIP_EXTRACT_ON_TRUNCATED = True
PARALLEL_WORKERS = 3              # = số mẫu độc lập chạy đồng thời (đặt 1 để tắt)

# --- Batch 2 (SC_POOL12 / PAL_POOL12) ---
POOL_PARALLEL = True              # False: ép _collect_*_samples chạy tuần tự
LMSTUDIO_MAX_SLOTS = 4            # = "Max Concurrent Predictions" của LM Studio (mỗi slot 4096 token)
PAL_EXEC_MAX_CONCURRENT = 2       # số subprocess Python/SymPy PAL chạy cùng lúc (import sympy ~1-2s CPU)

_GEN_LOCK = threading.Lock()
_TL = threading.local()           # finish_reason / truncated / bộ đếm lượt gọi+token theo từng luồng
_PAL_EXEC_SEM = threading.Semaphore(max(1, int(PAL_EXEC_MAX_CONCURRENT)))


# ----------------------------------------------------------------------
# Bộ đo theo luồng: mỗi mẫu chạy trong MỘT luồng, nên đếm riêng theo luồng cho kết quả đúng
# (hiệu số LLM_CALLS / TOKEN_STATS toàn cục sẽ lẫn lộn khi nhiều mẫu chạy đồng thời).
# ----------------------------------------------------------------------
def _meter_begin():
    _TL.calls = 0
    _TL.ptok = 0
    _TL.ctok = 0
    _TL.trunc = 0
    _TL.finish = None          # tránh dùng nhầm finish_reason của mẫu trước trên cùng luồng pool
    _TL.truncated = False


def _meter_end():
    p, c = int(getattr(_TL, "ptok", 0)), int(getattr(_TL, "ctok", 0))
    return {
        "llm_calls": int(getattr(_TL, "calls", 0)),
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "truncated_calls": int(getattr(_TL, "trunc", 0)),
    }


# ----------------------------------------------------------------------
# 1+2. call_llm cho model sinh: stream + loop-guard, an toàn đa luồng
# ----------------------------------------------------------------------
def _is_looping(text: str) -> bool:
    n = len(text)
    if n < 400:
        return False
    # Bắt vòng lặp câu ngắn (ví dụ: "Now, we can use the fact that..." dài ~35-50 ký tự lặp >= 4 lần)
    if text.count(text[-45:]) >= 4:
        return True
    # Bắt vòng lặp đoạn dài (80 ký tự lặp >= 3 lần)
    if n >= LOOP_GUARD_MIN_CHARS and text.count(text[-LOOP_GUARD_UNIT:]) >= LOOP_GUARD_REPEATS:
        return True
    return False


_STREAM_USAGE = True
_USE_REP_PENALTY = True


def _gen_call(messages, model, temperature, max_tokens, max_retries=3, seed=None, stop=None):
    global LLM_CALLS, _STREAM_USAGE, _USE_REP_PENALTY
    if stop is None:
        stop = list(DEFAULT_GEN_STOP)
    delay = 1.5
    for attempt in range(1, max_retries + 1):
        with _GEN_LOCK:
            LLM_CALLS += 1
        _TL.calls = getattr(_TL, "calls", 0) + 1
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0.20,
                presence_penalty=0.05,
                stream=True,
            )
            if _USE_REP_PENALTY:
                kwargs["extra_body"] = {"repetition_penalty": 1.08}
            if _STREAM_USAGE:
                kwargs["stream_options"] = {"include_usage": True}
            if seed is not None:
                kwargs["seed"] = seed
            if stop:
                kwargs["stop"] = stop
            stream = get_client().chat.completions.create(**kwargs)
            parts, finish, usage, n_chars, next_check = [], None, None, 0, 400
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                delta = getattr(ch.delta, "content", None)
                if delta:
                    parts.append(delta)
                    n_chars += len(delta)
                    if LOOP_GUARD and n_chars >= next_check:
                        next_check = n_chars + LOOP_GUARD_STEP
                        if _is_looping("".join(parts)):
                            finish = "repeat"
                            break
                if getattr(ch, "finish_reason", None):
                    finish = ch.finish_reason
            try:
                stream.close()
            except Exception:
                pass
            content = "".join(parts).strip()
            eff_finish = "length" if finish == "repeat" else finish
            _TL.finish = eff_finish
            with _GEN_LOCK:
                LLM_STATS["finish_" + str(eff_finish)] += 1
                if finish == "repeat":
                    LLM_STATS["loop_guard_abort"] += 1
                p0, c0 = TOKEN_STATS["prompt"], TOKEN_STATS["completion"]
                _record_tokens(SimpleNamespace(usage=usage), messages, model, content)
                _TL.ptok = getattr(_TL, "ptok", 0) + (TOKEN_STATS["prompt"] - p0)
                _TL.ctok = getattr(_TL, "ctok", 0) + (TOKEN_STATS["completion"] - c0)
                if eff_finish == "length":
                    _TL.trunc = getattr(_TL, "trunc", 0) + 1
            if content:
                return content
            with _GEN_LOCK:
                LLM_STATS["empty"] += 1
            if attempt <= MAX_EMPTY_RETRIES:
                temperature = max(temperature, 0.3)
                time.sleep(0.5)
                continue
            return ""
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            print(f"[API retry {attempt}/{max_retries}] {e}")
            time.sleep(delay)
            delay *= 1.5
        except Exception as e:
            err_low = str(e).lower()
            if _STREAM_USAGE and "stream_options" in err_low:
                _STREAM_USAGE = False
                continue
            if _USE_REP_PENALTY and ("repetition_penalty" in err_low or "extra_body" in err_low):
                _USE_REP_PENALTY = False
                continue
            print(f"[API ERROR] {type(e).__name__}: {e}")
            break
    return ""


if "_orig_call_llm_before_speed_patch" not in globals():
    _orig_call_llm_before_speed_patch = call_llm


def call_llm(messages, model, temperature, max_tokens, max_retries=3, seed=None, stop=None):
    if GEN_STREAMING and model == GENERATOR_MODEL:
        return _gen_call(messages, model, temperature, max_tokens, max_retries, seed, stop)
    # Critic (completions/alpaca) vẫn đi đường cũ; chỉ được gọi từ luồng chính.
    return _orig_call_llm_before_speed_patch(
        messages, model, temperature, max_tokens, max_retries=max_retries, seed=seed, stop=stop)


# ----------------------------------------------------------------------
# 3. _direct_once: không "cứu" đáp án từ đầu ra bị cắt cụt
# ----------------------------------------------------------------------
def _direct_once(problem, temperature, seed, system_prompt=None):
    nudge, raw = "", ""
    for attempt in range(3):
        raw0 = call_llm(
            [{"role": "system", "content": system_prompt or DIRECT_SYSTEM_PROMPT},
             {"role": "user", "content": problem + nudge}],
            GENERATOR_MODEL,
            temperature=temperature if attempt == 0 else max(temperature, 0.4),
            max_tokens=DIRECT_MAX_TOKENS,
            seed=None if attempt else seed,
            stop=list(DEFAULT_GEN_STOP),
        )
        raw = strip_function_call_hallucination(strip_tool_json(raw0 or "")).strip()
        if raw:
            break
        with _GEN_LOCK:
            LLM_STATS["direct_toolcall_retry"] += 1
        nudge = ("\n\n(Compute step by step in plain text and mathematical formulas, "
                 "one idea per line, then write FINAL_ANSWER; do not call any function.)")
    finish = getattr(_TL, "finish", None)
    if finish is None:                       # đường không-stream (GEN_STREAMING=False)
        finish = LAST_FINISH_REASON
    truncated = (finish == "length")
    _TL.truncated = truncated
    if truncated:
        with _GEN_LOCK:
            LLM_STATS["direct_truncated_last"] += 1
    pred = extract_final_answer(raw)
    if pred is None and raw and EXTRACT_FALLBACK and not (SKIP_EXTRACT_ON_TRUNCATED and truncated):
        pred = extract_answer_llm(problem, raw)
    return raw, pred


def _v5_sample_cot(problem, idx, cot_idx, seed, lang="en"):
    raw, pred = _direct_once(problem, 0.2 if idx == 0 else SC_TEMPERATURE, seed * 1000 + idx,
                             system_prompt=DIRECT_SYSTEM_PROMPT_EN)
    truncated = bool(getattr(_TL, "truncated", False))
    loc, msg = safe_sympy_check_equalities(raw or "")
    full_text = (raw or "").strip()
    return {
        "kind": "COT-en", "answer": pred, "key": answer_key(pred), "exec_ok": True,
        "error": "", "local": loc, "local_msg": msg, "debug_used": False,
        "truncated": truncated,
        "text": full_text, "critic_text": full_text[-2500:],
    }


# ----------------------------------------------------------------------
# 4. clean_final_answer: chỉ bỏ "**" khi nó là cặp in đậm bao TOÀN BỘ đáp án
#    (lỗi cũ: "R**2" -> "R2", "a**2/3" -> "a2/3" => key PAL không khớp key CoT)
# ----------------------------------------------------------------------
def clean_final_answer(s, for_gt=False):
    if s is None:
        return None
    t = strip_function_call_hallucination(str(s))
    t = t.strip().strip("`$ \t\n")
    if not t:
        return None

    boxed = extract_boxed_balanced(t)
    if boxed is not None:
        t = boxed.strip()
    m_bold = re.fullmatch(r"\*\*(.+?)\*\*", t, flags=re.S)       # **42** -> 42 ; R**2 giữ nguyên
    if m_bold:
        t = m_bold.group(1).strip()
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", t).strip()
    t = re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}", r"\1", t).strip()
    t = re.sub(r"\^\s*\{?\\circ\}?|°", "", t).strip()
    t = t.replace("\\$", "").replace("\\%", "").strip()
    t = re.split(r"\s*\((?=\s*[^\W\d_]+\s+[^\W\d_]+\s+[^\W\d_]+)", t, maxsplit=1)[0].strip()
    if "=" in t and not re.search(r"[<>!]=", t):
        lhs, rhs = t.rsplit("=", 1)
        if rhs.strip() and len(lhs.strip()) <= 12:
            t = rhs.strip()
    m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?(?:/\d+)?)\s+[^\W\d_]{2,}.*", t)
    if m:
        t = m.group(1)
    t = t.rstrip(" .;,").strip()
    if not t:
        return None
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
        t = t.replace(",", "")
    if for_gt:
        return t
    if '"' in t or len(t) > 120 or len(t.split()) > 12:
        return None
    if len(t) == 1 and t.isalpha() and t.upper() not in "ABCDE":
        return None
    if t.lower() in _FORBIDDEN_ANSWERS:
        return None
    if t.startswith("<") and t.endswith(">"):
        return None
    return t


# ----------------------------------------------------------------------
# 5. run_v5: logic GIỮ NGUYÊN, chỉ sinh song song các mẫu độc lập trong một vòng
# ----------------------------------------------------------------------
def _spawn_samples(problem, n_existing, cot_n, pal_n, target_cot, target_pal, seed):
    jobs, idx, c, p = [], n_existing, cot_n, pal_n
    while c < target_cot or p < target_pal:          # cùng thứ tự/idx như vòng lặp tuần tự cũ
        if c < target_cot:
            jobs.append(("COT", idx, c)); idx += 1; c += 1
        if p < target_pal:
            jobs.append(("PAL", idx, p)); idx += 1; p += 1

    def _do(job):
        kind, i, ci = job
        if kind == "COT":
            return _v5_sample_cot(problem, i, ci, seed, lang="en")
        return _v5_sample_pal(problem, i, seed, max_debug=1)

    workers = max(1, min(int(PARALLEL_WORKERS), len(jobs)))
    if workers == 1:
        results = [_do(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_do, jobs))          # map giữ đúng thứ tự job
    return results, c, p


def run_v5(item, method="V5", seed=0):
    use_critic = V5_SPECS.get(method, True)
    start, calls0 = time.time(), LLM_CALLS
    problem = item["problem"]
    single = "COT" if method == "V5_COT" else ("PAL" if method == "PAL_ONLY" else None)
    if single == "COT":
        route = "COT_ONLY"
    elif single == "PAL":
        route = "PAL_ONLY"
    else:
        route = _v5_route(problem)
    samples = []
    critic_calls = 0
    stop_reason = "KMAX"
    cot_n = pal_n = 0

    if single == "COT":
        target_cot, target_pal = 3, 0
    elif single == "PAL":
        target_cot, target_pal = 0, 3
    elif route == "PAL_FIRST":
        target_pal, target_cot = 2, 1
    else:
        target_cot, target_pal = 2, 1
    step_cot = 0 if single == "PAL" else (2 if single == "COT" else 1)
    step_pal = 0 if single == "COT" else (2 if single == "PAL" else 1)
    max_samples = 8

    while True:
        new, cot_n, pal_n = _spawn_samples(problem, len(samples), cot_n, pal_n,
                                           target_cot, target_pal, seed)
        samples.extend(new)

        ranked, support, best, cross_keys, win_cands = _v5_tally(samples, route, use_critic=use_critic)

        if cross_keys:
            top_cross = max(cross_keys, key=lambda k: support[k])
            if support[top_cross] >= 2:
                stop_reason = "CROSS_VALIDATED"
                break

        if single != "COT":
            pal_votes = Counter(sm["key"] for sm in samples
                                if sm["kind"] == "PAL" and sm["exec_ok"] and sm["key"])
            if pal_votes:
                top_pal_key, top_pal_count = pal_votes.most_common(1)[0]
                if top_pal_count >= 2:
                    if PAL_CONFIRM_REQUIRES_VALID_COT:
                        cot_pool = [s for s in samples if s["kind"].startswith("COT")
                                    and s["local"] != "REJECT" and s["key"]]
                    else:
                        cot_pool = [s for s in samples if s["kind"].startswith("COT")]
                    if any(s["key"] == top_pal_key for s in cot_pool):
                        stop_reason = "PAL_CROSS_CONFIRMED"
                        break
                    if top_pal_count >= 3:
                        stop_reason = "PAL_DOMINANT"
                        break

        if use_critic:
            for sm in samples:
                if (sm["kind"].startswith("COT") and "critic" not in sm
                        and not sm.get("truncated")
                        and sm["key"] is not None and sm["local"] != "REJECT"):
                    cr = _critic_whole(problem, sm["critic_text"], seed)
                    critic_calls += 1
                    sm["critic"] = cr["decision"] if cr["parse_ok"] else None

        ranked, support, best, cross_keys, win_cands = _v5_tally(samples, route, use_critic=use_critic)
        if ranked and support[ranked[0][0]] >= 2:
            w1 = ranked[0][1]
            w2 = ranked[1][1] if len(ranked) > 1 else 0.0
            if (w1 - w2) >= V5_MARGIN:
                stop_reason = "MARGIN"
                break

        if len(samples) >= max_samples:
            break
        target_cot += step_cot
        target_pal += step_pal

    winning_key = ranked[0][0] if ranked else None
    pred = best[winning_key][1] if winning_key else None
    win_cand = win_cands.get(winning_key) if winning_key else None
    sol_exp = format_solution_explanation(win_cand, samples, item["problem"])
    if pred is None:
        stop_reason = "NO_ANSWER"

    gt = item["ground_truth"]
    correct = evaluate_answer_match(pred, gt)
    oracle = any(evaluate_answer_match(sm["answer"], gt) for sm in samples if sm["answer"])
    pal_oracle = any(evaluate_answer_match(sm["answer"], gt)
                     for sm in samples if sm["kind"] == "PAL" and sm["answer"])
    local_vetoes = sum(1 for sm in samples if sm["local"] == "REJECT")
    w1 = ranked[0][1] if ranked else 0.0
    w2 = ranked[1][1] if len(ranked) > 1 else 0.0

    slim = [{k: (v[:4000] if isinstance(v, str) and k in ("text", "critic_text") else v)
             for k, v in sm.items()} for sm in samples]
    return {
        "id": item["id"], "category": item["category"], "method": method, "seed": seed,
        "ground_truth": gt, "predicted_answer": pred, "correct": int(correct),
        "oracle_any_correct": int(oracle), "pal_any_correct": int(pal_oracle),
        "route": route, "stop_reason": stop_reason, "n_samples": len(samples),
        "n_cot": sum(sm["kind"].startswith("COT") for sm in samples),
        "n_pal": sum(sm["kind"] == "PAL" for sm in samples),
        "pal_exec_ok": sum(sm["kind"] == "PAL" and sm["exec_ok"] for sm in samples),
        "debug_calls": sum(1 for sm in samples if sm.get("debug_attempted")),
        "debug_fixed": sum(1 for sm in samples if sm.get("debug_used")),
        "critic_calls": critic_calls,
        "critic_parse_fail": sum(1 for sm in samples if "critic" in sm and sm["critic"] is None),
        "local_vetoes": local_vetoes,
        "n_cot_truncated": sum(1 for sm in samples if sm.get("truncated")),
        "cross_validated": int(bool(cross_keys)),
        "top_weight": round(w1, 2), "margin": round(w1 - w2, 2),
        "vote_distribution": json.dumps({k: round(v, 2) for k, v in ranked}, ensure_ascii=False),
        "trace_json": json.dumps(slim, ensure_ascii=False),
        "solution_explanation": sol_exp,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }

# --- Tắt sinh lời giải tiếng Việt khi chạy benchmark ---
SKIP_SOLUTION_EXPLANATION = True

if "_orig_format_sol_exp" not in globals():
    _orig_format_sol_exp = format_solution_explanation

def format_solution_explanation(win_cand, samples, problem):
    if SKIP_SOLUTION_EXPLANATION:
        return (win_cand or {}).get("text", "") if isinstance(win_cand, dict) else ""
    return _orig_format_sol_exp(win_cand, samples, problem)


# ======================================================================
# 6. BATCH 2 -- song song hóa _collect_sc_samples / _collect_pal_samples
#    (SC_POOL12 & PAL_POOL12; run_sc_pool / run_pal_pool / eval_pool_at_k giữ nguyên)
# ======================================================================

# --- 6a. run_pal_code: logic y hệt evaluation_patches.py, chỉ thêm an toàn đa luồng ---
#   * LLM_STATS (Counter) cập nhật trong khóa (tránh mất lượt đếm khi nhiều luồng cùng ghi)
#   * semaphore giới hạn số subprocess đồng thời; timeout chỉ bắt đầu tính SAU khi giành được chỗ
#   * stdin=DEVNULL: tiến trình con không dùng chung stdin/console của tiến trình cha
#   * mỗi lần chạy có TemporaryDirectory riêng => không đụng file tạm giữa các luồng
def run_pal_code(code: str) -> Tuple[Optional[str], str]:
    err = _pal_static_check(code)
    if err:
        return None, err
    posix = (os.name == "posix")
    wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (%d, %d))\n" % (PAL_TIMEOUT, PAL_TIMEOUT)
    ) if posix else ""
    with _GEN_LOCK:
        LLM_STATS["pal_sandbox_" + ("posix_rlimit" if posix else "windows_timeout_only")] += 1
    extra = {"start_new_session": True} if posix else {}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(wrapper + code)
        try:
            with _PAL_EXEC_SEM:
                r = subprocess.run([sys.executable, "-B", path], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                                   timeout=PAL_TIMEOUT, cwd=d, **extra)
        except subprocess.TimeoutExpired:
            return None, "timeout"
    lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        return None, (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
    return lines[-1], ""


# --- 6b. công cụ chạy pool có thứ tự ---
def _pool_workers(k: int) -> int:
    """Số luồng cho pool; 1 = tuần tự. Không vượt số slot LM Studio, không vượt k."""
    if not (POOL_PARALLEL and GEN_STREAMING):         # đường không-stream không an toàn đa luồng
        return 1
    try:
        w = int(PARALLEL_WORKERS)
    except Exception:
        w = 1
    return max(1, min(w, int(LMSTUDIO_MAX_SLOTS), int(k)))


def _run_indexed(fn, k: int, workers: int):
    """Chạy fn(0..k-1) song song; trả list kết quả theo ĐÚNG chỉ số 0..k-1 (không theo thứ tự xong)."""
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pool")
    futures = [ex.submit(fn, i) for i in range(k)]      # futures[i] <-> mẫu i
    try:
        results = [f.result() for f in futures]         # duyệt theo i => thứ tự cố định
    except BaseException:                               # Ctrl+C: không đợi các mẫu còn xếp hàng
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return results


def _pool_error(kind: str, i: int, e: BaseException) -> str:
    msg = f"{type(e).__name__}: {e}"
    with _GEN_LOCK:
        LLM_STATS["pool_sample_error"] += 1
    print(f"[POOL {kind} sample {i} LỖI] {msg}\n{traceback.format_exc(limit=3)}")
    return msg[:300]


# --- 6c. một mẫu SC (CoT) / một mẫu PAL, đo bằng bộ đếm theo luồng ---
def _sc_sample_threaded(problem: str, i: int, seed: int, seed_mult: int) -> Dict[str, Any]:
    _meter_begin()
    t0, err, raw, pred = time.time(), "", "", None
    try:
        # LOOP_GUARD + DIRECT_MAX_TOKENS đã nằm trong _gen_call/_direct_once: mẫu CoT lặp vô hạn
        # bị ngắt sớm, chỉ chiếm 1 luồng tới lúc bị ngắt, không chặn các luồng còn lại.
        raw, pred = _direct_once(problem, SC_TEMPERATURE, seed * seed_mult + i)
    except Exception as e:
        err = _pool_error("SC", i, e)
    m = _meter_end()
    sm = {
        "idx": i,
        "kind": "COT",
        "answer": pred,
        "key": answer_key(pred),
        "exec_ok": not err,
        "llm_calls": m["llm_calls"],
        "prompt_tokens": m["prompt_tokens"],
        "completion_tokens": m["completion_tokens"],
        "total_tokens": m["total_tokens"],
        "truncated_calls": m["truncated_calls"],
        "elapsed_sec": round(time.time() - t0, 3),
        "raw_output": (raw or "")[:4000],
    }
    if err:
        sm["error"] = err
    return sm


def _pal_sample_threaded(problem: str, i: int, seed: int, max_debug: int) -> Dict[str, Any]:
    _meter_begin()
    t0, err, sm0 = time.time(), "", {}
    try:
        sm0 = _v5_sample_pal(problem, idx=i, seed=seed, max_debug=max_debug) or {}
    except Exception as e:
        err = _pool_error("PAL", i, e)
    m = _meter_end()
    return {
        "idx": i,
        "kind": "PAL",
        "answer": sm0.get("answer"),
        "key": sm0.get("key"),
        "exec_ok": bool(sm0.get("exec_ok")) and not err,
        "debug_attempted": bool(sm0.get("debug_attempted")),
        "debug_used": bool(sm0.get("debug_used")),
        "error": err or sm0.get("error", ""),
        "llm_calls": m["llm_calls"],
        "prompt_tokens": m["prompt_tokens"],
        "completion_tokens": m["completion_tokens"],
        "total_tokens": m["total_tokens"],
        "truncated_calls": m["truncated_calls"],
        "elapsed_sec": round(time.time() - t0, 3),
        "code": (sm0.get("text") or "")[:4000],
    }


# --- 6d. hai hàm bộ sinh mẫu (ĐÈ lên bản trong evaluation_patches.py) ---
def _collect_sc_samples(problem: str, k: int, seed: int, seed_mult: int = 1000) -> List[Dict[str, Any]]:
    workers = _pool_workers(k)
    if workers <= 1:                                   # tuần tự: như bản cũ (+ truncated_calls)
        samples = []
        for i in range(k):
            t0, c0, tok0 = time.time(), LLM_CALLS, token_snapshot()
            tr0 = LLM_STATS.get("finish_length", 0)
            raw, pred = _direct_once(problem, SC_TEMPERATURE, seed * seed_mult + i)
            td = token_delta(tok0)
            samples.append({
                "idx": i,
                "kind": "COT",
                "answer": pred,
                "key": answer_key(pred),
                "exec_ok": True,
                "llm_calls": LLM_CALLS - c0,
                "prompt_tokens": td["prompt_tokens"],
                "completion_tokens": td["completion_tokens"],
                "total_tokens": td["total_tokens"],
                "truncated_calls": LLM_STATS.get("finish_length", 0) - tr0,
                "elapsed_sec": round(time.time() - t0, 3),
                "raw_output": (raw or "")[:4000],
            })
        return samples
    samples = _run_indexed(lambda i: _sc_sample_threaded(problem, i, seed, seed_mult), k, workers)
    assert [sm["idx"] for sm in samples] == list(range(k))   # thứ tự 0..k-1 (SC@1,3,...,12)
    return samples


def _collect_pal_samples(problem: str, k: int, seed: int, max_debug: int = 1) -> List[Dict[str, Any]]:
    workers = _pool_workers(k)
    if workers <= 1:                                   # tuần tự: như bản cũ (+ truncated_calls)
        samples = []
        for i in range(k):
            t0, c0, tok0 = time.time(), LLM_CALLS, token_snapshot()
            tr0 = LLM_STATS.get("finish_length", 0)
            sm = _v5_sample_pal(problem, idx=i, seed=seed, max_debug=max_debug)
            td = token_delta(tok0)
            samples.append({
                "idx": i,
                "kind": "PAL",
                "answer": sm.get("answer"),
                "key": sm.get("key"),
                "exec_ok": bool(sm.get("exec_ok")),
                "debug_attempted": bool(sm.get("debug_attempted")),
                "debug_used": bool(sm.get("debug_used")),
                "error": sm.get("error", ""),
                "llm_calls": LLM_CALLS - c0,
                "prompt_tokens": td["prompt_tokens"],
                "completion_tokens": td["completion_tokens"],
                "total_tokens": td["total_tokens"],
                "truncated_calls": LLM_STATS.get("finish_length", 0) - tr0,
                "elapsed_sec": round(time.time() - t0, 3),
                "code": (sm.get("text") or "")[:4000],
            })
        return samples
    samples = _run_indexed(lambda i: _pal_sample_threaded(problem, i, seed, max_debug), k, workers)
    assert [sm["idx"] for sm in samples] == list(range(k))   # thứ tự 0..k-1 (PAL@1,3,...,12)
    return samples


# --- 6e. run_sc_pool / run_pal_pool: bản ghi cấp bài có n_truncated = tổng truncated_calls của các mẫu
#         (bản cũ luôn ghi 0 cho 2 tác vụ này). Mọi trường khác giữ nguyên.
def _with_pool_truncation(orig_fn):
    def _wrapped(item, max_k: int = 12, seed: int = 42):
        r = orig_fn(item, max_k=max_k, seed=seed)
        try:
            tr = json.loads(r.get("trace_json") or "[]")
            r["n_truncated"] = int(sum(int(sm.get("truncated_calls", 0) or 0) for sm in tr))
        except Exception:
            pass
        return r
    return _wrapped


if "_orig_run_sc_pool_before_speed_patch" not in globals():
    _orig_run_sc_pool_before_speed_patch = run_sc_pool
    _orig_run_pal_pool_before_speed_patch = run_pal_pool

run_sc_pool = _with_pool_truncation(_orig_run_sc_pool_before_speed_patch)
run_pal_pool = _with_pool_truncation(_orig_run_pal_pool_before_speed_patch)