# -*- coding: utf-8 -*-
"""
evaluation_patches.py  --  CÁC BẢN VÁ cho evaluation.py (DualRef, giai đoạn 3)

CÁCH DÙNG (chọn MỘT):
  (a) Không cần dán gì: run_stage3.py tự nạp file này vào namespace của `evaluation`
      (exec) -> mọi hàm/hằng bên dưới ĐÈ lên hàm cùng tên trong evaluation.py lúc chạy.
  (b) Dán đè thủ công: mỗi khối có dòng "REPLACES: ..." cho biết thay hàm/hằng nào
      trong evaluation.py (số dòng theo bản evaluation.py 2247 dòng đã gửi).

Mọi khối chỉ dùng tên đã có sẵn trong evaluation.py (re, json, math, sp, os, sys, time,
subprocess, tempfile, Counter, LLM_STATS, TOKEN_STATS, FINAL_ANSWER_TAG, ...), nên dán
nguyên văn được.  Khối chạy lặp lại nhiều lần vẫn an toàn (idempotent).
"""

import os
import re
import json
import math
import time
import sys
import subprocess
import tempfile
from collections import Counter
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import sympy as sp

if TYPE_CHECKING:
    from evaluation import (  # noqa: F401
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        GENERATOR_MODEL,
        FINAL_ANSWER_TAG,
        SC_TEMPERATURE,
        PAL_SYSTEM_PROMPT,
        PAL_DEBUG_SYSTEM_PROMPT,
        PAL_TIMEOUT,
        MAX_EMPTY_RETRIES,
        EXTRACT_FALLBACK,
        V5_MARGIN,
        DATASET_20,
        LLM_CALLS,
        LLM_STATS,
        TOKEN_STATS,
        get_client,
        _record_tokens,
        token_snapshot,
        token_delta,
        strip_tool_json,
        _FORBIDDEN_ANSWERS,
        safe_parse,
        normalize_math_string,
        _EQ_RUN,
        _balanced,
        _pal_static_check,
        extract_answer_llm,
        format_solution_explanation,
        _v5_route,
        _v5_sample_cot,
        _v5_tally,
        _critic_whole,
    )

# ======================================================================
# KHỐI 0 -- HẰNG SỐ MỚI / ĐỔI GIÁ TRỊ
# REPLACES: dòng 78 `DIRECT_MAX_TOKENS = 2500`  (+ thêm các hằng mới ngay bên dưới)
# ======================================================================
DIRECT_MAX_TOKENS = 3072

# Chuỗi dừng chống ảo giác "gọi hàm" của Llama-3.1 (cắt ngay khi sinh ra).
DEFAULT_GEN_STOP = [
    "<|python_tag|>",
    "Therefore, the JSON for a function call",
    "The JSON for a function call",
]
_HALLUCINATION_MARKERS = tuple(DEFAULT_GEN_STOP)
LAST_FINISH_REASON = None

# True: PAL_CROSS_CONFIRMED chỉ tính các mẫu CoT hợp lệ (không bị SymPy veto, có đáp án).
# False: hành vi cũ của evaluation.py (CoT bị veto vẫn có thể "xác nhận" PAL).
PAL_CONFIRM_REQUIRES_VALID_COT = True


# ======================================================================
# KHỐI 1 -- PROMPT ĐÁP ÁN CHUẨN HÓA
# REPLACES: dòng 382-389 `DIRECT_SYSTEM_PROMPT = ...`
#           dòng 454-459 `DIRECT_SYSTEM_PROMPT_EN = ...`
# ======================================================================
_ANSWER_RULES = (
    "Finish with exactly one final line:\n"
    "@TAG@ <answer>\n"
    "Write <answer> in simplest exact form: an integer, a fraction, or a LaTeX expression "
    "such as \\frac{\\sqrt{3}}{2}. You may wrap it as \\boxed{...}. "
    "The line must contain ONLY the answer, no explanation."
)

DIRECT_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step using plain text and "
    "mathematical formulas, concisely, one idea per line.\n"
    "You must reason in English. Never call a function and never output JSON or tool calls.\n\n"
    + _ANSWER_RULES
).replace("@TAG@", FINAL_ANSWER_TAG)

DIRECT_SYSTEM_PROMPT_EN = (
    "You are an expert mathematician. Solve the problem step by step with plain text and "
    "formulas, concisely, one idea per line.\n"
    "You must reason in English. Do not call any function and do not output JSON.\n\n"
    + _ANSWER_RULES
).replace("@TAG@", FINAL_ANSWER_TAG)


# ======================================================================
# KHỐI 2 -- call_llm  (thêm tham số `stop`; đếm LLM_CALLS theo từng request thật)
# REPLACES: dòng 554-606 `def call_llm(...)`
# ======================================================================
def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> str:
    global LLM_CALLS, LAST_FINISH_REASON
    # Mặc định mọi lời gọi tới model sinh đều có chuỗi dừng chống ảo giác tool-call.
    if stop is None and model == GENERATOR_MODEL:
        stop = list(DEFAULT_GEN_STOP)
    delay = 1.5
    for attempt in range(1, max_retries + 1):
        LLM_CALLS += 1  # mỗi request thật (kể cả retry) được tính một lượt
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if model == GENERATOR_MODEL:
                kwargs["frequency_penalty"] = 0.12
            if seed is not None:
                kwargs["seed"] = seed
            if stop:
                kwargs["stop"] = stop
            res = get_client().chat.completions.create(**kwargs)
            choice = res.choices[0] if res.choices else None
            msg = choice.message if choice is not None else None
            LAST_FINISH_REASON = getattr(choice, "finish_reason", None) if choice is not None else None
            if choice is not None:
                LLM_STATS["finish_" + str(LAST_FINISH_REASON)] += 1
            content = (msg.content or "").strip() if msg is not None else ""
            if not content and msg is not None and getattr(msg, "tool_calls", None):
                LLM_STATS["tool_calls"] += 1
                try:
                    content = json.dumps(
                        [{"name": tc.function.name, "parameters": json.loads(tc.function.arguments or "{}")}
                         for tc in msg.tool_calls], ensure_ascii=False)
                except Exception:
                    content = ""
            _record_tokens(res, messages, model, content)
            if content:
                return content
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
            print(f"[API ERROR] {type(e).__name__}: {e}")
            break
    return ""


# ======================================================================
# KHỐI 3 -- TIỆN ÍCH MỚI: cắt ảo giác function-call, bóc \boxed{} cân bằng ngoặc
# THÊM MỚI (dán ngay trước `def clean_final_answer`, khoảng dòng 700)
# ======================================================================
def strip_function_call_hallucination(text: Optional[str]) -> str:
    """Cắt toàn bộ phần văn bản từ dấu hiệu ảo giác tool-call trở đi."""
    if not text:
        return ""
    low = text.lower()
    cut = len(text)
    for mk in _HALLUCINATION_MARKERS:
        i = low.find(mk.lower())
        if i != -1:
            cut = min(cut, i)
    return text[:cut].rstrip()


def _find_matching_brace(s: str, i: int) -> int:
    """s[i] == '{' -> chỉ số '}' tương ứng (cân bằng ngoặc lồng nhau), hoặc -1."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def extract_boxed_balanced(text: Optional[str], last: bool = True) -> Optional[str]:
    """Nội dung của \\boxed{...} (hoặc \\fbox{...}), kể cả ngoặc lồng như \\frac{\\sqrt{3}}{2}."""
    if not text:
        return None
    found = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        i = m.end() - 1
        j = _find_matching_brace(text, i)
        if j != -1:
            found.append(text[i + 1:j])
    if not found:
        return None
    return found[-1] if last else found[0]


# ======================================================================
# KHỐI 4 -- clean_final_answer
# REPLACES: dòng 705-741 `def clean_final_answer(s)`
# ======================================================================
def clean_final_answer(s: Optional[str], for_gt: bool = False) -> Optional[str]:
    if s is None:
        return None
    t = strip_function_call_hallucination(str(s))
    t = t.strip().strip("`$ \t\n")
    if not t:
        return None

    boxed = extract_boxed_balanced(t)
    if boxed is not None:
        t = boxed.strip()
    t = re.sub(r"\*\*|\\\(|\\\)|\\\[|\\\]", "", t).strip()
    t = re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}", r"\1", t).strip()
    t = re.sub(r"\^\s*\{?\\circ\}?|°", "", t).strip()
    t = t.replace("\\$", "").replace("\\%", "").strip()

    # bỏ phần giải thích trong ngoặc ở cuối: "42 (vì ... )"
    t = re.split(
        r"\s*\((?=\s*[^\W\d_]+\s+[^\W\d_]+\s+[^\W\d_]+)", t, maxsplit=1
    )[0].strip()

    # "x = 5" -> "5" (áp dụng đối xứng cho cả đáp án và nhãn gốc)
    if "=" in t and not re.search(r"[<>!]=", t):
        lhs, rhs = t.rsplit("=", 1)
        if rhs.strip() and len(lhs.strip()) <= 12:
            t = rhs.strip()

    # "12 apples" -> "12"
    m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?(?:/\d+)?)\s+[^\W\d_]{2,}.*", t)
    if m:
        t = m.group(1)

    t = t.rstrip(" .;,").strip()
    if not t:
        return None
    # dấu phân cách hàng nghìn: 1,234,567 -> 1234567
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
        t = t.replace(",", "")

    if for_gt:
        return t
    # BỎ lệnh cấm dấu "{": đáp án LaTeX hợp lệ. Vẫn loại JSON/tool-call (có dấu nháy kép).
    if '"' in t or len(t) > 120 or len(t.split()) > 12:
        return None
    if len(t) == 1 and t.isalpha() and t.upper() not in "ABCDE":
        return None
    if t.lower() in _FORBIDDEN_ANSWERS:
        return None
    if t.startswith("<") and t.endswith(">"):
        return None
    return t


# ======================================================================
# KHỐI 5 -- extract_final_answer
# REPLACES: dòng 744-759 `def extract_final_answer(text)`
# ======================================================================
def extract_final_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = strip_function_call_hallucination(text)
    if not text.strip():
        return None
    matches = re.findall(rf"{re.escape(FINAL_ANSWER_TAG)}\s*(.+)", text, flags=re.I)
    if matches:
        c = clean_final_answer(matches[-1])
        if c:
            return c
    boxed = extract_boxed_balanced(text)
    if boxed:
        c = clean_final_answer(boxed)
        if c:
            return c
    alt = re.findall(
        r"(?:đáp\s*án|đáp\s*số|final\s+answer\s*(?:is)?|the\s+answer\s+is)\s*[:：]\s*(.+)"
        r"|(?:final\s+answer|the\s+answer)\s+is\s+(.+)",
        text, flags=re.I,
    )
    if alt:
        last = [g for g in alt[-1] if g][0]
        return clean_final_answer(last)
    return None


# ======================================================================
# KHỐI 6 -- chuẩn hóa biểu thức LaTeX (cân bằng ngoặc) + so khớp đáp án
# REPLACES: dòng 783-796 `def _normalize_math_expr(s)`
#           dòng 997-1013 `def evaluate_answer_match(...)`
# THÊM MỚI: _latex_group, _latex_conv, _outer_pair, _split_top_level, _expr_equal
# ======================================================================
def _latex_group(s: str, j: int):
    n = len(s)
    while j < n and s[j] == " ":
        j += 1
    if j >= n:
        return "", n
    if s[j] == "{":
        k = _find_matching_brace(s, j)
        if k == -1:
            return s[j + 1:], n
        return s[j + 1:k], k + 1
    if s[j] == "\\":
        m = re.match(r"\\[A-Za-z]+", s[j:])
        if m:
            return m.group(0), j + m.end()
    return s[j], j + 1


def _latex_conv(s: str) -> str:
    out = []
    i, n = 0, len(s)
    while i < n:
        if s.startswith("\\frac", i):
            a, j = _latex_group(s, i + 5)
            b, j = _latex_group(s, j)
            out.append(f"(({_latex_conv(a)})/({_latex_conv(b)}))")
            i = j
            continue
        if s.startswith("\\sqrt", i):
            j = i + 5
            idx = None
            if j < n and s[j] == "[":
                k = s.find("]", j)
                if k != -1:
                    idx = s[j + 1:k]
                    j = k + 1
            a, j = _latex_group(s, j)
            body = _latex_conv(a)
            out.append(f"(({body})**(1/({_latex_conv(idx)})))" if idx else f"sqrt({body})")
            i = j
            continue
        if s[i] == "^" and i + 1 < n and s[i + 1] == "{":
            k = _find_matching_brace(s, i + 1)
            if k != -1:
                out.append("^(" + _latex_conv(s[i + 2:k]) + ")")
                i = k + 1
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _normalize_math_expr(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)(?![A-Za-z])", "", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\q?quad", " ", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}", r"\1", s)
    for a, b in {
        "×": "*", "·": "*", "−": "-", "–": "-", "÷": "/",
        "π": "pi", "²": "^2", "³": "^3",
    }.items():
        s = s.replace(a, b)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/").replace("\\pi", "pi")
    s = s.replace("\\%", "").replace("\\$", "").replace("$", "").replace("%", "")
    s = re.sub(r"\^\s*\{?\\circ\}?|°", "", s)
    s = _latex_conv(s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"√\s*\(\s*([^()]+)\s*\)", r"sqrt(\1)", s)
    s = re.sub(r"√\s*([A-Za-z0-9_.]+)", r"sqrt(\1)", s)
    s = s.replace("căn", "sqrt")
    return s.strip()


def _outer_pair(t: str) -> bool:
    """t[0] là '(' hoặc '[' và ngoặc đóng tương ứng của nó chính là t[-1]."""
    if len(t) < 2 or t[0] not in "([" or t[-1] not in ")]":
        return False
    depth = 0
    for k, ch in enumerate(t):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return k == len(t) - 1
    return False


def _split_top_level(s: str):
    """'(3, \\frac{\\pi}{2})' -> ('(', ['3', '\\frac{\\pi}{2}'], ')'); không phải danh sách -> None."""
    t = s.strip()
    op = cl = ""
    inner = t
    if _outer_pair(t):
        op, cl, inner = t[0], t[-1], t[1:-1]
    parts, depth, cur = [], 0, []
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    parts = [p.strip() for p in parts]
    if len(parts) < 2 or any(not p for p in parts):
        return None
    return op, parts, cl


def _expr_equal(pred: str, gt: str) -> bool:
    p, g = safe_parse(pred), safe_parse(gt)
    if p is None or g is None:
        return False
    try:
        diff = sp.simplify(p - g)
        if diff == 0:
            return True
        if diff.is_number:
            return math.isclose(float(sp.N(diff)), 0.0, rel_tol=1e-9, abs_tol=1e-9)
    except Exception:
        pass
    return False


def evaluate_answer_match(predicted: Optional[str], ground_truth: str) -> bool:
    pred = clean_final_answer(predicted)
    if not pred:
        return False
    gt = clean_final_answer(ground_truth, for_gt=True) or str(ground_truth).strip()

    if normalize_math_string(pred) == normalize_math_string(gt):
        return True
    if _expr_equal(pred, gt):
        return True
    pa, ga = _split_top_level(pred), _split_top_level(gt)
    if pa is not None and ga is not None:
        (op1, items1, cl1), (op2, items2, cl2) = pa, ga
        if op1 == op2 and cl1 == cl2 and len(items1) == len(items2):
            return all(
                normalize_math_string(a) == normalize_math_string(b) or _expr_equal(a, b)
                for a, b in zip(items1, items2)
            )
    return False


# ======================================================================
# KHỐI 7 -- safe_sympy_check_equalities (dung sai số thực / làm tròn trung gian)
# REPLACES: dòng 850-903 `def safe_sympy_check_equalities(text)`
# THÊM MỚI: _DEC_RE, _equality_within_tolerance (dán ngay trước hàm)
# ======================================================================
_DEC_RE = re.compile(r"\d*\.(\d+)")


def _equality_within_tolerance(lhs: str, rhs: str, lv, rv, diff) -> bool:
    """True nếu 'đẳng thức sai' chỉ là sai số dấu phẩy động hoặc làm tròn thập phân trung gian."""
    try:
        d = abs(float(sp.N(diff)))
        scale = max(1.0, abs(float(sp.N(lv))), abs(float(sp.N(rv))))
    except Exception:
        return False
    if d <= 1e-6 * scale:                       # sai số IEEE-754 (0.1+0.2 = 0.3)
        return True
    places = [len(x) for x in _DEC_RE.findall(lhs + " " + rhs)]
    if places:                                  # có số thập phân -> làm tròn trung gian
        return d <= max(1e-6, min(10.0 ** (-max(places)), 1e-1))
    return False                                # không có số thập phân: so sánh chính xác


def safe_sympy_check_equalities(text: str) -> Tuple[str, str]:
    body = "\n".join(
        ln for ln in text.splitlines() if FINAL_ANSWER_TAG.lower() not in ln.lower()
    )
    for m in _EQ_RUN.finditer(body):
        run = m.group(0)
        if "=" not in run or re.search(r"[<>!]=|==", body[max(0, m.start() - 1): m.end() + 1]):
            continue
        lead = len(run) - len(run.lstrip())
        start = m.start() + lead
        end = m.start() + len(run.rstrip())
        imm_before = body[start - 1:start] if start > 0 else ""
        prev_nonspace = body[:start].rstrip()[-1:]
        after_txt = body[end:end + 8].lower().lstrip()
        glued_after = body[end:end + 1].isalpha()

        segs = run.split("=")
        if not segs[0].strip():
            segs = segs[1:]
        else:
            first = segs[0].strip()
            glued_before = imm_before.isalnum() or imm_before in ("_", ")")
            dangling_op = first[:1] in "+*/^)!" or (
                first[:1] == "-" and (prev_nonspace.isalnum() or prev_nonspace in ("_", ")"))
            )
            if glued_before or dangling_op:
                segs = segs[1:]
        if len(segs) < 2:
            continue
        if after_txt.startswith(("mod", "(mod", "%", "dư", "(dư", "≡")) or glued_after:
            segs = segs[:-1]
        if len(segs) < 2:
            continue

        cleaned = []
        for sg in segs:
            t = sg.strip()
            ok = (t and t[0] in "0123456789(-." and t[-1] in "0123456789)!." and _balanced(t))
            cleaned.append(t if ok else None)

        for lhs, rhs in zip(cleaned, cleaned[1:]):
            if lhs is None or rhs is None:
                continue
            lv, rv = safe_parse(lhs), safe_parse(rhs)
            if lv is None or rv is None:
                continue
            try:
                if lv.is_number and rv.is_number:
                    diff = sp.simplify(lv - rv)
                    if diff.is_zero is False:
                        if _equality_within_tolerance(lhs, rhs, lv, rv, diff):
                            continue
                        return "REJECT", f"Đẳng thức sai: {lhs} = {rhs}; SymPy cho {lv} != {rv}."
            except Exception:
                continue
    return "ACCEPT", "Không phát hiện đẳng thức số học sai rõ ràng."


# ======================================================================
# KHỐI 8 -- _direct_once (4096 token, stop chống tool-call, cắt ảo giác)
# REPLACES: dòng 1109-1136 `def _direct_once(...)`
# ======================================================================
def _direct_once(problem: str, temperature: float, seed: Optional[int],
                 system_prompt: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Một lời giải CoT trực tiếp. Dùng chung cho Direct, SC@k và nhánh CoT của DualRef."""
    nudge = ""
    raw = ""
    for attempt in range(3):
        raw0 = call_llm(
            [
                {"role": "system", "content": system_prompt or DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": problem + nudge},
            ],
            GENERATOR_MODEL,
            temperature=temperature if attempt == 0 else max(temperature, 0.4),
            max_tokens=DIRECT_MAX_TOKENS,
            seed=None if attempt else seed,
            stop=list(DEFAULT_GEN_STOP),
        )
        raw = strip_function_call_hallucination(strip_tool_json(raw0 or "")).strip()
        if raw:
            break
        LLM_STATS["direct_toolcall_retry"] += 1
        nudge = (
            "\n\n(Compute step by step in plain text and mathematical formulas, "
            "one idea per line, then write FINAL_ANSWER; do not call any function.)"
        )
    if LAST_FINISH_REASON == "length":
        LLM_STATS["direct_truncated_last"] += 1
    pred = extract_final_answer(raw)
    if pred is None and raw and EXTRACT_FALLBACK:
        pred = extract_answer_llm(problem, raw)
    return raw, pred


# ======================================================================
# KHỐI 9 -- sandbox PAL (ghi chế độ sandbox; kill cả nhóm tiến trình trên POSIX)
# REPLACES: dòng 1615-1636 `def run_pal_code(code)`
# ======================================================================
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
    # Trên Windows KHÔNG có giới hạn bộ nhớ 1 GB (chỉ có timeout): ghi lại để báo cáo trung thực.
    LLM_STATS["pal_sandbox_" + ("posix_rlimit" if posix else "windows_timeout_only")] += 1
    extra = {"start_new_session": True} if posix else {}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(wrapper + code)
        try:
            r = subprocess.run([sys.executable, "-B", path], capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=PAL_TIMEOUT, cwd=d, **extra)
        except subprocess.TimeoutExpired:
            return None, "timeout"
    lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        return None, (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
    return lines[-1], ""


# ======================================================================
# KHỐI 10 -- _v5_sample_pal (đếm lượt gọi gỡ lỗi ĐÃ THỬ, không chỉ lượt thành công)
# REPLACES: dòng 1782-1824 `def _v5_sample_pal(...)`
# ======================================================================
def _v5_sample_pal(problem: str, idx: int, seed: int, max_debug: int = 1) -> Dict[str, Any]:
    raw = call_llm(
        [{"role": "system", "content": PAL_SYSTEM_PROMPT}, {"role": "user", "content": problem}],
        GENERATOR_MODEL, temperature=0.2 if idx == 0 else SC_TEMPERATURE, max_tokens=700,
        seed=seed * 1000 + idx,
    )
    m = re.search(r"```(?:python)?\s*(.*?)```", raw or "", flags=re.S)
    code = m.group(1) if m else (raw or "")
    out, err = run_pal_code(code) if code.strip() else (None, "no code")
    debug_used = False
    debug_attempted = False

    if (out is None or err) and max_debug > 0 and code.strip():
        debug_attempted = True
        debug_prompt = (
            f"Problem:\n{problem}\n\n"
            f"Python code written:\n```python\n{code[:1200]}\n```\n\n"
            f"Execution error:\n{err}\n\n"
            f"Analyze the error and rewrite the complete corrected Python code. "
            f"Return only one ```python ... ``` block."
        )
        debug_raw = call_llm(
            [{"role": "system", "content": PAL_DEBUG_SYSTEM_PROMPT},
             {"role": "user", "content": debug_prompt}],
            GENERATOR_MODEL, temperature=0.1, max_tokens=750,
            seed=seed * 1000 + 800 + idx,
        )
        m_dbg = re.search(r"```(?:python)?\s*(.*?)```", debug_raw or "", flags=re.S)
        code_dbg = m_dbg.group(1) if m_dbg else (debug_raw or "")
        out_dbg, err_dbg = run_pal_code(code_dbg) if code_dbg.strip() else (None, "no code")
        if out_dbg is not None:
            code = code_dbg
            out = out_dbg
            err = ""
            debug_used = True
        else:
            err = f"{err} -> dbg: {err_dbg}"

    ans = clean_final_answer(out) if out else None
    return {
        "kind": "PAL", "answer": ans, "key": answer_key(ans), "exec_ok": out is not None,
        "error": err, "local": "ACCEPT", "local_msg": "", "debug_used": debug_used,
        "debug_attempted": debug_attempted,
        "text": code[:1500], "critic_text": f"Python program:\n{code[:1500]}\nProgram output: {ans}"
    }


# ======================================================================
# KHỐI 11 -- run_v5 + V5_SPECS (thêm DUALREF, DUALREF_NOCRIT, PAL_ONLY; K_max thống nhất 9)
# REPLACES: dòng 1744-1748 `V5_SPECS = {...}`  và  dòng 1881-2006 `def run_v5(...)`
# ======================================================================
V5_SPECS = {
    "V5": True,                 # = DUALREF (tên cũ)
    "V5_NOCRIT": False,         # = DUALREF_NOCRIT (tên cũ)
    "V5_COT": False,            # chỉ CoT
    "DUALREF": True,
    "DUALREF_NOCRIT": False,
    "PAL_ONLY": False,          # chỉ PAL (ablation mới)
}


def run_v5(item: Dict[str, str], method: str = "V5", seed: int = 0) -> Dict[str, Any]:
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
    samples: List[Dict[str, Any]] = []
    critic_calls = 0
    stop_reason = "KMAX"
    cot_n = 0
    pal_n = 0

    # Chỉ tiêu mẫu ban đầu K0 = 3
    if single == "COT":
        target_cot, target_pal = 3, 0
    elif single == "PAL":
        target_cot, target_pal = 0, 3
    elif route == "PAL_FIRST":
        target_pal, target_cot = 2, 1
    else:  # COT_FIRST
        target_cot, target_pal = 2, 1
    # Bước mở rộng: DualRef +1 CoT +1 PAL; biến thể một nhánh +2 mẫu cùng loại,
    # để mọi cấu hình đều đi 3 -> 5 -> 7 -> 9 mẫu.
    step_cot = 0 if single == "PAL" else (2 if single == "COT" else 1)
    step_pal = 0 if single == "COT" else (2 if single == "PAL" else 1)

    max_samples = 8   # Hướng 2A: giữ nguyên (tối đa 9 mẫu); KHÔNG ép trần 12 lượt gọi LLM

    while True:
        while cot_n < target_cot or pal_n < target_pal:
            idx = len(samples)
            if cot_n < target_cot:
                samples.append(_v5_sample_cot(problem, idx, cot_n, seed, lang="en"))
                cot_n += 1
            if pal_n < target_pal:
                samples.append(_v5_sample_pal(problem, idx, seed, max_debug=1))
                pal_n += 1

        ranked, support, best, cross_keys, win_cands = _v5_tally(samples, route, use_critic=use_critic)

        # Luật 1: CROSS_VALIDATED
        if cross_keys:
            top_cross = max(cross_keys, key=lambda k: support[k])
            if support[top_cross] >= 2:
                stop_reason = "CROSS_VALIDATED"
                break

        # Luật 2: PAL_CROSS_CONFIRMED / PAL_DOMINANT (không áp dụng cho biến thể chỉ-CoT)
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

        # Luật 3: Critic chấm các mẫu CoT hợp lệ chưa chấm
        if use_critic:
            for sm in samples:
                if (sm["kind"].startswith("COT") and "critic" not in sm
                        and sm["key"] is not None and sm["local"] != "REJECT"):
                    cr = _critic_whole(problem, sm["critic_text"], seed)
                    critic_calls += 1
                    sm["critic"] = cr["decision"] if cr["parse_ok"] else None

        # Luật 4: MARGIN
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
        "debug_calls": sum(1 for sm in samples if sm.get("debug_attempted")),   # số lượt gỡ lỗi ĐÃ THỬ
        "debug_fixed": sum(1 for sm in samples if sm.get("debug_used")),        # số lượt gỡ lỗi thành công
        "critic_calls": critic_calls,
        "critic_parse_fail": sum(1 for sm in samples if "critic" in sm and sm["critic"] is None),
        "local_vetoes": local_vetoes,
        "cross_validated": int(bool(cross_keys)),
        "top_weight": round(w1, 2), "margin": round(w1 - w2, 2),
        "vote_distribution": json.dumps({k: round(v, 2) for k, v in ranked}, ensure_ascii=False),
        "trace_json": json.dumps(slim, ensure_ascii=False),
        "solution_explanation": sol_exp,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }


# ======================================================================
# KHỐI 12 -- LƯU LOG CHUẨN HÓA CHO MỌI CẤU HÌNH
# THÊM MỚI (dán sau `def load_done`, khoảng dòng 2121)
# ======================================================================
STANDARD_COLUMNS = (
    "id", "category", "method", "seed", "ground_truth", "predicted_answer", "correct",
    "stop_reason", "llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
    "elapsed_sec", "trace_json",
)


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or v == ""


def standardize_record(r: Dict[str, Any], tok: Optional[Dict[str, int]] = None,
                       n_truncated: int = 0) -> Dict[str, Any]:
    """Bảo đảm MỌI bản ghi (Direct, SC@k, PAL@k, V5_COT, PAL_ONLY, DUALREF_NOCRIT, DUALREF)
    có đủ stop_reason, llm_calls, prompt/completion/total_tokens, elapsed_sec, trace_json."""
    if tok:
        r.update(tok)
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        r[k] = int(r.get(k, 0) or 0)
    r["llm_calls"] = int(r.get("llm_calls", 0) or 0)
    r["elapsed_sec"] = float(r.get("elapsed_sec", 0.0) or 0.0)
    r["n_truncated"] = int(n_truncated)
    m = str(r.get("method", ""))
    pred_none = _is_missing(r.get("predicted_answer"))
    if _is_missing(r.get("stop_reason")):
        if m == "DIRECT":
            r["stop_reason"] = "NO_ANSWER" if pred_none else "SINGLE_SAMPLE"
        elif m.startswith("SC@"):
            r["stop_reason"] = "NO_ANSWER" if pred_none else "K_SAMPLES_VOTE"
        elif m.startswith("PAL@"):
            r["stop_reason"] = ("FALLBACK_DIRECT" if r.get("used_fallback")
                                else ("NO_ANSWER" if pred_none else "K_SAMPLES_VOTE"))
        else:
            r["stop_reason"] = "UNKNOWN"
    if _is_missing(r.get("trace_json")):
        r["trace_json"] = json.dumps({
            "method": m, "predicted_answer": r.get("predicted_answer"),
            "n_truncated": n_truncated,
            "raw_output": str(r.get("raw_output") or r.get("solution_explanation") or "")[:6000],
        }, ensure_ascii=False)
    return r


def run_logged(fn, *args, **kwargs) -> Dict[str, Any]:
    """Chạy một runner (run_direct, run_v5, ...) và gắn token/lượt gọi/số lần bị cắt cụt."""
    tok0 = token_snapshot()
    trunc0 = LLM_STATS.get("finish_length", 0)
    r = fn(*args, **kwargs)
    trunc = LLM_STATS.get("finish_length", 0) - trunc0
    return standardize_record(r, token_delta(tok0), trunc)


def append_jsonl_checkpoint(path: str, r: Dict[str, Any]) -> None:
    """Ghi checkpoint từng bài: flush + fsync để không mất dữ liệu nếu bị ngắt."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


# ======================================================================
# CẬP NHẬT KHỐI 7 -- Dung sai tương đối theo scale cho số thập phân trung gian
# ======================================================================
def _equality_within_tolerance(lhs: str, rhs: str, lv, rv, diff) -> bool:
    """True nếu 'đẳng thức sai' chỉ là sai số dấu phẩy động hoặc làm tròn thập phân trung gian."""
    try:
        d = abs(float(sp.N(diff)))
        scale = max(1.0, abs(float(sp.N(lv))), abs(float(sp.N(rv))))
    except Exception:
        return False
    if d <= 1e-6 * scale:                       # sai số IEEE-754 (0.1 + 0.2 = 0.3)
        return True
    places = [len(x) for x in _DEC_RE.findall(lhs + " " + rhs)]
    if places:                                  # làm tròn p chữ số thập phân (kể cả sau nhân/chia)
        p = min(places)
        tol = max(1e-6 * scale, 0.5 * (10.0 ** (-p)) * scale)
        return d <= tol
    return False                                # không có số thập phân: so sánh chính xác


# ======================================================================
# KHỐI 13 -- answer_key (đồng bộ tách bộ số / tọa độ với Khối 6)
# REPLACES: dòng 1016-1031 `def answer_key(ans)`
# ======================================================================
def _single_answer_key(a: str) -> str:
    p = safe_parse(a)
    if p is not None:
        try:
            q = sp.simplify(p)
            if q.is_Float:
                r = sp.nsimplify(q, rational=True, tolerance=1e-9)
                q = r if (r.is_Rational and abs(r.q) <= 10000) else sp.Float(q, 8)
            return "S:" + str(q)
        except Exception:
            pass
    return "T:" + normalize_math_string(a)


def answer_key(ans: Optional[str]) -> Optional[str]:
    a = clean_final_answer(ans)
    if not a:
        return None
    sp_parts = _split_top_level(a)
    if sp_parts is not None:
        op, items, cl = sp_parts
        sub_keys = [_single_answer_key(it) for it in items]
        return f"L:{op}{','.join(sub_keys)}{cl}"
    return _single_answer_key(a)


# ======================================================================
# KHỐI 14 -- run_self_consistency, run_pal & BỘ SINH MẪU GỘP POOL (ĐỢT 2)
# REPLACES: dòng 1152-1174 `def run_self_consistency(...)`
#           dòng 1639-1677 `def run_pal(...)`
# ======================================================================
def _collect_sc_samples(problem: str, k: int, seed: int, seed_mult: int = 1000) -> List[Dict[str, Any]]:
    samples = []
    for i in range(k):
        t0, c0, tok0 = time.time(), LLM_CALLS, token_snapshot()
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
            "elapsed_sec": round(time.time() - t0, 3),
            "raw_output": (raw or "")[:4000],
        })
    return samples


def _collect_pal_samples(problem: str, k: int, seed: int, max_debug: int = 1) -> List[Dict[str, Any]]:
    samples = []
    for i in range(k):
        t0, c0, tok0 = time.time(), LLM_CALLS, token_snapshot()
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
            "elapsed_sec": round(time.time() - t0, 3),
            "code": (sm.get("text") or "")[:4000],
        })
    return samples


def run_self_consistency(item: Dict[str, str], k: int, seed: int = 0) -> Dict[str, Any]:
    start, calls0, tok0 = time.time(), LLM_CALLS, token_snapshot()
    samples = _collect_sc_samples(item["problem"], k, seed, seed_mult=1000)
    votes: Counter = Counter()
    exemplars: Dict[str, str] = {}
    raw_outputs: Dict[str, str] = {}
    for sm in samples:
        key = sm["key"]
        if key is not None:
            votes[key] += 1
            exemplars.setdefault(key, sm["answer"])
            raw_outputs.setdefault(key, sm["raw_output"])
    top_key = votes.most_common(1)[0][0] if votes else None
    pred = exemplars[top_key] if top_key else None
    sol_exp = raw_outputs[top_key] if top_key else ""
    correct = evaluate_answer_match(pred, item["ground_truth"])
    rec = {
        "id": item["id"], "category": item["category"], "method": f"SC@{k}", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred,
        "correct": int(correct), "stop_reason": "NO_ANSWER" if pred is None else "K_SAMPLES_VOTE",
        "vote_distribution": json.dumps(dict(votes), ensure_ascii=False),
        "trace_json": json.dumps(samples, ensure_ascii=False),
        "solution_explanation": sol_exp,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }
    return standardize_record(rec, token_delta(tok0))


def run_pal(item: Dict[str, str], k: int, seed: int = 0, fallback_direct: bool = False) -> Dict[str, Any]:
    start, calls0, tok0 = time.time(), LLM_CALLS, token_snapshot()
    samples = _collect_pal_samples(item["problem"], k, seed, max_debug=1)
    votes: Counter = Counter()
    exemplars: Dict[str, str] = {}
    codes: Dict[str, str] = {}
    for sm in samples:
        key = sm["key"]
        if sm["exec_ok"] and key is not None:
            votes[key] += 1
            exemplars.setdefault(key, sm["answer"])
            codes.setdefault(key, sm["code"])
    top_key = votes.most_common(1)[0][0] if votes else None
    pred = exemplars[top_key] if top_key else None
    sol_exp = f"```python\n{codes[top_key]}\n```\nProgram Output: {pred}" if top_key else ""
    used_fb = 0
    if pred is None and fallback_direct:
        raw_fb, pred = _direct_once(item["problem"], 0.1, seed)
        sol_exp = raw_fb
        used_fb = 1
    correct = evaluate_answer_match(pred, item["ground_truth"])
    stop_r = "FALLBACK_DIRECT" if used_fb else ("NO_ANSWER" if pred is None else "K_SAMPLES_VOTE")
    rec = {
        "id": item["id"], "category": item["category"], "method": f"PAL@{k}", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred, "correct": int(correct),
        "used_fallback": used_fb, "stop_reason": stop_r,
        "pal_exec_ok": sum(1 for sm in samples if sm["exec_ok"]),
        "debug_calls": sum(1 for sm in samples if sm["debug_attempted"]),
        "debug_fixed": sum(1 for sm in samples if sm["debug_used"]),
        "vote_distribution": json.dumps(dict(votes), ensure_ascii=False),
        "trace_json": json.dumps(samples, ensure_ascii=False),
        "solution_explanation": sol_exp,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }
    return standardize_record(rec, token_delta(tok0))


def run_sc_pool(item: Dict[str, str], max_k: int = 12, seed: int = 42) -> Dict[str, Any]:
    """Sinh một lần `max_k` mẫu SC để trích xuất offline đường cong SC@1..SC@max_k."""
    start, calls0, tok0 = time.time(), LLM_CALLS, token_snapshot()
    samples = _collect_sc_samples(item["problem"], max_k, seed, seed_mult=100)
    votes = Counter(sm["key"] for sm in samples if sm["key"] is not None)
    top_key = votes.most_common(1)[0][0] if votes else None
    pred = next((sm["answer"] for sm in samples if sm["key"] == top_key), None) if top_key else None
    rec = {
        "id": item["id"], "category": item["category"], "method": f"SC_POOL{max_k}", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred,
        "correct": int(evaluate_answer_match(pred, item["ground_truth"])),
        "stop_reason": "NO_ANSWER" if pred is None else "POOL_COMPLETE",
        "trace_json": json.dumps(samples, ensure_ascii=False),
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }
    return standardize_record(rec, token_delta(tok0))


def run_pal_pool(item: Dict[str, str], max_k: int = 12, seed: int = 42) -> Dict[str, Any]:
    """Sinh một lần `max_k` mẫu PAL (có self-debug) để trích xuất offline đường cong PAL@1..PAL@max_k."""
    start, calls0, tok0 = time.time(), LLM_CALLS, token_snapshot()
    samples = _collect_pal_samples(item["problem"], max_k, seed, max_debug=1)
    votes = Counter(sm["key"] for sm in samples if sm["exec_ok"] and sm["key"] is not None)
    top_key = votes.most_common(1)[0][0] if votes else None
    pred = next((sm["answer"] for sm in samples if sm["key"] == top_key), None) if top_key else None
    rec = {
        "id": item["id"], "category": item["category"], "method": f"PAL_POOL{max_k}", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred,
        "correct": int(evaluate_answer_match(pred, item["ground_truth"])),
        "stop_reason": "NO_ANSWER" if pred is None else "POOL_COMPLETE",
        "trace_json": json.dumps(samples, ensure_ascii=False),
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }
    return standardize_record(rec, token_delta(tok0))


def eval_pool_at_k(record: Dict[str, Any], ks: Tuple[int, ...] = (1, 3, 5, 7, 9, 12)) -> List[Dict[str, Any]]:
    """Tính offline kết quả bỏ phiếu đa số & chi phí cộng dồn tại các mốc k từ bản ghi *_POOL."""
    raw_trace = record.get("trace_json", "[]")
    samples = json.loads(raw_trace) if isinstance(raw_trace, str) else list(raw_trace)
    base_method = str(record.get("method", ""))
    prefix = "PAL" if base_method.startswith("PAL") else "SC"
    gt = str(record.get("ground_truth", ""))
    out = []
    for k in ks:
        if k > len(samples):
            continue
        sub = samples[:k]
        votes: Counter = Counter()
        exemplars: Dict[str, str] = {}
        for sm in sub:
            if sm.get("exec_ok", True) and sm.get("key") is not None:
                votes[sm["key"]] += 1
                exemplars.setdefault(sm["key"], sm["answer"])
        top_key = votes.most_common(1)[0][0] if votes else None
        pred = exemplars[top_key] if top_key else None
        out.append({
            "id": record["id"],
            "category": record.get("category", ""),
            "method": f"{prefix}@{k}",
            "k": k,
            "seed": record.get("seed", 0),
            "ground_truth": gt,
            "predicted_answer": pred,
            "correct": int(evaluate_answer_match(pred, gt)),
            "stop_reason": "NO_ANSWER" if pred is None else "K_SAMPLES_VOTE",
            "llm_calls": sum(int(sm.get("llm_calls", 0)) for sm in sub),
            "prompt_tokens": sum(int(sm.get("prompt_tokens", 0)) for sm in sub),
            "completion_tokens": sum(int(sm.get("completion_tokens", 0)) for sm in sub),
            "total_tokens": sum(int(sm.get("total_tokens", 0)) for sm in sub),
            "elapsed_sec": round(sum(float(sm.get("elapsed_sec", 0.0)) for sm in sub), 3),
        })
    return out


# ======================================================================
# KHỐI 15 -- DATASET_40, NẠP MATH-500 & HÀM ÁP DỤNG PATCH
# ======================================================================
DATASET_40 = DATASET_20[:40]

MATH500_URL = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/phase2_test.jsonl"


def load_math500(cache_path: str = "math500_phase2_test.jsonl", url: str = MATH500_URL) -> List[Dict[str, str]]:
    """Tải (và cache cục bộ) tập chuẩn MATH-500 của OpenAI PRM800K."""
    if not os.path.exists(cache_path):
        import urllib.request
        print(f"Đang tải MATH-500 từ {url} ...")
        urllib.request.urlretrieve(url, cache_path)
    items: List[Dict[str, str]] = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            q = row.get("question", row)
            problem = str(q.get("problem", "")).strip()
            gt = q.get("ground_truth_answer") or row.get("answer")
            if not gt:
                gt = extract_final_answer(q.get("ground_truth_solution", "")) or ""
            items.append({
                "id": f"MATH500_{i:04d}",
                "category": str(q.get("subject") or row.get("subject") or "unknown"),
                "level": str(q.get("level") or row.get("level") or ""),
                "problem": problem,
                "ground_truth": str(gt).strip(),
            })
    return items


def stratified_sample_math500(
    items: List[Dict[str, str]],
    n: int = 300,
    seed: int = 42,
    save_ids_path: Optional[str] = "math500_stratified_300_ids.txt",
) -> List[Dict[str, str]]:
    """Lấy mẫu phân tầng đúng `n` bài theo `category` với `seed` cố định."""
    import random
    if n >= len(items):
        chosen = list(items)
    else:
        rng = random.Random(seed)
        by_cat: Dict[str, List[Dict[str, str]]] = {}
        for it in items:
            by_cat.setdefault(it["category"], []).append(it)
        total = len(items)
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
        id_order = {it["id"]: idx for idx, it in enumerate(items)}
        chosen.sort(key=lambda x: id_order[x["id"]])
    if save_ids_path:
        with open(save_ids_path, "w", encoding="utf-8") as f:
            for it in chosen:
                f.write(it["id"] + "\n")
    return chosen


def apply_patches_to_evaluation(ev_module: Any, patch_path: str = "evaluation_patches.py") -> Any:
    """Nạp toàn bộ bản vá từ `patch_path` đè vào namespace của module `evaluation`."""
    with open(patch_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, patch_path, "exec"), ev_module.__dict__)
    return ev_module


# ======================================================================
# KHỐI 13 -- CRITIC ALPACA COMPLETIONS, CẮT GỌN INPUT & LOG CHI TIẾT LỖI 400
# ======================================================================
try:
    from openai import BadRequestError
except ImportError:
    class BadRequestError(Exception):
        pass

CRITIC_MODEL = "dualref_critic_q4_k_m_v2"
VERIFIER_MODEL = CRITIC_MODEL

# Giới hạn ký tự an toàn cho toàn bộ prompt gửi sang Critic (~1,600 tokens, an toàn cả khi n_ctx=2048)
# Nếu bạn đã nâng Context Length của Critic trên LM Studio lên 4096 hoặc 8192, có thể tăng lên 10000.
CRITIC_MAX_PROMPT_CHARS = 5200

PRM_CRITIC_INSTRUCTION = (
    "You are a Value Network Critic for a math solver. Evaluate the current step against the problem and history.\n"
    "1. Verify if the math calculation is correct and the logic moves efficiently toward the final answer.\n"
    "2. Score the step from 0.0 (completely wrong) to 1.0 (perfectly correct and optimal).\n"
    "Output EXACTLY in this format:\n"
    "Score: [number]\n"
    "<hint> [Optional brief explanation if score < 0.7] </hint>"
)

CRITIC_SYSTEM_PROMPT = PRM_CRITIC_INSTRUCTION
BON_CRITIC_SYSTEM_PROMPT = PRM_CRITIC_INSTRUCTION

ALPACA_CRITIC_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

_CRITIC_SCORE_RE = re.compile(r"Score\s*:\s*([0-9]+(?:\.[0-9]+)?)", flags=re.I)
_CRITIC_HINT_RE = re.compile(r"<hint>\s*(.*?)\s*</hint>", flags=re.I | re.S)


def _clean_problem_for_critic(problem_text: str) -> str:
    """Lược bỏ mã vẽ hình Asymptote [asy]...[/asy] gây tràn context ở các bài hình học."""
    cleaned = re.sub(
        r"\[asy\].*?\[/asy\]",
        "[Diagram omitted]",
        problem_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\\begin\{asy\}.*?\\end\{asy\}",
        "[Diagram omitted]",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned[:1500]


def _sanitize_solution_for_critic(sol: str, head_chars: int = 1200, tail_chars: int = 1000) -> str:
    """Làm sạch chuỗi rỗng/ký tự cắt cụt và giữ phần đầu + phần kết luận cuối của lời giải."""
    if not sol or not sol.strip():
        return "[Empty solution]"
    text = sol.strip()
    # Xóa dấu '\' lẻ ở cuối nếu lời giải bị cắt ngang giữa lệnh LaTeX
    if text.endswith("\\") and not text.endswith("\\\\"):
        text = text[:-1]
    max_len = head_chars + tail_chars
    if len(text) <= max_len:
        return text
    return (
        f"{text[:head_chars].rstrip()}\n\n"
        f"... [Middle steps omitted] ...\n\n"
        f"{text[-tail_chars:].lstrip()}"
    )


def _build_critic_alpaca_prompt(messages: List[Dict[str, str]]) -> str:
    sys_msg, usr_msg = "", ""
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            sys_msg = str(m.get("content", "")).strip()
        elif role == "user":
            usr_msg = str(m.get("content", "")).strip()

    m_step = re.match(
        r"^Problem:\s*(.*?)\n+Valid history:\s*(.*?)\n+Proposed next step:\s*(.*)$",
        usr_msg, flags=re.S
    )
    m_whole = re.match(
        r"^Problem:\s*(.*?)\n+Complete solution:\s*(.*)$",
        usr_msg, flags=re.S
    )
    if m_step:
        prob = _clean_problem_for_critic(m_step.group(1))
        hist = _sanitize_solution_for_critic(m_step.group(2), head_chars=800, tail_chars=800)
        step = _sanitize_solution_for_critic(m_step.group(3), head_chars=800, tail_chars=600)
        if hist.lower() == "none":
            hist = ""
        alpaca_input = f"Problem: {prob}\nSolution Process so far:\n{hist}\nCurrent Step to Evaluate:\n{step}"
    elif m_whole:
        prob = _clean_problem_for_critic(m_whole.group(1))
        sol = _sanitize_solution_for_critic(m_whole.group(2), head_chars=1200, tail_chars=1000)
        alpaca_input = f"Problem: {prob}\nSolution Process so far:\n\nCurrent Step to Evaluate:\n{sol}"
    else:
        alpaca_input = _sanitize_solution_for_critic(usr_msg, head_chars=1500, tail_chars=1500)

    instruction = PRM_CRITIC_INSTRUCTION if (
        "Value Network Critic" in sys_msg or "strict mathematical" in sys_msg or not sys_msg
    ) else sys_msg

    prompt = ALPACA_CRITIC_TEMPLATE.format(instruction=instruction, input=alpaca_input)
    # Chốt chặn cuối cùng: đảm bảo tổng độ dài prompt không bao giờ vượt ngưỡng an toàn
    if len(prompt) > CRITIC_MAX_PROMPT_CHARS:
        overflow = len(prompt) - CRITIC_MAX_PROMPT_CHARS
        alpaca_input = alpaca_input[:-overflow].rstrip()
        prompt = ALPACA_CRITIC_TEMPLATE.format(instruction=instruction, input=alpaca_input)
    return prompt


_orig_call_llm_before_critic_patch = call_llm


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> str:
    global LLM_CALLS, LAST_FINISH_REASON
    sys_txt = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    use_alpaca_completions = (
        model in (CRITIC_MODEL, VERIFIER_MODEL)
        and "Return VALID JSON ONLY" not in sys_txt
    )
    if not use_alpaca_completions:
        return _orig_call_llm_before_critic_patch(
            messages, model, temperature, max_tokens,
            max_retries=max_retries, seed=seed, stop=stop
        )

    prompt = _build_critic_alpaca_prompt(messages)
    comp_stop = stop if stop is not None else ["###", "<|eot_id|>", "<|end_of_text|>"]
    out_max_tokens = min(max_tokens, 160)
    delay = 1.5
    for attempt in range(1, max_retries + 1):
        LLM_CALLS += 1
        try:
            kwargs = dict(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=out_max_tokens,
                stop=comp_stop,
            )
            if seed is not None:
                kwargs["seed"] = seed
            res = get_client().completions.create(**kwargs)
            choice = res.choices[0] if res.choices else None
            LAST_FINISH_REASON = getattr(choice, "finish_reason", None) if choice is not None else None
            if choice is not None:
                LLM_STATS["finish_" + str(LAST_FINISH_REASON)] += 1
            content = (getattr(choice, "text", "") or "").strip() if choice is not None else ""
            _record_tokens(res, messages, model, content)
            if content:
                return content
            LLM_STATS["empty"] += 1
            if attempt <= MAX_EMPTY_RETRIES:
                temperature = max(temperature, 0.2)
                time.sleep(0.5)
                continue
            return ""
        except BadRequestError as e:
            err_body = getattr(e, "body", None) or getattr(getattr(e, "response", None), "text", str(e))
            print(
                f"[Critic API ERROR 400] prompt_chars={len(prompt)} | max_tokens={out_max_tokens} | "
                f"detail={json.dumps(err_body, ensure_ascii=False)}"
            )
            break
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            print(f"[Critic API retry {attempt}/{max_retries}] {e}")
            time.sleep(delay)
            delay *= 1.5
        except Exception as e:
            print(f"[Critic API ERROR] {type(e).__name__}: {e}")
            break
    return ""


_orig_parse_critic_before_patch = globals().get("parse_critic")


def parse_critic(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    m_score = _CRITIC_SCORE_RE.search(text)
    if m_score:
        try:
            score = max(0.0, min(1.0, float(m_score.group(1))))
        except Exception:
            score = 0.0
        decision = "ACCEPT" if score >= globals().get("ACCEPT_THRESHOLD", 0.70) else "REJECT"
        m_hint = _CRITIC_HINT_RE.search(text)
        if m_hint:
            fb = m_hint.group(1).strip()
        else:
            fb = re.sub(r"</?hint>", " ", text[m_score.end():], flags=re.I).strip()
        if not fb:
            fb = "Step verified." if decision == "ACCEPT" else "Logic or calculation error detected in this step."
        return {
            "parse_ok": True,
            "decision": decision,
            "completion": "UNKNOWN",
            "score": score,
            "confidence": 1.0 if score in (0.1, 1.0) else 0.7,
            "error_type": "NONE" if decision == "ACCEPT" else "LOGIC",
            "error_step": "" if decision == "ACCEPT" else fb[:120],
            "feedback": fb[:200],
        }
    return _orig_parse_critic_before_patch(raw)