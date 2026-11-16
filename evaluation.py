import os
import re
import json
import math
import time
import argparse
import difflib
import ast
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import subprocess
import tempfile
from collections import Counter
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
    convert_xor,
)

try:
    from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError
except ImportError:  # cho phép import/test offline
    OpenAI = None

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class InternalServerError(Exception):
        pass


# ------------------------- CONFIG ----------------------------
BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
TIMEOUT = 600

GENERATOR_MODEL = "meta-llama-3.1-8b-instruct"
CRITIC_MODEL = "dualref_critic_q4_k_m_v2"
VERIFIER_MODEL = CRITIC_MODEL  
VERIFY_MODE = "sc"             # "llm" | "sc" 
VERIFY_K = 3
CRITIC_POLICY = "decision"     
EXTRACT_FALLBACK = True        

OUT_PREFIX = "benchmark_v5_8b_full__"
FINAL_ANSWER_TAG = "FINAL_ANSWER:"

HARD_REJECT_THRESHOLD = 0.55
ACCEPT_THRESHOLD = 0.70
FORCE_ACCEPT_FLOOR = 0.30  

MAX_STEPS = 10
MAX_RETRIES_PER_STEP = 3
MAX_CALLS_PER_PROBLEM = 40  
CRITIC_MAX_TOKENS = 700
PROBE_FROM_STEP = 2         
BON_N = 5
PAL_K = 5
PAL_TIMEOUT = 8
MAX_EMPTY_RETRIES = 2
MAX_FINAL_VERIFY_REPAIRS = 2
FINAL_CONSENSUS_VOTES = 2  

DIRECT_MAX_TOKENS = 2500
SC_TEMPERATURE = 0.7

ALLOW_TOOL_JSON = False  

_client = None
LLM_CALLS = 0  
LLM_STATS: Counter = Counter()  
TOKEN_STATS: Counter = Counter()  


def get_client():
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError("Chưa cài thư viện openai: pip install openai")
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)
    return _client


# ============================================================
# 1. DATASET
# ============================================================
DATASET_20 = [
    {"id": "M01", "category": "Number Theory",
     "problem": "Tìm hai chữ số tận cùng của tổng: S = 1! + 2! + 3! + ... + 2026!.",
     "ground_truth": "13"},
    {"id": "M02", "category": "Combinatorics",
     "problem": "Có bao nhiêu xâu nhị phân độ dài 10 không chứa ba số 0 liên tiếp?",
     "ground_truth": "504"},
    {"id": "M03", "category": "Algebra",
     "problem": "Tìm số nghiệm nguyên dương của phương trình x*y + 2x + 3y = 2026.",
     "ground_truth": "6"},
    {"id": "M04", "category": "Number Theory",
     "problem": "Tìm số dư khi chia 7^(2026) cho 100.",
     "ground_truth": "49"},
    {"id": "M05", "category": "Geometry / Algebra",
     "problem": "Tìm giá trị nhỏ nhất của biểu thức P = sqrt(x^2 + (1 - y)^2) + sqrt((x - 1)^2 + y^2) với mọi x, y thuộc R.",
     "ground_truth": "sqrt(2)"},
    {"id": "M06", "category": "Polynomials",
     "problem": "Cho đa thức P(x) = (x - 1)(x - 2)...(x - 10). Hệ số của x^9 trong đa thức P(x) bằng bao nhiêu?",
     "ground_truth": "-55"},
    {"id": "M07", "category": "Probability",
     "problem": "Tung một đồng xu công bằng liên tiếp. Kỳ vọng số lần tung để lần đầu tiên xuất hiện cụm 'Ngửa - Sấp' (HT) là bao nhiêu?",
     "ground_truth": "4"},
    {"id": "M08", "category": "Number Theory",
     "problem": "Tìm ước chung lớn nhất của hai số: 2^2026 - 1 và 2^2022 - 1.",
     "ground_truth": "3"},
    {"id": "M09", "category": "Combinatorics",
     "problem": "Có bao nhiêu cách chia 8 cái kẹo giống nhau cho 4 đứa trẻ sao cho mỗi đứa trẻ nhận được ít nhất 1 cái kẹo?",
     "ground_truth": "35"},
    {"id": "M10", "category": "Algebra",
     "problem": "Cho x + 1/x = 3. Tính giá trị chính xác của biểu thức Q = x^5 + 1/(x^5).",
     "ground_truth": "123"},
    {"id": "M11", "category": "Combinatorics",
     "problem": "Tính số tập con khác rỗng của tập hợp S = {1, 2, 3, 4, 5, 6, 7, 8} sao cho tổng các phần tử trong tập con đó là một số lẻ.",
     "ground_truth": "128"},
    {"id": "M12", "category": "Calculus / Sequence",
     "problem": "Tính tổng của chuỗi vô hạn: S = 1/2 + 2/4 + 3/8 + 4/16 + ... + n/(2^n) + ...",
     "ground_truth": "2"},
    {"id": "M13", "category": "Number Theory",
     "problem": "Số ước nguyên dương của số 360000 là bao nhiêu?",
     "ground_truth": "105"},
    {"id": "M14", "category": "Linear Algebra",
     "problem": "Cho ma trận vuông A cấp 3 thỏa mãn det(A) = 4. Tính định thức của ma trận 2 * A^(-1).",
     "ground_truth": "2"},
    {"id": "M15", "category": "Trigonometry",
     "problem": "Tính giá trị của tích P = cos(20 độ) * cos(40 độ) * cos(80 độ).",
     "ground_truth": "1/8"},
    {"id": "M16", "category": "Probability",
     "problem": "Chọn ngẫu nhiên 3 đỉnh từ một đa giác đều 12 cạnh. Xác suất để tam giác tạo thành là tam giác vuông bằng bao nhiêu?",
     "ground_truth": "3/11"},
    {"id": "M17", "category": "Number Theory",
     "problem": "Tìm nghiệm nguyên không âm x nhỏ nhất thỏa mãn hệ đồng dư: x đồng dư 2 (mod 3), x đồng dư 3 (mod 5), x đồng dư 2 (mod 7).",
     "ground_truth": "23"},
    {"id": "M18", "category": "Algebra",
     "problem": "Tìm số nghiệm thực của phương trình: 2^x + 3^x = 5^x.",
     "ground_truth": "1"},
    {"id": "M19", "category": "Logarithms",
     "problem": "Nếu log_3(2) = a và log_3(5) = b, hãy tính log_3(50) theo a và b.",
     "ground_truth": "a + 2b"},
    {"id": "M20", "category": "Combinatorics",
     "problem": "Một bàn tròn có 6 ghế đánh số. Có bao nhiêu cách xếp 3 nam và 3 nữ ngồi quanh bàn sao cho không có 2 người cùng giới nào ngồi cạnh nhau?",
     "ground_truth": "72"},
    {"id": "M21", "category": "Number Theory",
     "problem": "Tìm số dư khi chia 3^2027 cho 11.",
     "ground_truth": "9"},
    {"id": "M22", "category": "Number Theory",
     "problem": "Số ước nguyên dương của số 5040 là bao nhiêu?",
     "ground_truth": "60"},
    {"id": "M23", "category": "Combinatorics",
     "problem": "Có bao nhiêu xâu nhị phân độ dài 12 không chứa hai số 1 liên tiếp?",
     "ground_truth": "377"},
    {"id": "M24", "category": "Combinatorics",
     "problem": "Từ một nhóm gồm 6 nam và 5 nữ, có bao nhiêu cách chọn ra một ban gồm 3 nam và 2 nữ?",
     "ground_truth": "200"},
    {"id": "M25", "category": "Probability",
     "problem": "Tung ba đồng xu cân đối. Xác suất để có đúng 2 mặt ngửa là bao nhiêu?",
     "ground_truth": "3/8"},
    {"id": "M26", "category": "Algebra",
     "problem": "Tính tổng bình phương các nghiệm của phương trình x^3 - 6x^2 + 11x - 6 = 0.",
     "ground_truth": "14"},
    {"id": "M27", "category": "Sequences",
     "problem": "Cho dãy số a_1 = 2 và a_(n+1) = 2*a_n + 1 với mọi n >= 1. Tính a_8.",
     "ground_truth": "383"},
    {"id": "M28", "category": "Geometry",
     "problem": "Một tam giác vuông có hai cạnh góc vuông dài 9 và 12. Tính bán kính đường tròn nội tiếp tam giác đó.",
     "ground_truth": "3"},
    {"id": "M29", "category": "Calculus",
     "problem": "Cho hàm số f(x) = x^4 - 3x^2. Tính đạo hàm cấp hai f''(2).",
     "ground_truth": "42"},
    {"id": "M30", "category": "Linear Algebra",
     "problem": "Cho ma trận A = [[2, 1], [1, 3]]. Tính det(A^3).",
     "ground_truth": "125"},
    {"id": "M31", "category": "Number Theory",
     "problem": "Tìm ước chung lớn nhất của hai số: 2^30 - 1 và 2^45 - 1.",
     "ground_truth": "32767"},
    {"id": "M32", "category": "Combinatorics",
     "problem": "Có bao nhiêu cách xếp 5 người thành một hàng ngang sao cho hai người A và B đứng cạnh nhau?",
     "ground_truth": "48"},
    {"id": "M33", "category": "Number Theory",
     "problem": "Tìm số nguyên không âm x nhỏ nhất thỏa mãn hệ đồng dư: x đồng dư 1 (mod 4), x đồng dư 2 (mod 5), x đồng dư 3 (mod 7).",
     "ground_truth": "17"},
    {"id": "M34", "category": "Combinatorics",
     "problem": "Tìm số nghiệm nguyên dương của phương trình x + y + z = 12.",
     "ground_truth": "55"},
    {"id": "M35", "category": "Series",
     "problem": "Tính tổng S = 1/(1*2) + 1/(2*3) + 1/(3*4) + ... + 1/(99*100).",
     "ground_truth": "99/100"},
    {"id": "M36", "category": "Trigonometry",
     "problem": "Tính giá trị của biểu thức P = sin(15 độ) * cos(15 độ).",
     "ground_truth": "1/4"},
    {"id": "M37", "category": "Logarithms",
     "problem": "Nếu log_2(3) = a, hãy tính log_2(72) theo a.",
     "ground_truth": "3 + 2a"},
    {"id": "M38", "category": "Combinatorics",
     "problem": "Có bao nhiêu tập con (kể cả tập rỗng) của tập S = {1, 2, ..., 10} không chứa hai số nguyên liên tiếp?",
     "ground_truth": "144"},
    {"id": "M39", "category": "Probability",
     "problem": "Một hộp có 4 bi đỏ và 6 bi xanh. Rút ngẫu nhiên đồng thời 2 bi. Xác suất để hai bi cùng màu là bao nhiêu?",
     "ground_truth": "7/15"},
    {"id": "M40", "category": "Number Theory",
     "problem": "Tìm số dư khi chia 2^100 cho 7.",
     "ground_truth": "2"},
    #   --- NUMBER THEORY (10 câu) ---
    {"id": "M41", "category": "Number Theory",
     "problem": "Tìm hai chữ số tận cùng của tổng: S = 1^2026 + 2^2026 + 3^2026 + ... + 2026^2026.",
     "ground_truth": "70"},
    {"id": "M42", "category": "Number Theory",
     "problem": "Tìm số dư khi chia 13^(2026) cho 100.",
     "ground_truth": "69"},
    {"id": "M43", "category": "Number Theory",
     "problem": "Tìm số nguyên không âm x nhỏ nhất thỏa mãn hệ đồng dư: x đồng dư 1 (mod 5), x đồng dư 4 (mod 7), x đồng dư 3 (mod 9).",
     "ground_truth": "291"},
    {"id": "M44", "category": "Number Theory",
     "problem": "Tìm ước chung lớn nhất của hai số: 2^2024 - 1 và 2^2016 - 1.",
     "ground_truth": "255"},
    {"id": "M45", "category": "Number Theory",
     "problem": "Số ước nguyên dương của số 180000 là bao nhiêu?",
     "ground_truth": "90"},
    {"id": "M46", "category": "Number Theory",
     "problem": "Tìm số dư khi chia 5^2026 cho 13.",
     "ground_truth": "12"},
    {"id": "M47", "category": "Number Theory",
     "problem": "Số ước nguyên dương của số 2520 là bao nhiêu?",
     "ground_truth": "48"},
    {"id": "M48", "category": "Number Theory",
     "problem": "Tìm số dư khi chia 3^100 cho 13.",
     "ground_truth": "3"},
    {"id": "M49", "category": "Number Theory",
     "problem": "Tìm hai chữ số tận cùng của số 7^(100).",
     "ground_truth": "01"},
    {"id": "M50", "category": "Number Theory",
     "problem": "Tìm số nguyên dương n nhỏ nhất sao cho n! chia hết cho 1000.",
     "ground_truth": "15"},

    # --- COMBINATORICS (10 câu) ---
    {"id": "M51", "category": "Combinatorics",
     "problem": "Có bao nhiêu xâu nhị phân độ dài 10 không chứa hai số 1 liên tiếp?",
     "ground_truth": "144"},
    {"id": "M52", "category": "Combinatorics",
     "problem": "Có bao nhiêu xâu nhị phân độ dài 8 không chứa ba số 0 liên tiếp?",
     "ground_truth": "149"},
    {"id": "M53", "category": "Combinatorics",
     "problem": "Có bao nhiêu cách chia 10 cái kẹo giống nhau cho 3 đứa trẻ sao cho mỗi đứa nhận được ít nhất 1 cái kẹo?",
     "ground_truth": "36"},
    {"id": "M54", "category": "Combinatorics",
     "problem": "Một bàn tròn có 8 ghế đánh số. Có bao nhiêu cách xếp 4 nam và 4 nữ ngồi quanh bàn sao cho không có 2 người cùng giới nào ngồi cạnh nhau?",
     "ground_truth": "1152"},
    {"id": "M55", "category": "Combinatorics",
     "problem": "Có bao nhiêu tập con (kể cả tập rỗng) của tập S = {1, 2, ..., 8} không chứa hai số nguyên liên tiếp?",
     "ground_truth": "55"},
    {"id": "M56", "category": "Combinatorics",
     "problem": "Tính số tập con khác rỗng của tập hợp S = {1, 2, 3, 4, 5, 6} sao cho tổng các phần tử trong tập con đó là một số lẻ.",
     "ground_truth": "32"},
    {"id": "M57", "category": "Combinatorics",
     "problem": "Có bao nhiêu cách xếp 6 người thành một hàng ngang sao cho hai người A và B luôn đứng cạnh nhau?",
     "ground_truth": "240"},
    {"id": "M58", "category": "Combinatorics",
     "problem": "Tìm số nghiệm nguyên dương của phương trình x + y + z + t = 10.",
     "ground_truth": "84"},
    {"id": "M59", "category": "Combinatorics",
     "problem": "Từ một nhóm gồm 7 nam và 4 nữ, có bao nhiêu cách chọn ra một ban gồm 4 nam và 2 nữ?",
     "ground_truth": "210"},
    {"id": "M60", "category": "Combinatorics",
     "problem": "Có bao nhiêu đường đi ngắn nhất trên lưới tọa độ từ điểm (0,0) đến điểm (5,5) chỉ đi theo chiều dương trục Ox và Oy?",
     "ground_truth": "252"},

    # --- ALGEBRA & POLYNOMIALS (10 câu) ---
    {"id": "M61", "category": "Algebra",
     "problem": "Tìm số nghiệm nguyên dương của phương trình x*y + 3x + 2y = 2026.",
     "ground_truth": "8"},
    {"id": "M62", "category": "Algebra",
     "problem": "Cho x + 1/x = 4. Tính giá trị chính xác của biểu thức Q = x^4 + 1/(x^4).",
     "ground_truth": "194"},
    {"id": "M63", "category": "Algebra",
     "problem": "Tìm số nghiệm thực của phương trình: 3^x + 4^x = 5^x.",
     "ground_truth": "2"},
    {"id": "M64", "category": "Logarithms",
     "problem": "Nếu log_2(5) = a và log_2(3) = b, hãy tính log_2(45) theo a và b.",
     "ground_truth": "a + 2b"},
    {"id": "M65", "category": "Algebra",
     "problem": "Tính tổng bình phương các nghiệm của phương trình x^3 - 7x^2 + 14x - 8 = 0.",
     "ground_truth": "21"},
    {"id": "M66", "category": "Polynomials",
     "problem": "Cho đa thức P(x) = (x - 1)(x - 2)...(x - 8). Hệ số của x^7 trong đa thức P(x) bằng bao nhiêu?",
     "ground_truth": "-36"},
    {"id": "M67", "category": "Algebra",
     "problem": "Tìm giá trị nhỏ nhất của biểu thức P = sqrt(x^2 + (2 - y)^2) + sqrt((x - 2)^2 + y^2) với mọi x, y thuộc R.",
     "ground_truth": "2*sqrt(2)"},
    {"id": "M68", "category": "Logarithms",
     "problem": "Nếu log_5(2) = a, hãy tính log_5(200) theo a.",
     "ground_truth": "3a + 2"},
    {"id": "M69", "category": "Sequences",
     "problem": "Cho dãy số a_1 = 3 và a_(n+1) = 2*a_n - 1 với mọi n >= 1. Tính a_6.",
     "ground_truth": "65"},
    {"id": "M70", "category": "Series",
     "problem": "Tính tổng S = 1/(1*3) + 1/(3*5) + 1/(5*7) + ... + 1/(19*21).",
     "ground_truth": "10/21"},

    # --- PROBABILITY, GEOMETRY & OTHERS (10 câu) ---
    {"id": "M71", "category": "Probability",
     "problem": "Tung một đồng xu công bằng liên tiếp. Kỳ vọng số lần tung để lần đầu tiên xuất hiện cụm 'Ngửa - Ngửa' (HH) là bao nhiêu?",
     "ground_truth": "6"},
    {"id": "M72", "category": "Probability",
     "problem": "Tung bốn đồng xu cân đối. Xác suất để có đúng 3 mặt ngửa là bao nhiêu?",
     "ground_truth": "1/4"},
    {"id": "M73", "category": "Probability",
     "problem": "Một hộp có 5 bi đỏ và 5 bi xanh. Rút ngẫu nhiên đồng thời 2 bi. Xác suất để hai bi cùng màu là bao nhiêu?",
     "ground_truth": "4/9"},
    {"id": "M74", "category": "Probability",
     "problem": "Chọn ngẫu nhiên 3 đỉnh từ một đa giác đều 10 cạnh. Xác suất để tam giác tạo thành là tam giác vuông bằng bao nhiêu?",
     "ground_truth": "1/3"},
    {"id": "M75", "category": "Geometry",
     "problem": "Một tam giác vuông có hai cạnh góc vuông dài 5 và 12. Tính bán kính đường tròn nội tiếp tam giác đó.",
     "ground_truth": "2"},
    {"id": "M76", "category": "Calculus",
     "problem": "Cho hàm số f(x) = x^5 - 2x^3. Tính đạo hàm cấp hai f''(3).",
     "ground_truth": "504"},
    {"id": "M77", "category": "Linear Algebra",
     "problem": "Cho ma trận vuông A cấp 3 thỏa mãn det(A) = 5. Tính định thức của ma trận 3 * A^(-1).",
     "ground_truth": "27/5"},
    {"id": "M78", "category": "Linear Algebra",
     "problem": "Cho ma trận A = [[3, 1], [2, 2]]. Tính det(A^3).",
     "ground_truth": "64"},
    {"id": "M79", "category": "Trigonometry",
     "problem": "Tính giá trị của biểu thức P = sin(22.5 độ) * cos(22.5 độ).",
     "ground_truth": "sqrt(2)/4"},
    {"id": "M80", "category": "Calculus / Sequence",
     "problem": "Tính tổng của chuỗi vô hạn: S = 1/3 + 2/9 + 3/27 + 4/81 + ... + n/(3^n) + ...",
     "ground_truth": "3/4"}
]

# DATASET_20 = [
#    
# ]

def load_dataset(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return DATASET_20
    if path.endswith(".csv"):
        df = pd.read_csv(path, dtype=str).fillna("")
        rows = df.to_dict("records")
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    out = []
    for i, r in enumerate(rows, 1):
        out.append({
            "id": str(r.get("id", f"P{i:04d}")),
            "category": str(r.get("category", "")),
            "problem": str(r["problem"]),
            "ground_truth": str(r["ground_truth"]),
        })
    return out


# ============================================================
# 2. PROMPTS  (100% English for reasoning tasks)
# ============================================================

DIRECT_SYSTEM_PROMPT = f"""
You are an expert mathematician. Solve the problem step by step using plain text and mathematical formulas, concisely, one idea per line.
You must reason in English.

End with exactly one final line:
{FINAL_ANSWER_TAG} <answer>
This line must contain ONLY the answer (a number, a fraction, or an expression), no explanation.
""".strip()

STEP_GENERATOR_SYSTEM_PROMPT = f"""
You are an expert mathematician solving the problem step by step.

On each turn, write ONE new reasoning step in plain text, starting with "STEP:". A step should be substantial (about 2-5 sentences with concrete computation) and must not repeat earlier steps.
IMPORTANT: as soon as your step yields the final result of the problem, you MUST append one final line: {FINAL_ANSWER_TAG} <answer> (answer only) and stop writing any further step.
""".strip()

PROBE_SYSTEM_PROMPT = f"""
You inspect the progress of a mathematical solution.
If the existing steps are sufficient to determine the final answer of the problem, print exactly one line: {FINAL_ANSWER_TAG} <answer>
If not sufficient, print exactly one word: CONTINUE
""".strip()

FINALIZE_SYSTEM_PROMPT = f"""
You are an expert mathematician. Below is an incomplete solution.
Complete the remaining part concisely, self-correct if you find a mistake, then end with exactly one line:
{FINAL_ANSWER_TAG} <answer>
""".strip()

BON_CRITIC_SYSTEM_PROMPT = """
You are a strict mathematical solution checker.
Judge whether the COMPLETE solution below is correct and its final answer is right.

Reply with ONE compact JSON object and nothing else. Write every text field in English, at most 12 words:
{"decision": "ACCEPT" or "REJECT", "completion": "COMPLETE", "score": 0.0-1.0, "confidence": 0.0-1.0, "error_type": "NONE" or "ARITHMETIC" or "LOGIC" or "OTHER", "error_step": "...", "feedback": "..."}
""".strip()

PAL_SYSTEM_PROMPT = """
You are an expert mathematical programmer.
Write ONE short Python program that computes the answer to the problem.
Allowed libraries only: math, itertools, fractions, sympy, collections, functools, decimal, statistics.
You may brute force small ranges.

IMPORTANT — symbolic / parametric answers:
If the problem asks for the answer in terms of parameters (e.g. "in terms of a and b", "express ... using a, b"), use sympy.Symbol to construct and print the symbolic expression. Example:
    from sympy import symbols
    a, b = symbols('a b')
    # Example: log_3(50) = log_3(2) + 2*log_3(5) = a + 2*b
    result = a + 2*b
    print(result)
Never substitute arbitrary numeric values for a and b in such cases.

Output rules:
- The program must end with EXACTLY ONE print() statement that outputs the final answer, EXACTLY, no rounding.
- Accepted answer formats: integer, Fraction(a, b), or a sympy expression such as sqrt(2), 2*sqrt(2), a + 2*b.
- Do NOT print debug information and do NOT call print() more than once.
Reply with exactly ONE python code block: ```python ... ```.
""".strip()

PAL_DEBUG_SYSTEM_PROMPT = """
You are a Python developer who fixes broken math-solving code.
The previous program either crashed (syntax error, runtime exception) or printed no result.
Analyze the error message and rewrite ONE complete, corrected Python program that computes the correct answer.

IMPORTANT — symbolic / parametric answers:
If the problem asks for the answer in terms of parameters (e.g. "in terms of a and b"), use sympy.Symbol to construct and print the symbolic expression.

Output rules:
- The program must end with EXACTLY ONE print() statement that outputs the final answer, EXACTLY, no rounding.
- Do NOT print debug information and do NOT call print() more than once.
Reply with exactly ONE python code block: ```python ... ```.
""".strip()

DIRECT_SYSTEM_PROMPT_EN = f"""
You are an expert mathematician. Solve the problem step by step with plain text and formulas, concisely, one idea per line.
You must reason in English. Do not call any function.
The last line must be exactly: {FINAL_ANSWER_TAG} <answer>
It contains only the answer (a number, a fraction, or an expression), no explanation.
""".strip()

REPAIR_FROM_ERROR_PROMPT = f"""
You are an expert mathematician. The solution below contains a FALSE arithmetic equality (machine-verified by SymPy).
Rewrite the solution, keep the correct parts, and fix the erroneous step, concisely.
Final line: {FINAL_ANSWER_TAG} <answer>
""".strip()

REPAIR_SYSTEM_PROMPT = f"""
You are an expert mathematician repairing a rejected step.

Write ONE corrected replacement step that directly addresses the feedback, in plain text, starting with "STEP:".
Do not repeat the rejected step. If the problem is now solved, append a final line: {FINAL_ANSWER_TAG} <answer>.
""".strip()

CRITIC_SYSTEM_PROMPT = """
You are a strict mathematical step checker.
Judge ONLY the proposed next step against the problem and the valid history.

Reply with ONE compact JSON object and nothing else. Write every text field in English, at most 12 words:
{"decision": "ACCEPT" or "REJECT", "completion": "COMPLETE" or "INCOMPLETE", "score": 0.0-1.0, "confidence": 0.0-1.0, "error_type": "NONE" or "ARITHMETIC" or "LOGIC" or "RELEVANCE" or "DUPLICATION" or "OTHER", "error_step": "...", "feedback": "..."}

ACCEPT = the step is mathematically correct and makes progress. REJECT = it contains an error or repeats earlier work.
COMPLETE only if the whole problem is solved with a justified final answer.
""".strip()

FINAL_VERIFY_SYSTEM_PROMPT = """
You are an independent final mathematical verifier.

Determine whether the generated solution really solves the original problem.
Do NOT use or infer a hidden ground-truth target. Judge from the problem and reasoning.

Return VALID JSON ONLY:
{
  "decision": "PASS" | "FAIL",
  "score": 0.0,
  "reason": "short reason"
}

PASS requires a concrete final answer and sufficient reasoning with no critical mathematical error.
""".strip()

# NOTE: PAL_TO_TEXT_PROMPT is intentionally kept in Vietnamese:
# it is a *presentation-only* synthesizer (not used for reasoning/answer generation).
# It turns PAL code into readable Vietnamese math text for human reports.
PAL_TO_TEXT_PROMPT = """
Bạn là một giáo viên toán giỏi. Hãy dựa vào bài toán, mã Python và đáp số dưới đây để trình bày lại bài giải từng bước bằng LỜI VĂN TIẾNG VIỆT và CÔNG THỨC TOÁN HỌC rõ ràng, mạch lạc.

YÊU CẦU BẮT BUỘC:
1. TUYỆT ĐỐI KHÔNG xuất hiện mã code Python, tên biến code, hay bất kỳ khối ```python nào trong bài giải.
2. Trình bày lập luận toán học từng bước theo đúng logic mà thuật toán đã giải (ví dụ: dùng Quy tắc nhân, Hoán vị, Định lý dư Trung Hoa, Tổ hợp,...).
3. Sử dụng công thức toán học (Latex hoặc ký hiệu toán) để trình bày các phép tính.
4. Kết thúc bài giải bằng đúng dòng:
FINAL_ANSWER: <đáp án>
""".strip()


# ============================================================
# 3. LLM API
# ============================================================


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.2)) if text else 0


def _record_tokens(res: Any, messages: List[Dict[str, str]], model: str, content: str) -> None:
    usage = getattr(res, "usage", None)
    pt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    ct = getattr(usage, "completion_tokens", None) if usage is not None else None
    if pt is None or ct is None:
        pt = _approx_tokens("".join(str(m.get("content", "")) for m in messages))
        ct = _approx_tokens(content)
        TOKEN_STATS["estimated_calls"] += 1
    TOKEN_STATS["prompt"] += int(pt)
    TOKEN_STATS["completion"] += int(ct)
    TOKEN_STATS["generator" if model == GENERATOR_MODEL else "other"] += int(pt) + int(ct)


def token_snapshot() -> Dict[str, int]:
    return dict(TOKEN_STATS)


def token_delta(before: Dict[str, int]) -> Dict[str, int]:
    d = {k: TOKEN_STATS[k] - before.get(k, 0)
         for k in ("prompt", "completion", "generator", "other", "estimated_calls")}
    return {
        "prompt_tokens": d["prompt"], "completion_tokens": d["completion"],
        "total_tokens": d["prompt"] + d["completion"],
        "tokens_generator_model": d["generator"],
        "tokens_other_model": d["other"],
        "token_estimated_calls": d["estimated_calls"],
    }


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    seed: Optional[int] = None,
) -> str:
    global LLM_CALLS
    LLM_CALLS += 1
    delay = 1.5
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if seed is not None:
                kwargs["seed"] = seed
            res = get_client().chat.completions.create(**kwargs)
            choice = res.choices[0] if res.choices else None
            msg = choice.message if choice is not None else None
            if choice is not None:
                LLM_STATS["finish_" + str(getattr(choice, "finish_reason", None))] += 1
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
                kwargs_temp = max(temperature, 0.3)
                temperature = kwargs_temp
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


# ============================================================
# 4. GENERATOR FORMAT ADAPTER
# ============================================================


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text.strip())
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _collect_strings(obj: Any, skip_keys=("name", "type", "id")) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            out.extend(_collect_strings(v, skip_keys))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_strings(v, skip_keys))
    return out


_TOOL_JSON = re.compile(r'\{\s*"name"\s*:[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')


def strip_tool_json(text: str) -> str:
    return _TOOL_JSON.sub("", text).strip() if text else text


def normalize_generator_output(text: str) -> Tuple[str, str]:
    if not text:
        return "FORMAT_REJECT", "Generator trả về rỗng. Hãy trả về STEP: <suy luận toán học>."

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:text|markdown|json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    status = "ACCEPT"
    if not cleaned.startswith("{") and '"name"' in cleaned:
        mixed = strip_tool_json(cleaned)
        if mixed:
            cleaned = mixed
            status = "ACCEPT_ADAPTED"
    if cleaned.startswith("{"):
        obj = extract_json_object(cleaned)
        if obj is not None:
            strings = [s.strip() for s in _collect_strings(obj) if s and s.strip()]
            content = " ".join(strings).strip()
            if len(content) >= 8:
                cleaned = content
                status = "ACCEPT_ADAPTED"
            else:
                return "FORMAT_REJECT", (
                    "Generator sinh JSON/tool-call không có nội dung toán. "
                    "Hãy trả về STEP: <suy luận toán học>."
                )

    if not re.search(r"\bSTEP\s*:", cleaned, flags=re.I):
        final_match = re.search(
            rf"{re.escape(FINAL_ANSWER_TAG)}\s*(.+)", cleaned, flags=re.I
        )
        if final_match:
            body = cleaned[:final_match.start()].strip()
            answer = final_match.group(1).strip()
            cleaned = (
                f"STEP: {body}\n{FINAL_ANSWER_TAG} {answer}"
                if body
                else f"STEP: Hoàn tất phép tính.\n{FINAL_ANSWER_TAG} {answer}"
            )
        else:
            cleaned = f"STEP: {cleaned}"

    return status, cleaned


_FORBIDDEN_ANSWERS = {
    "<đáp án>", "<giá trị>", "<giá trị cụ thể>", "<đáp số>", "<answer>", "<value>",
}


def clean_final_answer(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().strip("`$ \t\n")
    if not t:
        return None

    m = re.search(r"\\boxed\{(.+)\}", t)
    if m:
        t = m.group(1)
    t = re.sub(r"\*\*|\\\(|\\\)|\\\[|\\\]", "", t).strip()

    t = re.split(
        r"\s*\((?=\s*[^\W\d_]+\s+[^\W\d_]+\s+[^\W\d_]+)", t, maxsplit=1
    )[0].strip()

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
    if "{" in t or '"' in t or len(t) > 60 or len(t.split()) > 6:
        return None
    if len(t) == 1 and t.isalpha() and t.upper() not in "ABCDE":
        return None
    if t.lower() in _FORBIDDEN_ANSWERS:
        return None
    if t.startswith("<") and t.endswith(">"):
        return None
    return t


def extract_final_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(rf"{re.escape(FINAL_ANSWER_TAG)}\s*(.+)", text, flags=re.I)
    if matches:
        return clean_final_answer(matches[-1])
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return clean_final_answer(boxed[-1])
    alt = re.findall(
        r"(?:đáp\s*án|đáp\s*số|final\s+answer)\s*[:：]\s*(.+)",
        text, flags=re.I,
    )
    if alt:
        return clean_final_answer(alt[-1])
    return None


def strip_final_line(step: str) -> str:
    lines = [
        ln for ln in step.splitlines()
        if not re.match(rf"\s*{re.escape(FINAL_ANSWER_TAG)}", ln, flags=re.I)
    ]
    return "\n".join(lines).strip()


# ============================================================
# 5. PARSER TOÁN AN TOÀN + LOCAL GATES
# ============================================================

_ALLOWED_FUNCS = {
    "sqrt": sp.sqrt, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos,
    "tan": sp.tan, "log": sp.log, "exp": sp.exp,
}
_TRANSFORMS = standard_transformations + (implicit_multiplication, convert_xor)
_LOCAL_DICT = {c: sp.Symbol(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
_LOCAL_DICT.update(_ALLOWED_FUNCS)


def _normalize_math_expr(s: str) -> str:
    s = s.strip()
    for a, b in {
        "×": "*", "·": "*", "−": "-", "–": "-", "÷": "/",
        "π": "pi", "²": "^2", "³": "^3",
    }.items():
        s = s.replace(a, b)
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace(r"\sqrt", "sqrt").replace(r"\pi", "pi").replace(r"\cdot", "*")
    s = re.sub(r"sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"√\s*\(\s*([^()]+)\s*\)", r"sqrt(\1)", s)
    s = re.sub(r"√\s*([A-Za-z0-9_.]+)", r"sqrt(\1)", s)
    s = s.replace("căn", "sqrt").replace("$", "")
    return s


def safe_parse(s: Optional[str]):
    if not s:
        return None
    t = _normalize_math_expr(str(s)).strip().rstrip(".")
    if not t or len(t) > 160:
        return None
    if not re.fullmatch(r"[0-9A-Za-z\s\+\-\*/\^\(\)\.!]+", t):
        return None
    if "!=" in t or re.search(r"(?<!\d)\.|\.(?!\d)", t):
        return None
    for w in re.findall(r"[A-Za-z]+", t):
        if w not in _ALLOWED_FUNCS and len(w) > 1:
            return None
    if re.search(r"(\^|\*\*)[^+\-*/]*(\^|\*\*)", t):
        return None
    for m in re.finditer(r"(?:\^|\*\*)\s*\(?\s*(\d+)", t):
        if int(m.group(1)) > 5000:
            return None
    for m in re.finditer(r"(\d+)\s*!", t):
        if int(m.group(1)) > 3000:
            return None
    try:
        return parse_expr(
            t, local_dict=_LOCAL_DICT, transformations=_TRANSFORMS, evaluate=True
        )
    except Exception:
        return None


def detect_repetition(text: str, history: List[str], threshold: float = 0.92) -> Optional[str]:
    current = text.strip().lower()
    offset = max(0, len(history) - 3)
    for i, prev in enumerate(history[offset:], start=offset):
        ratio = difflib.SequenceMatcher(None, current, prev.strip().lower()).ratio()
        if ratio >= threshold:
            return f"Bước quá giống Bước {i+1} (similarity={ratio:.2f})."
    return None


_EQ_RUN = re.compile(r"[\d\.\(\)\s\+\-\*/\^!=×·−÷]+")


def _balanced(t: str) -> bool:
    depth = 0
    for ch in t:
        depth += (ch == "(") - (ch == ")")
        if depth < 0:
            return False
    return depth == 0


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
                        return "REJECT", f"Đẳng thức sai: {lhs} = {rhs}; SymPy cho {lv} != {rv}."
            except Exception:
                continue
    return "ACCEPT", "Không phát hiện đẳng thức số học sai rõ ràng."


def local_gate(candidate: str, history: List[str]) -> Tuple[str, str, str]:
    repetition = None if extract_final_answer(candidate) else detect_repetition(candidate, history)
    if repetition:
        return "REJECT", "DUPLICATION", repetition
    decision, feedback = safe_sympy_check_equalities(candidate)
    if decision != "ACCEPT":
        return "REJECT", "ARITHMETIC", feedback
    return "ACCEPT", "NONE", "Local gate passed."


# ============================================================
# 6. CRITIC PARSER + POLICY
# ============================================================


def parse_critic(raw: str) -> Dict[str, Any]:
    obj = extract_json_object(raw)
    if not obj and raw:
        dm = re.search(r'"decision"\s*:\s*"(ACCEPT|REJECT)"', raw, flags=re.I)
        if dm:
            def _f(name, default):
                m = re.search(rf'"{name}"\s*:\s*"?([A-Za-z]+|[0-9.]+)', raw)
                return m.group(1) if m else default
            obj = {"decision": dm.group(1), "completion": _f("completion", "UNKNOWN"),
                   "score": _f("score", 0.0), "confidence": _f("confidence", 0.0),
                   "error_type": _f("error_type", "OTHER"), "error_step": "(truncated)",
                   "feedback": "(truncated critic output)"}
    if not obj:
        return {
            "parse_ok": False, "decision": "REJECT", "completion": "UNKNOWN",
            "score": 0.0, "confidence": 0.0, "error_type": "FORMAT",
            "error_step": "Critic không trả JSON hợp lệ.",
            "feedback": "Critic output format invalid.",
        }

    decision = str(obj.get("decision", "REJECT")).upper()
    completion = str(obj.get("completion", "UNKNOWN")).upper()
    try:
        score = float(obj.get("score", 0.0))
    except Exception:
        score = 0.0
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return {
        "parse_ok": True,
        "decision": decision if decision in {"ACCEPT", "REJECT"} else "REJECT",
        "completion": completion if completion in {"COMPLETE", "INCOMPLETE", "UNKNOWN"} else "UNKNOWN",
        "score": max(0.0, min(1.0, score)),
        "confidence": max(0.0, min(1.0, confidence)),
        "error_type": str(obj.get("error_type", "OTHER")),
        "error_step": str(obj.get("error_step", "")),
        "feedback": str(obj.get("feedback", "")),
    }


def critic_gate(critic: Dict[str, Any]) -> Tuple[str, str, bool]:
    if not critic.get("parse_ok", True):
        return "ACCEPT_WITH_WARNING", "Critic không parse được; bỏ qua critic cho bước này.", False

    score, decision = critic["score"], critic["decision"]
    contradiction = (decision == "REJECT" and score >= ACCEPT_THRESHOLD) or \
                    (decision == "ACCEPT" and score <= 0.2)

    if CRITIC_POLICY == "decision":
        if decision == "REJECT":
            return "HARD_REJECT", f"Critic REJECT (score={score:.2f}).", contradiction
        return "ACCEPT", f"Critic ACCEPT (score={score:.2f}).", contradiction

    if score < HARD_REJECT_THRESHOLD:
        return "HARD_REJECT", f"Critic score thấp ({score:.2f} < {HARD_REJECT_THRESHOLD:.2f}).", contradiction
    if score < ACCEPT_THRESHOLD:
        return "ACCEPT_WITH_WARNING", f"Critic score biên ({score:.2f}).", contradiction
    if decision == "REJECT":
        return "ACCEPT_WITH_WARNING", f"Mâu thuẫn: decision=REJECT nhưng score={score:.2f}.", True
    return "ACCEPT", f"Critic mạnh: score={score:.2f}.", False


# ============================================================
# 7. ANSWER EVALUATION
# ============================================================


def normalize_math_string(s: str) -> str:
    t = _normalize_math_expr(str(s)).lower()
    t = t.replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", "", t)


def evaluate_answer_match(predicted: Optional[str], ground_truth: str) -> bool:
    pred = clean_final_answer(predicted)
    if not pred:
        return False
    gt = clean_final_answer(ground_truth) or str(ground_truth)

    p, g = safe_parse(pred), safe_parse(gt)
    if p is not None and g is not None:
        try:
            diff = sp.simplify(p - g)
            if diff == 0:
                return True
            if diff.is_number:
                return math.isclose(float(sp.N(diff)), 0.0, rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            pass
    return normalize_math_string(pred) == normalize_math_string(gt)


def answer_key(ans: Optional[str]) -> Optional[str]:
    a = clean_final_answer(ans)
    if not a:
        return None
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


# ============================================================
# 8. FINAL VERIFY
# ============================================================


def verify_by_agreement(problem: str, predicted: str, seed: Optional[int]) -> Dict[str, Any]:
    key = answer_key(predicted)
    hits = 0
    for i in range(VERIFY_K):
        _, p = _direct_once(problem, SC_TEMPERATURE, (seed or 0) * 1000 + 500 + i)
        hits += int(key is not None and answer_key(p) == key)
    need = VERIFY_K // 2 + 1
    return {"decision": "PASS" if hits >= need else "FAIL", "score": hits / VERIFY_K,
            "reason": f"{hits}/{VERIFY_K} lời giải độc lập trùng đáp án", "raw": ""}


def final_verify(problem: str, solution: str, predicted: str, seed: Optional[int] = None) -> Dict[str, Any]:
    prompt = f"""
Problem:
{problem}

Generated solution:
{solution}

Extracted FINAL_ANSWER:
{predicted}
""".strip()

    raw = call_llm(
        [
            {"role": "system", "content": FINAL_VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        VERIFIER_MODEL,
        temperature=0.0,
        max_tokens=400,
        seed=seed,
    )
    obj = extract_json_object(raw) or {}
    decision = str(obj.get("decision", "FAIL")).upper()
    try:
        score = float(obj.get("score", 0.0))
    except Exception:
        score = 0.0
    return {
        "decision": "PASS" if decision == "PASS" and score >= 0.70 else "FAIL",
        "score": max(0.0, min(1.0, score)),
        "reason": str(obj.get("reason", "")),
        "raw": raw,
    }


# ============================================================
# 9. BASELINES: DIRECT và SELF-CONSISTENCY
# ============================================================


EXTRACT_PROMPT = (
    "Below is a mathematical solution. Print ONLY the final answer "
    "(a number, a fraction, or an expression), with no explanation. "
    "If the solution does not conclude a final answer, print only: NONE."
)

def extract_answer_llm(problem: str, raw: str) -> Optional[str]:
    out = call_llm(
        [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"Problem:\n{problem}\n\nSolution:\n{raw[-3000:]}"},
        ],
        GENERATOR_MODEL, temperature=0.0, max_tokens=60,
    )
    if not out or out.strip().upper().startswith("NONE"):
        return None
    return clean_final_answer(out.splitlines()[0])


def _direct_once(problem: str, temperature: float, seed: Optional[int],
                 system_prompt: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Single direct CoT solve. Default prompt is English (DIRECT_SYSTEM_PROMPT)."""
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
        )
        raw = strip_tool_json(raw0)
        if raw:
            break
        LLM_STATS["direct_toolcall_retry"] += 1
        nudge = (
            "\n\n(Compute step by step in plain text and mathematical formulas, "
            "one idea per line, then write FINAL_ANSWER; do not call any function.)"
        )
    pred = extract_final_answer(raw)
    if pred is None and raw and EXTRACT_FALLBACK:
        pred = extract_answer_llm(problem, raw)
    return raw, pred


def run_direct(item: Dict[str, str], seed: int = 0) -> Dict[str, Any]:
    start, calls0 = time.time(), LLM_CALLS
    raw, pred = _direct_once(item["problem"], 0.1, seed)
    correct = evaluate_answer_match(pred, item["ground_truth"])
    return {
        "id": item["id"], "category": item["category"], "method": "DIRECT", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred,
        "correct": int(correct), "format_status": "ACCEPT" if raw else "FORMAT_REJECT",
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
        "raw_output": raw, "solution_explanation": raw
    }


def run_self_consistency(item: Dict[str, str], k: int, seed: int = 0) -> Dict[str, Any]:
    start, calls0 = time.time(), LLM_CALLS
    votes: Counter = Counter()
    exemplars: Dict[str, str] = {}
    raw_outputs: Dict[str, str] = {}
    for i in range(k):
        raw, pred = _direct_once(item["problem"], SC_TEMPERATURE, seed * 1000 + i)
        key = answer_key(pred)
        if key is not None:
            votes[key] += 1
            exemplars.setdefault(key, pred)
            raw_outputs.setdefault(key, raw)
    top_key = votes.most_common(1)[0][0] if votes else None
    pred = exemplars[top_key] if top_key else None
    sol_exp = raw_outputs[top_key] if top_key else ""
    correct = evaluate_answer_match(pred, item["ground_truth"])
    return {
        "id": item["id"], "category": item["category"], "method": f"SC@{k}", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred,
        "correct": int(correct), "vote_distribution": json.dumps(dict(votes), ensure_ascii=False),
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
        "solution_explanation": sol_exp
    }


# ============================================================
# 10. STEPWISE / DUAL CONTROLLER
# ============================================================

def probe_final(problem: str, steps: List[str]) -> Optional[str]:
    hist = "\n".join(f"Step {i+1}: {x}" for i, x in enumerate(steps))
    out = call_llm(
        [
            {"role": "system", "content": PROBE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{problem}\n\nSteps:\n{hist}"},
        ],
        GENERATOR_MODEL, temperature=0.0, max_tokens=80,
    )
    if out and "FINAL_ANSWER" in out.upper():
        return extract_final_answer(out)
    return None


def finalize_from_steps(problem: str, steps: List[str]) -> Optional[str]:
    hist = "\n".join(f"Step {i+1}: {x}" for i, x in enumerate(steps)) or "(no steps yet)"
    out = call_llm(
        [
            {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{problem}\n\nIncomplete solution:\n{hist}"},
        ],
        GENERATOR_MODEL, temperature=0.1, max_tokens=500,
    )
    out = strip_tool_json(out or "")
    return extract_final_answer(out) or (extract_answer_llm(problem, out) if out and EXTRACT_FALLBACK else None)


METHOD_SPECS = {
    "STEP_PLAIN": (False, False, False),   
    "STEP_LOCAL": (True, False, False),    
    "STEP_CRITIC": (True, True, False),    
    "DUAL_FULL": (True, True, True),       
}


def _blank_record(step_num: int, attempt: int, is_repair: bool, raw: str) -> Dict[str, Any]:
    return {
        "step": step_num, "attempt": attempt, "is_repair_attempt": is_repair,
        "raw_generator": raw, "normalized_candidate": "", "format_status": "",
        "local_gate": "", "local_error_type": "", "local_feedback": "",
        "raw_critic": "", "critic_decision": "", "critic_completion": "",
        "critic_score": None, "critic_confidence": None, "critic_error_type": "",
        "critic_error_step": "", "critic_feedback": "", "critic_policy": "",
        "critic_contradiction": False, "controller_action": "", "accepted": 0,
    }


def run_stepwise(
    item: Dict[str, str],
    method: str = "DUAL_FULL",
    seed: int = 0,
    fallback_direct: bool = True,
) -> Dict[str, Any]:
    use_local, use_critic, use_verify = METHOD_SPECS[method]
    start, calls0 = time.time(), LLM_CALLS

    valid_steps: List[str] = []
    trace: List[Dict[str, Any]] = []
    c = Counter()  

    final_pred: Optional[str] = None       
    provisional: Optional[str] = None      
    final_votes: Counter = Counter()
    verify_result: Dict[str, Any] = {"decision": "FAIL", "score": 0.0, "reason": "Not run", "raw": ""}
    termination = "MAX_STEPS"
    carry_feedback = ""
    rounds, max_rounds = 0, MAX_STEPS + MAX_FINAL_VERIFY_REPAIRS

    over_budget = False
    while len(valid_steps) < MAX_STEPS and rounds < max_rounds:
        if LLM_CALLS - calls0 >= MAX_CALLS_PER_PROBLEM:
            termination, over_budget = "BUDGET_EXCEEDED", True
            break
        rounds += 1
        step_num = len(valid_steps) + 1
        history_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(valid_steps)) or "None"

        feedback = carry_feedback
        carry_feedback = ""
        last_rejected = ""
        best: Optional[Dict[str, Any]] = None  
        chosen: Optional[Dict[str, Any]] = None

        for attempt in range(1, MAX_RETRIES_PER_STEP + 1):
            if LLM_CALLS - calls0 >= MAX_CALLS_PER_PROBLEM:
                over_budget = True
                break
            is_repair = bool(feedback)
            if is_repair:
                c["repair_calls"] += 1
                system_prompt = REPAIR_SYSTEM_PROMPT
                user_prompt = (
                    f"Problem:\n{item['problem']}\n\nValid history:\n{history_text}\n\n"
                    + (f"Rejected step:\n{last_rejected}\n\n" if last_rejected else "")
                    + f"Feedback requiring repair:\n{feedback}\n\nGenerate ONE corrected replacement step."
                )
            else:
                system_prompt = STEP_GENERATOR_SYSTEM_PROMPT
                user_prompt = (
                    f"Problem:\n{item['problem']}\n\nValid history:\n{history_text}\n\n"
                    "Generate the next step only."
                )

            raw_candidate = call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                GENERATOR_MODEL,
                temperature=0.15 + 0.1 * (attempt - 1),  
                max_tokens=400,
                seed=seed * 100 + step_num * 10 + attempt,
            )
            rec = _blank_record(step_num, attempt, is_repair, raw_candidate)

            # ---------- FORMAT ----------
            fmt_status, candidate = normalize_generator_output(raw_candidate)
            rec["format_status"], rec["normalized_candidate"] = fmt_status, candidate
            if fmt_status == "ACCEPT_ADAPTED":
                c["format_adapted"] += 1
            if fmt_status == "FORMAT_REJECT":
                c["format_rejections"] += 1
                c["rejected_attempts"] += 1
                rec["controller_action"] = "REPAIR_FORMAT"
                rec["local_feedback"] = candidate
                trace.append(rec)
                feedback, last_rejected = candidate, ""
                continue

            # ---------- LOCAL GATE ----------
            if use_local:
                l_dec, l_type, l_fb = local_gate(candidate, valid_steps)
                rec["local_gate"], rec["local_error_type"], rec["local_feedback"] = l_dec, l_type, l_fb
                if l_dec != "ACCEPT":
                    c["local_rejections"] += 1
                    c["rejected_attempts"] += 1
                    rec["controller_action"] = "REPAIR_LOCAL_GATE"
                    trace.append(rec)
                    feedback, last_rejected = l_fb, candidate
                    continue

            # ---------- CRITIC ----------
            critic: Optional[Dict[str, Any]] = None
            policy = "ACCEPT"
            if use_critic:
                critic_prompt = (
                    f"Problem:\n{item['problem']}\n\nValid history:\n{history_text}\n\n"
                    f"Proposed next step:\n{candidate}"
                )
                raw_critic = call_llm(
                    [
                        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                        {"role": "user", "content": critic_prompt},
                    ],
                    CRITIC_MODEL, temperature=0.0, max_tokens=CRITIC_MAX_TOKENS, seed=seed,
                )
                c["critic_calls"] += 1
                critic = parse_critic(raw_critic)
                if not critic["parse_ok"]:
                    c["critic_parse_failures"] += 1
                policy, _reason, contradiction = critic_gate(critic)
                if contradiction:
                    c["critic_contradictions"] += 1
                rec.update({
                    "raw_critic": raw_critic, "critic_decision": critic["decision"],
                    "critic_completion": critic["completion"], "critic_score": critic["score"],
                    "critic_confidence": critic["confidence"], "critic_error_type": critic["error_type"],
                    "critic_error_step": critic["error_step"], "critic_feedback": critic["feedback"],
                    "critic_policy": policy, "critic_contradiction": contradiction,
                })

                if policy == "HARD_REJECT":
                    c["critic_rejections"] += 1
                    c["rejected_attempts"] += 1
                    rec["controller_action"] = "REPAIR_CRITIC"
                    trace.append(rec)
                    if best is None or critic["score"] > best["critic"]["score"]:
                        best = {"candidate": candidate, "critic": critic, "rec": rec}
                    feedback = (
                        f"{critic['feedback']} (score={critic['score']:.2f}, "
                        f"error={critic['error_type']})"
                    ).strip()
                    last_rejected = candidate
                    continue

            chosen = {"candidate": candidate, "critic": critic, "policy": policy, "rec": rec}
            break

        if chosen is None and best is not None and best["critic"]["score"] >= FORCE_ACCEPT_FLOOR:
            c["forced_accepts"] += 1
            chosen = {
                "candidate": best["candidate"], "critic": best["critic"],
                "policy": "FORCED_ACCEPT", "rec": best["rec"],
            }
            best["rec"]["controller_action"] = "FORCED_ACCEPT"
            best["rec"]["critic_policy"] = "FORCED_ACCEPT"

        if chosen is None:
            termination = "BUDGET_EXCEEDED" if over_budget else "STEP_ABORTED"
            break

        candidate, critic, policy, rec = chosen["candidate"], chosen["critic"], chosen["policy"], chosen["rec"]
        rec["accepted"] = 1
        if policy == "ACCEPT_WITH_WARNING":
            c["critic_warning_accepts"] += 1

        candidate_final = extract_final_answer(candidate)
        completion = critic["completion"] if critic else "UNKNOWN"

        if candidate_final is None and len(valid_steps) + 1 >= PROBE_FROM_STEP:
            c["probe_calls"] += 1
            pa = probe_final(item["problem"], valid_steps + [candidate])
            if pa:
                c["probe_finals"] += 1
                candidate_final = pa
                candidate = f"{candidate}\n{FINAL_ANSWER_TAG} {pa}"
                rec["normalized_candidate"] = candidate

        if candidate_final is None:
            valid_steps.append(candidate)
            rec["controller_action"] = rec["controller_action"] or (
                "CONTINUE_WARNING" if policy != "ACCEPT" else "CONTINUE"
            )
            if rec not in trace:
                trace.append(rec)
            continue

        key = answer_key(candidate_final)
        final_votes[key] += 1
        consensus = final_votes[key] >= FINAL_CONSENSUS_VOTES
        is_complete = (not use_critic) or consensus or policy != "FORCED_ACCEPT"

        if not is_complete:
            provisional = candidate_final
            valid_steps.append(strip_final_line(candidate) or candidate)
            rec["controller_action"] = "CONTINUE_PROVISIONAL_FINAL"
            if rec not in trace:
                trace.append(rec)
            continue

        valid_steps.append(candidate)
        final_pred = candidate_final

        if not use_verify:
            rec["controller_action"] = "STOP_FINAL"
            if rec not in trace:
                trace.append(rec)
            termination = "FINAL_ANSWER"
            break

        c["final_verify_calls"] += 1
        if VERIFY_MODE == "sc":
            verify_result = verify_by_agreement(item["problem"], final_pred, seed)
        else:
            verify_result = final_verify(item["problem"], "\n".join(valid_steps), final_pred, seed=seed)
        if verify_result["decision"] == "PASS":
            rec["controller_action"] = "STOP_FINAL_VERIFY_PASS"
            if rec not in trace:
                trace.append(rec)
            termination = "FINAL_VERIFY_PASS"
            break

        c["final_verify_repairs"] += 1
        rec["controller_action"] = "REPAIR_FINAL_VERIFY"
        if rec not in trace:
            trace.append(rec)
        valid_steps.pop()
        final_votes.clear()
        if c["final_verify_repairs"] >= MAX_FINAL_VERIFY_REPAIRS:
            termination = "FINAL_VERIFY_FAILED"
            break
        carry_feedback = "Final Verify failed: " + verify_result.get("reason", "") + \
            " Hãy kiểm tra lại và đưa ra bước sửa."
        final_pred = None

    solution = "\n".join(valid_steps)
    finalized = 0
    if final_pred is None:
        final_pred = provisional or extract_final_answer(solution)
    if final_pred is None:
        c["finalize_calls"] += 1
        fin = finalize_from_steps(item["problem"], valid_steps)
        if fin:
            final_pred, finalized = fin, 1
            if use_verify:
                c["final_verify_calls"] += 1
                verify_result = (
                    verify_by_agreement(item["problem"], final_pred, seed)
                    if VERIFY_MODE == "sc"
                    else final_verify(item["problem"], solution, final_pred, seed=seed)
                )

    used_fallback = 0
    answer_dual = final_pred
    correct_no_fallback = evaluate_answer_match(final_pred, item["ground_truth"])
    if final_pred is None and fallback_direct:
        _, fb_pred = _direct_once(item["problem"], 0.1, seed)
        final_pred, used_fallback = fb_pred, 1

    answer_match = evaluate_answer_match(final_pred, item["ground_truth"])
    final_verified = verify_result["decision"] == "PASS"

    if c["final_verify_calls"] > 0:
        last_pass = final_verified
        verifier_outcome = (
            "TP" if last_pass and correct_no_fallback else
            "FP" if last_pass and not correct_no_fallback else
            "FN" if (not last_pass) and correct_no_fallback else "TN"
        )
    else:
        verifier_outcome = ""

    if answer_match:
        attribution = "NONE"
    elif verifier_outcome == "FP":
        attribution = "VERIFIER_FALSE_POSITIVE"
    elif termination in ("STEP_ABORTED", "BUDGET_EXCEEDED"):
        dominant = max(
            [("FORMAT", c["format_rejections"]), ("LOCAL_GATE", c["local_rejections"]),
             ("CRITIC", c["critic_rejections"])],
            key=lambda x: x[1],
        )[0]
        attribution = f"ABORT_{dominant}"
    elif termination == "FINAL_VERIFY_FAILED":
        attribution = "FINAL_VERIFY_FAILED"
    elif final_pred is None:
        attribution = "GENERATOR_NO_FINAL"
    elif c["critic_calls"] > 0 and c["critic_rejections"] == 0:
        attribution = "CRITIC_MISS"          
    elif c["forced_accepts"] > 0:
        attribution = "GENERATOR_WRONG_FORCED"
    else:
        attribution = "WRONG_ANSWER"

    return {
        "id": item["id"], "category": item["category"], "method": method, "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": final_pred,
        "dual_answer_no_fallback": answer_dual,
        "correct": int(answer_match),                      
        "correct_no_fallback": int(correct_no_fallback),
        "used_fallback": used_fallback, "finalized": finalized,
        "probe_calls": c["probe_calls"], "probe_finals": c["probe_finals"],
        "finalize_calls": c["finalize_calls"],
        "accepted_steps": len(valid_steps),
        "rejected_attempts": c["rejected_attempts"], "critic_calls": c["critic_calls"],
        "repair_calls": c["repair_calls"], "format_rejections": c["format_rejections"],
        "format_adapted": c["format_adapted"], "local_rejections": c["local_rejections"],
        "critic_rejections": c["critic_rejections"],
        "critic_warning_accepts": c["critic_warning_accepts"],
        "critic_contradictions": c["critic_contradictions"],
        "critic_parse_failures": c["critic_parse_failures"],
        "forced_accepts": c["forced_accepts"],
        "final_verify_calls": c["final_verify_calls"],
        "final_verify_repairs": c["final_verify_repairs"],
        "final_verified": int(final_verified), "final_verify_score": verify_result["score"],
        "final_verify_reason": verify_result["reason"], "verifier_outcome": verifier_outcome,
        "termination_reason": termination, "error_attribution": attribution,
        "valid_trace": solution, "trace_json": json.dumps(trace, ensure_ascii=False),
        "solution_explanation": solution,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }


def _critic_whole(problem: str, solution: str, seed: Optional[int]) -> Dict[str, Any]:
    raw = call_llm(
        [
            {"role": "system", "content": BON_CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{problem}\n\nComplete solution:\n{solution[-2500:]}"},
        ],
        CRITIC_MODEL, temperature=0.0, max_tokens=CRITIC_MAX_TOKENS, seed=seed,
    )
    return parse_critic(raw)


def run_best_of_n(item: Dict[str, str], n: int, seed: int = 0) -> Dict[str, Any]:
    start, calls0 = time.time(), LLM_CALLS
    sols = []
    for i in range(n):
        raw, pred = _direct_once(item["problem"], SC_TEMPERATURE, seed * 1000 + i)
        key = answer_key(pred)
        if key is None:
            continue
        loc, _ = safe_sympy_check_equalities(raw)
        critic = _critic_whole(item["problem"], raw, seed)
        ok = loc == "ACCEPT" and critic["parse_ok"] and critic["decision"] == "ACCEPT"
        sols.append({"key": key, "pred": pred, "ok": ok, "local": loc,
                     "decision": critic["decision"], "score": critic["score"], "raw": raw})
    pool = [x for x in sols if x["ok"]] or sols
    votes: Counter = Counter(x["key"] for x in pool)
    pred = None
    sol_exp = ""
    if votes:
        top = max(votes.values())
        best = [x for x in pool if votes[x["key"]] == top]
        best_item = max(best, key=lambda x: x["score"])
        pred = best_item["pred"]
        sol_exp = best_item.get("raw", "")
    correct = evaluate_answer_match(pred, item["ground_truth"])
    return {
        "id": item["id"], "category": item["category"], "method": f"BON@{n}", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred, "correct": int(correct),
        "kept": sum(x["ok"] for x in sols), "sampled": len(sols),
        "trace_json": json.dumps(sols, ensure_ascii=False),
        "solution_explanation": sol_exp,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }


# ----- PAL: model viết code Python -----
_PAL_ALLOWED_IMPORTS = {"math", "itertools", "fractions", "sympy", "collections", "functools", "decimal", "statistics"}
_PAL_BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                     "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit"}


def _pal_static_check(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _PAL_ALLOWED_IMPORTS:
                    return f"import không được phép: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _PAL_ALLOWED_IMPORTS:
                return f"import không được phép: {node.module}"
        elif isinstance(node, ast.Name) and node.id in _PAL_BANNED_NAMES:
            return f"tên bị cấm: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"thuộc tính bị cấm: {node.attr}"
    return None


def run_pal_code(code: str) -> Tuple[Optional[str], str]:
    err = _pal_static_check(code)
    if err:
        return None, err
    wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024))\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (%d, %d))\n" % (PAL_TIMEOUT, PAL_TIMEOUT)
    ) if os.name == "posix" else ""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(wrapper + code)
        try:
            r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                               timeout=PAL_TIMEOUT, cwd=d)
        except subprocess.TimeoutExpired:
            return None, "timeout"
    lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        return None, (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
    return lines[-1], ""


def run_pal(item: Dict[str, str], k: int, seed: int = 0) -> Dict[str, Any]:
    start, calls0 = time.time(), LLM_CALLS
    votes: Counter = Counter()
    exemplars: Dict[str, str] = {}
    codes: Dict[str, str] = {}
    log = []
    for i in range(k):
        raw = call_llm(
            [{"role": "system", "content": PAL_SYSTEM_PROMPT}, {"role": "user", "content": item["problem"]}],
            GENERATOR_MODEL, temperature=0.2 if i == 0 else SC_TEMPERATURE, max_tokens=700,
            seed=seed * 1000 + i,
        )
        m = re.search(r"```(?:python)?\s*(.*?)```", raw or "", flags=re.S)
        code = m.group(1) if m else (raw or "")
        out, err = run_pal_code(code) if code.strip() else (None, "no code")
        ans = clean_final_answer(out) if out else None
        key = answer_key(ans)
        log.append({"answer": ans, "error": err, "code": code})
        if key:
            votes[key] += 1
            exemplars.setdefault(key, ans)
            codes.setdefault(key, code)
    top_key = votes.most_common(1)[0][0] if votes else None
    pred = exemplars[top_key] if top_key else None
    sol_exp = f"```python\n{codes[top_key]}\n```\nProgram Output: {pred}" if top_key else ""
    used_fb = 0
    if pred is None:
        raw_fb, pred = _direct_once(item["problem"], 0.1, seed)
        sol_exp = raw_fb
        used_fb = 1
    correct = evaluate_answer_match(pred, item["ground_truth"])
    return {
        "id": item["id"], "category": item["category"], "method": f"PAL@{k}", "seed": seed,
        "ground_truth": item["ground_truth"], "predicted_answer": pred, "correct": int(correct),
        "used_fallback": used_fb, "vote_distribution": json.dumps(dict(votes), ensure_ascii=False),
        "trace_json": json.dumps(log, ensure_ascii=False),
        "solution_explanation": sol_exp,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }


def convert_pal_to_text_explanation(problem: str, code: str, answer: str) -> str:
    """Gọi LLM để biên dịch mã Code PAL thành bài giải bằng lời văn toán học hoàn chỉnh."""
    user_prompt = (
        f"Bài toán:\n{problem}\n\n"
        f"Thuật toán/Code đã tính đúng ra đáp số:\n{code}\n\n"
        f"Đáp số chính xác: {answer}\n\n"
        f"Hãy trình bày bài giải chi tiết bằng lời văn toán học (không chứa code):"
    )
    explanation = call_llm(
        [
            {"role": "system", "content": PAL_TO_TEXT_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=GENERATOR_MODEL,
        temperature=0.1,
        max_tokens=800,
    )
    return explanation


def format_solution_explanation(
    winning_cand: Optional[Dict[str, Any]],
    all_samples: List[Dict[str, Any]],
    problem: str,
) -> str:
    """Định dạng đầu ra hoàn toàn bằng lời văn giải thích:
    - Nếu câu thắng là CoT -> Trả về văn bản CoT.
    - Nếu câu thắng là PAL -> Tìm CoT cùng đáp án. Nếu không có CoT -> Biên dịch Code PAL sang lời văn.
    """
    if not winning_cand:
        return "Không tìm thấy lời giải phù hợp."

    kind = winning_cand.get("kind", "")
    text = winning_cand.get("text", "")
    ans = winning_cand.get("answer", "")
    key = winning_cand.get("key", "")

    # 1. Trường hợp Candidate thắng là CoT -> Lấy trực tiếp văn bản suy luận
    if kind.startswith("COT"):
        return text

    # 2. Trường hợp Candidate thắng là PAL:
    # Tìm xem trong các mẫu (samples) có bài CoT nào tính ra CÙNG ĐÁP ÁN (key) và hợp lệ không
    same_answer_cots = [
        sm
        for sm in all_samples
        if sm.get("kind", "").startswith("COT")
        and sm.get("key") == key
        and sm.get("local") != "REJECT"
        and sm.get("text")
    ]

    if same_answer_cots:
        # Tận dụng luôn lời giải văn bản của CoT (Tốc độ tối đa, 0 call LLM)
        return same_answer_cots[0].get("text", "")

    # 3. Nếu PAL thắng đơn độc (không có CoT nào ra đúng đáp án này)
    # Gọi Synthesizer để biến Code PAL thành Lời giải văn bản từng bước
    return convert_pal_to_text_explanation(problem, text, ans)


# ============================================================
# 10c. V5: ADAPTIVE HYBRID WITH PAL SELF-DEBUG & CROSS-VALIDATION
# ============================================================
V5_SPECS = {
    "V5": True,          
    "V5_NOCRIT": False,  
    "V5_COT": False,     
}
V5_K0 = 3
V5_KMAX = 7
V5_MARGIN = 2


def _v5_route(problem: str) -> str:
    _COMPUTE_CUES = (
        # Vietnamese
        "số dư", "chữ số tận cùng", "bao nhiêu", "số ước", "ước chung", "đồng dư",
        "xâu nhị phân", "bao nhiêu cách", "nghiệm nguyên", "tập con", "xác suất",
        "đếm", "có bao nhiêu", "chia hết", "tính tổng", "kỳ vọng", "tổng các",
        "hệ số", "giá trị của tích", "tính giá trị",
        # English
        "remainder", "last digit", "last two digit", "how many", "number of divisor",
        "gcd", "congruent", " mod ", "binary string", "number of ways",
        "integer solution", "subset", "probability", "count", "divisible",
        "compute the sum", "expected", "coefficient", "value of the product",
        "evaluate", "compute",
    )
    _SYMBOLIC_CUES = (
        # Vietnamese
        "theo a", "theo x", "theo b", "biểu thức", "giá trị nhỏ nhất",
        "giá trị lớn nhất", "nghiệm thực", "chính xác", "hệ thức",
        # English
        "in terms of", "expression", "minimum value", "maximum value",
        "real solution", "exact", "in a, b", "in a and b",
    )
    p = problem.lower()
    c = sum(k in p for k in _COMPUTE_CUES)
    sy = sum(k in p for k in _SYMBOLIC_CUES)
    return "PAL_FIRST" if c > sy else "COT_FIRST"


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

    if (out is None or err) and max_debug > 0 and code.strip():
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
        # ... phần parse code + run_pal_code + cập nhật out/err/debug_used giữ nguyên
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
        "text": code[:1500], "critic_text": f"Python program:\n{code[:1500]}\nProgram output: {ans}"
    }


def _v5_sample_cot(problem: str, idx: int, cot_idx: int, seed: int, lang: str = "en") -> Dict[str, Any]:
    # Ép buộc 100% CoT sinh bằng Tiếng Anh (Chỉ Tiếng Anh)
    sys_prompt = DIRECT_SYSTEM_PROMPT_EN
    raw, pred = _direct_once(problem, 0.2 if idx == 0 else SC_TEMPERATURE, seed * 1000 + idx,
                             system_prompt=sys_prompt)
    loc, msg = safe_sympy_check_equalities(raw or "")
    full_text = (raw or "").strip()
    return {
        "kind": "COT-en", "answer": pred, "key": answer_key(pred), "exec_ok": True,
        "error": "", "local": loc, "local_msg": msg, "debug_used": False,
        "text": full_text, "critic_text": full_text[-2500:]
    }


def _v5_tally(samples: List[Dict[str, Any]], route: str, use_critic: bool = True):
    pal_valid = {sm["key"] for sm in samples if sm["kind"] == "PAL" and sm["exec_ok"] and sm["key"]}
    cot_valid = {sm["key"] for sm in samples if sm["kind"].startswith("COT") and sm["local"] != "REJECT" and sm["key"]}
    cross_keys = pal_valid.intersection(cot_valid)

    weights: Dict[str, float] = {}
    support: Counter = Counter()
    best: Dict[str, Tuple[float, str]] = {}
    winning_candidates: Dict[str, Dict[str, Any]] = {}

    for sm in samples:
        key = sm["key"]
        if key is None or not sm["exec_ok"]:
            continue
        if sm["local"] == "REJECT":
            continue

        if sm["kind"] == "PAL":
            w = 3.0 if key in cross_keys else (2.5 if route == "PAL_FIRST" else 2.0)
        else:
            w = 1.0
            if key in cross_keys:
                w += 1.0
            if use_critic:
                c = sm.get("critic")
                if c == "REJECT":
                    w *= 0.7
                elif c == "ACCEPT":
                    w *= 1.3

        weights[key] = weights.get(key, 0.0) + w
        support[key] += 1
        if key not in best or w > best[key][0]:
            best[key] = (w, sm["answer"])
            winning_candidates[key] = sm

    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, support, best, cross_keys, winning_candidates


def run_v5(item: Dict[str, str], method: str = "V5", seed: int = 0) -> Dict[str, Any]:
    use_critic = V5_SPECS.get(method, True)
    start, calls0 = time.time(), LLM_CALLS
    problem = item["problem"]
    route = "COT_ONLY" if method == "V5_COT" else _v5_route(problem)
    samples: List[Dict[str, Any]] = []
    critic_calls, debug_calls = 0, 0
    stop_reason = "KMAX"

    cot_n = 0
    pal_n = 0

    # 1. Khởi tạo chỉ tiêu mẫu ban đầu dựa trên Dynamic Task Router / Method
    if method == "V5_COT":
        target_cot = 3
        target_pal = 0
    elif route == "PAL_FIRST":
        target_pal = 2
        target_cot = 1
    else:  # COT_FIRST
        target_cot = 2
        target_pal = 1

    max_samples = 8  # Trần tối đa số mẫu để tránh lãng phí API

    while True:
        # 2. Sinh mẫu cho đủ chỉ tiêu do Router/Method ấn định (hoặc chỉ tiêu mở rộng)
        while cot_n < target_cot or pal_n < target_pal:
            idx = len(samples)
            if cot_n < target_cot:
                sm = _v5_sample_cot(problem, idx, cot_n, seed, lang="en")
                cot_n += 1
                samples.append(sm)
            if pal_n < target_pal:
                sm = _v5_sample_pal(problem, idx, seed, max_debug=1)
                pal_n += 1
                if sm.get("debug_used"):
                    debug_calls += 1
                samples.append(sm)

        # 3. Tính toán trọng số & kiểm tra cờ dừng sớm
        ranked, support, best, cross_keys, win_cands = _v5_tally(samples, route, use_critic=use_critic)

        # Cờ 1: XÁC NHẬN CHÉO (Cả PAL lẫn CoT ra cùng 1 đáp án >= 2 lần)
        if cross_keys:
            top_cross = max(cross_keys, key=lambda k: support[k])
            if support[top_cross] >= 2:
                stop_reason = "CROSS_VALIDATED"
                break

        # Cờ 2: PAL ÁP ĐẢO VÀ CÓ SỰ ĐỒNG THUẬN (Không áp dụng cho V5_COT)
        if method != "V5_COT":
            pal_votes = Counter(sm["key"] for sm in samples if sm["kind"] == "PAL" and sm["exec_ok"] and sm["key"])
            if pal_votes:
                top_pal_key, top_pal_count = pal_votes.most_common(1)[0]
                if top_pal_count >= 2:
                    cot_samples = [s for s in samples if s["kind"].startswith("COT")]
                    if cot_samples:
                        if any(s["key"] == top_pal_key for s in cot_samples):
                            stop_reason = "PAL_CROSS_CONFIRMED"
                            break
                        elif top_pal_count >= 3:
                            stop_reason = "PAL_DOMINANT"
                            break

        # Cờ 3: CHẤM ĐIỂM CRITIC CHO MẪU COT MỚI (NẾU CÓ)
        if use_critic:
            for sm in samples:
                if sm["kind"].startswith("COT") and "critic" not in sm and sm["key"] is not None and sm["local"] != "REJECT":
                    cr = _critic_whole(problem, sm["critic_text"], seed)
                    critic_calls += 1
                    sm["critic"] = cr["decision"] if cr["parse_ok"] else None

        # Cờ 4: CHÊNH LỆCH TRỌNG SỐ (MARGIN)
        ranked, support, best, cross_keys, win_cands = _v5_tally(samples, route, use_critic=use_critic)
        if ranked and support[ranked[0][0]] >= 2:
            w1 = ranked[0][1]
            w2 = ranked[1][1] if len(ranked) > 1 else 0.0
            if (w1 - w2) >= V5_MARGIN:
                stop_reason = "MARGIN"
                break

        # Kiểm tra đạt trần tối đa số mẫu
        if len(samples) >= max_samples:
            break

        # 4. Chưa đủ điều kiện dừng (Bất đồng/Chưa đạt Consensus) -> Tăng chỉ tiêu thêm (+1 CoT, +1 PAL nếu không phải V5_COT)
        target_cot += 1
        if method != "V5_COT":
            target_pal += 1

    winning_key = ranked[0][0] if ranked else None
    pred = best[winning_key][1] if winning_key else None
    win_cand = win_cands.get(winning_key) if winning_key else None
    sol_exp = format_solution_explanation(win_cand, samples, item["problem"])

    if pred is None:
        stop_reason = "NO_ANSWER"

    gt = item["ground_truth"]
    correct = evaluate_answer_match(pred, gt)
    oracle = any(evaluate_answer_match(sm["answer"], gt) for sm in samples if sm["answer"])
    pal_oracle = any(evaluate_answer_match(sm["answer"], gt) for sm in samples if sm["kind"] == "PAL" and sm["answer"])
    local_vetoes = sum(1 for sm in samples if sm["local"] == "REJECT")
    w1 = ranked[0][1] if ranked else 0.0
    w2 = ranked[1][1] if len(ranked) > 1 else 0.0

    slim = [{k: (v[:4000] if isinstance(v, str) and k in ("text", "critic_text") else v) for k, v in sm.items()}
            for sm in samples]
    return {
        "id": item["id"], "category": item["category"], "method": method, "seed": seed,
        "ground_truth": gt, "predicted_answer": pred, "correct": int(correct),
        "oracle_any_correct": int(oracle), "pal_any_correct": int(pal_oracle),
        "route": route, "stop_reason": stop_reason, "n_samples": len(samples),
        "n_cot": sum(sm["kind"].startswith("COT") for sm in samples),
        "n_pal": sum(sm["kind"] == "PAL" for sm in samples),
        "pal_exec_ok": sum(sm["kind"] == "PAL" and sm["exec_ok"] for sm in samples),
        "debug_calls": debug_calls, "critic_calls": critic_calls,
        "local_vetoes": local_vetoes,
        "cross_validated": int(bool(cross_keys)),
        "top_weight": round(w1, 2), "margin": round(w1 - w2, 2),
        "vote_distribution": json.dumps({k: round(v, 2) for k, v in ranked}, ensure_ascii=False),
        "trace_json": json.dumps(slim, ensure_ascii=False),
        "solution_explanation": sol_exp,
        "llm_calls": LLM_CALLS - calls0, "elapsed_sec": round(time.time() - start, 3),
    }

# ============================================================
# 11. THỐNG KÊ / LƯU KẾT QUẢ
# ============================================================


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY V5 (Text Solution Synthesizer)")
    print("=" * 100)
    print(f"{'method':<14}{'n':>5}{'acc%':>8}{'CI95%':>16}{'calls/run':>11}{'time/run(s)':>13}{'tok/run':>10}")
    for method, g in df.groupby("method", sort=False):
        n, k = len(g), int(g["correct"].sum())
        lo, hi = wilson_ci(k, n)
        extra = ""
        if "correct_no_fallback" in g and g["correct_no_fallback"].notna().any():
            extra = (f"   | không-fallback: {100*g['correct_no_fallback'].mean():.1f}% "
                     f"(fallback dùng {int(g['used_fallback'].sum())}/{n})")
        tok = g["total_tokens"].mean() if "total_tokens" in g else float("nan")
        print(f"{method:<14}{n:>5}{100*k/n:>8.1f}{f'[{100*lo:.0f}-{100*hi:.0f}]':>16}"
              f"{g['llm_calls'].mean():>11.1f}{g['elapsed_sec'].mean():>13.1f}{tok:>10.0f}{extra}")

    if "total_tokens" in df.columns:
        print("\nToken (trung bình mỗi bài) và hiệu quả:")
        print(f"{'method':<14}{'prompt':>10}{'completion':>12}{'total':>10}{'tok/đáp-án-đúng':>18}")
        for method, g in df.groupby("method", sort=False):
            right = int(g["correct"].sum())
            per_right = g["total_tokens"].sum() / right if right else float("nan")
            print(f"{method:<14}{g['prompt_tokens'].mean():>10.0f}{g['completion_tokens'].mean():>12.0f}"
                  f"{g['total_tokens'].mean():>10.0f}{per_right:>18.0f}")
        est = int(df["token_estimated_calls"].sum()) if "token_estimated_calls" in df else 0
        if est:
            print(f"  (!) {est} lần gọi không có `usage` từ server -> token được ước lượng thô (~3,2 ký tự/token).")

    if "DIRECT" in set(df["method"]):
        base = df[df["method"] == "DIRECT"].set_index(["id", "seed"])["correct"]
        print("\nSo sánh cặp với DIRECT (thắng / thua / hoà):")
        for method, g in df.groupby("method", sort=False):
            if method == "DIRECT":
                continue
            m = g.set_index(["id", "seed"])["correct"]
            joined = pd.concat([base.rename("b"), m.rename("m")], axis=1, join="inner")
            w = int(((joined.m == 1) & (joined.b == 0)).sum())
            l = int(((joined.m == 0) & (joined.b == 1)).sum())
            print(f"  {method:<14} +{w} / -{l} / ={len(joined) - w - l}")

    v5 = df[df["method"].isin(list(V5_SPECS))]
    if not v5.empty:
        print("\nV5: accuracy vs oracle:")
        for method, g in v5.groupby("method", sort=False):
            dbg_cnt = int(g['debug_calls'].sum()) if 'debug_calls' in g else 0
            veto_cnt = int(g['local_vetoes'].sum()) if 'local_vetoes' in g else 0
            print(f"  [{method}] acc={100*g['correct'].mean():.1f}%  oracle={100*g['oracle_any_correct'].mean():.1f}%  "
                  f"pal_oracle={100*g['pal_any_correct'].mean():.1f}%  mẫu/bài={g['n_samples'].mean():.1f}  "
                  f"PAL chạy được={int(g['pal_exec_ok'].sum())}/{max(1, int(g['n_pal'].sum()))}  "
                  f"SymPy-veto={veto_cnt}  debugs={dbg_cnt}")
            print(f"       stop: {dict(g['stop_reason'].value_counts())} | route: {dict(g['route'].value_counts())}")
    if TOKEN_STATS:
        print(f"\nTổng token cả lần chạy: prompt={TOKEN_STATS['prompt']:,} "
              f"completion={TOKEN_STATS['completion']:,} "
              f"total={TOKEN_STATS['prompt'] + TOKEN_STATS['completion']:,}")
    if LLM_STATS:
        print(f"\nChẩn đoán API: {dict(LLM_STATS)}")
    print("=" * 100)


def _truncate_cells(df: pd.DataFrame, limit: int = 32000) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: v[:limit] if isinstance(v, str) and len(v) > limit else v)
    return out


def save_results(df: pd.DataFrame, dataset: List[Dict[str, str]], prefix: str) -> None:
    pq = {x["id"]: x["problem"] for x in dataset}
    num_cols = [c for c in ("correct", "correct_no_fallback", "used_fallback", "total_tokens") if c in df.columns]
    str_cols = [c for c in ("predicted_answer", "termination_reason", "error_attribution") if c in df.columns]
    agg = {c: "mean" for c in num_cols}
    agg.update({c: (lambda x: "|".join(sorted({str(v) for v in x if pd.notna(v)}))) for c in str_cols})
    g = df.groupby(["id", "category", "method"], sort=False).agg(agg)
    g = g.unstack("method")
    g.columns = [f"{m}__{c}" for c, m in g.columns]
    wide = g.reset_index()
    wide.insert(2, "problem", wide["id"].map(pq))
    gt = df.drop_duplicates("id").set_index("id")["ground_truth"]
    wide.insert(3, "ground_truth", wide["id"].map(gt))
    for m in df["method"].unique():
        if f"{m}__correct" in wide.columns:
            wide[m] = wide[f"{m}__correct"]
    wide.to_csv(f"{prefix}_comparison.csv", index=False, encoding="utf-8-sig")
    df.to_csv(f"{prefix}_raw.csv", index=False, encoding="utf-8-sig")  
    try:
        with pd.ExcelWriter(f"{prefix}.xlsx", engine="openpyxl") as w:
            wide.to_excel(w, sheet_name="Comparison", index=False)
            _truncate_cells(df).to_excel(w, sheet_name="Raw", index=False)
    except Exception as e:
        print(f"[XLSX skipped] {e}")


def load_done(jsonl_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================
# 12. MAIN
# ============================================================


def parse_args():
    ap = argparse.ArgumentParser(description="Generator-Critic benchmark V5")
    ap.add_argument("--dataset", default=None, help="csv/jsonl/json: id,category,problem,ground_truth")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N bài đầu")
    ap.add_argument("--seeds", type=int, default=1, help="số lần chạy lặp (seed 0..N-1)")
    ap.add_argument("--methods", default="V5,V5_NOCRIT,V5_COT",
                    help="DIRECT,SC,BON,PAL,V5,V5_NOCRIT,V5_COT,STEP_PLAIN,STEP_LOCAL,STEP_CRITIC,DUAL_FULL hoặc 'all'")
    ap.add_argument("--bon-n", type=int, default=BON_N, help="số lời giải cho BON (best-of-N + critic)")
    ap.add_argument("--pal-k", type=int, default=PAL_K, help="số chương trình cho PAL")
    ap.add_argument("--sc-k", type=int, default=5, help="số mẫu cho self-consistency")
    ap.add_argument("--no-fallback", action="store_true", help="tắt fallback Direct khi dual không có đáp án")
    ap.add_argument("--resume", action="store_true", help="bỏ qua các (id,method,seed) đã có trong file .jsonl")
    ap.add_argument("--out-prefix", default=OUT_PREFIX)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--generator", default=GENERATOR_MODEL)
    ap.add_argument("--critic", default=CRITIC_MODEL)
    ap.add_argument("--verifier", default=None, help="model Final Verify (mặc định = critic)")
    ap.add_argument("--critic-policy", choices=["decision", "score"], default=CRITIC_POLICY)
    ap.add_argument("--verify-mode", choices=["llm", "sc"], default=VERIFY_MODE,
                    help="llm: Verifier là một LLM; sc: đồng thuận nhiều lời giải Direct (độc lập)")
    ap.add_argument("--verify-k", type=int, default=VERIFY_K)
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS_PER_PROBLEM)
    ap.add_argument("--no-extract-fallback", action="store_true")
    ap.add_argument("--hard-threshold", type=float, default=HARD_REJECT_THRESHOLD)
    ap.add_argument("--accept-threshold", type=float, default=ACCEPT_THRESHOLD)
    return ap.parse_args()


def main():
    global BASE_URL, GENERATOR_MODEL, CRITIC_MODEL, VERIFIER_MODEL
    global HARD_REJECT_THRESHOLD, ACCEPT_THRESHOLD, _client
    global CRITIC_POLICY, VERIFY_MODE, VERIFY_K, MAX_CALLS_PER_PROBLEM, EXTRACT_FALLBACK
    args = parse_args()
    BASE_URL, GENERATOR_MODEL, CRITIC_MODEL = args.base_url, args.generator, args.critic
    VERIFIER_MODEL = args.verifier or args.critic
    HARD_REJECT_THRESHOLD, ACCEPT_THRESHOLD = args.hard_threshold, args.accept_threshold
    CRITIC_POLICY, VERIFY_MODE, VERIFY_K = args.critic_policy, args.verify_mode, args.verify_k
    MAX_CALLS_PER_PROBLEM, EXTRACT_FALLBACK = args.max_calls, not args.no_extract_fallback
    _client = None

    methods = (
        ["DIRECT", "SC", "BON", "PAL", "V5", "V5_NOCRIT", "V5_COT", "STEP_PLAIN", "STEP_LOCAL", "STEP_CRITIC", "DUAL_FULL"]
        if args.methods.strip().lower() == "all"
        else [m.strip() for m in args.methods.split(",") if m.strip()]
    )
    for m in methods:
        if m not in METHOD_SPECS and m not in V5_SPECS and m not in ("DIRECT", "SC", "BON", "PAL"):
            raise SystemExit(f"Method không hợp lệ: {m}")

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    print("=" * 100)
    print("GENERATOR-CRITIC V5 (Text Solution Synthesizer)")
    print(f"Generator: {GENERATOR_MODEL} | Critic: {CRITIC_MODEL} | Verifier: {VERIFIER_MODEL}")
    print(f"Endpoint : {BASE_URL} | problems={len(dataset)} | seeds={args.seeds} | methods={methods}")
    print(f"Critic policy={CRITIC_POLICY} | verify={VERIFY_MODE} | max_calls/problem={MAX_CALLS_PER_PROBLEM}")
    print("=" * 100)

    try:
        ids = [m.id for m in get_client().models.list().data]
        print("LM Studio ONLINE | Models:", ids)
    except Exception as e:
        print("LM Studio connection error:", e)
        return

    jsonl_path = f"{args.out_prefix}_raw.jsonl"
    results: List[Dict[str, Any]] = load_done(jsonl_path) if args.resume else []
    if not args.resume and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    done = {(r["id"], r["method"], r["seed"]) for r in results}

    total = len(dataset) * len(methods) * args.seeds
    n = 0
    for seed in range(args.seeds):
        for method in methods:
            label = {"SC": f"SC@{args.sc_k}", "BON": f"BON@{args.bon_n}", "PAL": f"PAL@{args.pal_k}"}.get(method, method)
            for item in dataset:
                n += 1
                if (item["id"], label, seed) in done:
                    continue
                tok0 = token_snapshot()
                if method == "DIRECT":
                    r = run_direct(item, seed)
                elif method == "SC":
                    r = run_self_consistency(item, args.sc_k, seed)
                elif method == "BON":
                    r = run_best_of_n(item, args.bon_n, seed)
                elif method == "PAL":
                    r = run_pal(item, args.pal_k, seed)
                elif method in V5_SPECS:
                    r = run_v5(item, method, seed)
                else:
                    r = run_stepwise(item, method, seed, fallback_direct=not args.no_fallback)
                r.update(token_delta(tok0))
                results.append(r)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"[{n}/{total}] s{seed} {label:<12} {item['id']} | pred={r['predicted_answer']} | "
                      f"GT={item['ground_truth']} | correct={r['correct']} | calls={r['llm_calls']} | "
                      f"tok={r['total_tokens']} | "
                      f"{r.get('termination_reason', '')} {r.get('error_attribution', '')}")
                
                if "solution_explanation" in r:
                    print("\n--- LỜI GIẢI CHI TIẾT ---")
                    print(r["solution_explanation"])
                    print(f"\nFINAL_ANSWER: {r['predicted_answer']}\n" + "-"*40)

    df = pd.DataFrame(results)
    save_results(df, dataset, args.out_prefix)
    print_summary(df)
    print(f"\nCSV  : {os.path.abspath(args.out_prefix + '_comparison.csv')}")
    print(f"RAW  : {os.path.abspath(args.out_prefix + '_raw.csv')}  (có trace_json)")
    print(f"JSONL: {os.path.abspath(jsonl_path)}")


if __name__ == "__main__":
    main()
