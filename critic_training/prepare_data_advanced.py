import json
import re
import os
from typing import List, Dict
from datasets import load_dataset
from openai import OpenAI

# ==========================================
# CẤU HÌNH API VÀ ĐƯỜNG DẪN LOCAL
# ==========================================
# 1. Đường dẫn tới thư mục chứa dữ liệu PRM800K raw của bạn
PRM800K_RAW_DIR = r"prm800k_raw"

# 2. Kết nối tới LM Studio đang chạy ở localhost:1234
local_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL_NAME = "llama-3.2-3b-instruct" 

PRM_SYSTEM_PROMPT = (
    "You are a Value Network Critic for a math solver. Evaluate the current step against the problem and history.\n"
    "1. Verify if the math calculation is correct and the logic moves efficiently toward the final answer.\n"
    "2. Score the step from 0.0 (completely wrong) to 1.0 (perfectly correct and optimal).\n"
    "Output EXACTLY in this format:\n"
    "Score: [number]\n"
    "<hint> [Optional brief explanation if score < 0.7] </hint>"
)

def extract_final_answer(text: str) -> str:
    match = re.search(r"####\s*(.*)", text)
    if match:
        ans = match.group(1).strip()
        # Loại bỏ các thẻ như [answer: $10] -> còn lại $10
        ans = re.sub(r'\[.*?\]', '', ans) 
        # Loại bỏ toàn bộ ký tự không phải là số, dấu chấm thập phân hoặc dấu trừ
        ans = re.sub(r'[^\d\.\-]', '', ans) 
        return ans.strip()
    return ""

# ==========================================
# PHẦN 1: BỘ PARSER CHO PRM800K RAW LOCAL
# ==========================================
def parse_local_prm800k(max_samples_per_file: int = 5000) -> List[Dict]:
    print(f"Đang đọc dữ liệu PRM800K từ: {PRM800K_RAW_DIR}...")
    prm_data = []
    
    files_to_read = ["phase1_train.jsonl", "phase2_train.jsonl"]
    
    for filename in files_to_read:
        filepath = os.path.join(PRM800K_RAW_DIR, filename)
        if not os.path.exists(filepath):
            print(f" -> [BỎ QUA] Không tìm thấy file: {filepath}")
            continue
            
        print(f" -> Đang xử lý: {filename}...")
        count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= max_samples_per_file:
                    break
                try:
                    item = json.loads(line.strip())
                    
                    # Trích xuất problem (Cấu trúc của OpenAI raw)
                    problem = ""
                    if "question" in item and "problem" in item["question"]:
                        problem = item["question"]["problem"]
                    elif "problem" in item:
                        problem = item["problem"]
                    else:
                        continue
                    
                    # Bóc tách cấu trúc cây bước giải
                    if "label" in item and "steps" in item["label"]:
                        history = []
                        for step in item["label"]["steps"]:
                            best_text = None
                            
                            if "completions" in step:
                                for comp in step["completions"]:
                                    text = comp.get("text", "").strip()
                                    rating = comp.get("rating", 0) # 1: Tốt, -1: Sai, 0: Neutral
                                    
                                    if not text: continue
                                    
                                    if rating == 1:
                                        score_str = "Score: 1.0"
                                        # Giữ lại một bước đúng để làm history cho bước tiếp theo
                                        if best_text is None: best_text = text 
                                    elif rating == -1:
                                        score_str = "Score: 0.1\n<hint> Logic or calculation error detected in this step. </hint>"
                                    else:
                                        score_str = "Score: 0.5"
                                        
                                    prm_data.append({
                                        "instruction": PRM_SYSTEM_PROMPT,
                                        "input": f"Problem: {problem}\nSolution Process so far:\n{chr(10).join(history)}\nCurrent Step to Evaluate:\n{text}",
                                        "output": score_str
                                    })
                            
                            if best_text:
                                history.append(best_text)
                            else:
                                break # Nếu không có bước nào đúng, ngừng nối history
                    count += 1
                except Exception as e:
                    continue
                    
        print(f" -> Đã trích xuất {len(prm_data)} mẫu (Cộng dồn).")
    
    return prm_data

# ==========================================
# PHẦN 2: ON-POLICY DATA ENGINE (100% LOCAL)
# ==========================================
def generate_on_policy_data(max_problems: int = 150) -> List[Dict]:
    print(f"\nĐang chạy On-Policy Data Engine bằng {MODEL_NAME} (Local LM Studio)...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(" -> Lỗi tải GSM8K cho On-policy. Bỏ qua bước này:", e)
        return []
        
    on_policy_data = []
    
    for i, item in enumerate(dataset):
        if i >= max_problems:
            break
            
        problem = item["question"]
        ground_truth = extract_final_answer(item["answer"])
        
        try:
            response = local_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Solve the problem step by step. Use <thinking> tags for reasoning. End your response with '#### ' followed ONLY by the final number (e.g., #### 72). Do NOT write '[answer]' or units."},
                    {"role": "user", "content": problem}
                ],
                temperature=0.7,
                max_tokens=800
            )
            pred_solution = response.choices[0].message.content
        except Exception as e:
            print(f" -> Lỗi gọi API ở bài {i}: {e}. (Hãy chắc chắn đã Start Server trên LM Studio)")
            continue
            
        pred_ans = extract_final_answer(pred_solution)
        steps = [s.strip() for s in pred_solution.split('\n') if s.strip()]
        
        # Nếu mô hình tự giải sai -> Dùng LLM tự bắt lỗi chính mình
        if pred_ans and pred_ans != ground_truth:
            print(f"[-] Bài {i}: Giải SAI (Đoán: {pred_ans} | Thực tế: {ground_truth}) -> Phân tích...")
            
            teacher_prompt = f"""
            You are an expert math teacher. A student tried to solve this problem but got the wrong answer.
            Problem: {problem}
            Correct Answer: {ground_truth}
            
            Student's Solution:
            {pred_solution}
            
            Identify the EXACT FIRST STEP where the student's logic or math went wrong.
            Respond strictly in this JSON format:
            {{
                "history": ["valid step 1", "valid step 2"],
                "faulty_step": "The exact step that is wrong",
                "hint": "A short hint explaining why it's wrong without revealing the final answer"
            }}
            """
            
            try:
                eval_res = local_client.chat.completions.create(
                    model=MODEL_NAME,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": teacher_prompt}],
                    temperature=0.1
                )
                
                feedback = json.loads(eval_res.choices[0].message.content)
                history_str = "\n".join(feedback.get("history", []))
                faulty_step = feedback.get("faulty_step", "")
                hint = feedback.get("hint", "Calculation or logic error detected.")
                
                if faulty_step:
                    on_policy_data.append({
                        "instruction": PRM_SYSTEM_PROMPT,
                        "input": f"Problem: {problem}\nSolution Process so far:\n{history_str}\nCurrent Step to Evaluate:\n{faulty_step}",
                        "output": f"Score: 0.1\n<hint> {hint} </hint>"
                    })
            except Exception as e:
                pass

    print(f" -> Đã trích xuất {len(on_policy_data)} mẫu On-Policy.")
    return on_policy_data

# ==========================================
# THỰC THI VÀ XUẤT FILE
# ==========================================
def main():
    final_dataset = []
    
    # 1. Thu thập dữ liệu từ PRM800K Local
    prm_samples = parse_local_prm800k(max_samples_per_file=5000)
    final_dataset.extend(prm_samples)
    
    # 2. Thu thập dữ liệu lỗi thực tế (On-policy qua LM Studio)
    on_policy_samples = generate_on_policy_data(max_problems=150)
    final_dataset.extend(on_policy_samples)
    
    # 3. Lưu file
    output_file = "prm_step_dataset_v4_advanced.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in final_dataset:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    print(f"\n=================================================")
    print(f"[HOÀN TẤT] Đã lưu tổng cộng {len(final_dataset)} mẫu dữ liệu vào {output_file}.")
    print(f"Bây giờ bạn có thể chạy lệnh: python train_critic_hf.py")

if __name__ == "__main__":
    main()