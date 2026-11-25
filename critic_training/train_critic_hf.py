# ==========================================================
# BẢN VÁ LỖI TỔNG HỢP CHO PYTORCH NIGHTLY
# ==========================================================
import torch
import torch.utils._pytree
import torch.library
import torch.distributed.fsdp

# 1. Vá lỗi register_constant (đánh lừa torchao)
if not hasattr(torch.utils._pytree, 'register_constant'):
    torch.utils._pytree.register_constant = lambda x: x

# 2. Vá lỗi float4_e2m1fn_x2 (đánh lừa torchao)
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = None 

# 3. Vá lỗi triton_op và wrap_triton (đánh lừa torch.library)
if not hasattr(torch.library, 'triton_op'):
    torch.library.triton_op = lambda *args, **kwargs: lambda func: func
if not hasattr(torch.library, 'wrap_triton'):
    torch.library.wrap_triton = lambda *args, **kwargs: lambda func: func

# 4. Vá lỗi FSDPModule (đánh lừa TRL)
if not hasattr(torch.distributed.fsdp, 'FSDPModule'):
    class DummyFSDPModule: pass
    torch.distributed.fsdp.FSDPModule = DummyFSDPModule
# ==========================================================

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# ==========================================================
# 5. Vá lỗi Dynamo compiler set_stance TRỰC TIẾP VÀO LÕI UNSLOTH
# ==========================================================
import unsloth.models.vision
if hasattr(unsloth.models.vision, 'torch_compiler_set_stance'):
    orig_set_stance = unsloth.models.vision.torch_compiler_set_stance
    def patched_set_stance(*args, **kwargs):
        kwargs.pop('skip_guard_eval_unsafe', None)
        if orig_set_stance:
            return orig_set_stance(*args, **kwargs)
    unsloth.models.vision.torch_compiler_set_stance = patched_set_stance
# ==========================================================

# 1. CẤU HÌNH THÔNG SỐ VÀ GIỚI HẠN VRAM
max_seq_length = 1024
dtype = torch.bfloat16
load_in_4bit = True  # Lượng tử hóa 4-bit

# 2. TẢI MÔ HÌNH VÀ TOKENIZER BẰNG UNSLOTH
model_id = "unsloth/Llama-3.2-3B-Instruct" 

print("Đang tải Mô hình & Tokenizer bằng Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Llama 3 cần set pad_token cho quá trình train
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. CẤU HÌNH LORA SIÊU TỐI ƯU
print("Đang chuẩn bị bộ trọng số LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 4. CHUẨN BỊ DỮ LIỆU BẢO TOÀN SYSTEM PROMPT
def formatting_prompts_func(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        messages = [
            {"role": "system", "content": inst},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": out}
        ]
        # [CẢI TIẾN 1]: Chỉ gộp chuỗi bằng chat_template, để SFTTrainer tự động cắt gọt (Truncation)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

print("Đang xử lý dữ liệu...")
dataset = load_dataset("json", data_files="prm_step_dataset_v4_advanced.jsonl", split="train")

# Để test nhanh, bạn có thể uncomment dòng dưới để chạy thử 3000 mẫu
# dataset = dataset.select(range(3000))
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

# 5. CẤU HÌNH HUẤN LUYỆN
training_args = TrainingArguments(
    output_dir="./dualref_critic_outputs",
    per_device_train_batch_size=4,       # Tăng lên 4 nhờ 16GB VRAM
    gradient_accumulation_steps=4,       # Giảm xuống 4 để bù trừ
    num_train_epochs=1,                  # 1 Epoch là đủ cho 60k data
    save_strategy="steps",               # Lưu dự phòng định kỳ
    save_steps=500,                      
    save_total_limit=3,                  
    learning_rate=2e-4,
    fp16=False,                        
    bf16=True,                           # Hoàn hảo cho RTX 50-series
    logging_steps=10,
    optim="adamw_8bit",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=max_seq_length,
    dataset_text_field="text", # [CẢI TIẾN 2]: Chỉ rõ trường dữ liệu để SFTTrainer tự động tokenize và truncate
    packing=False, # Tránh việc trộn lẫn (pack) nhiều câu hỏi lại với nhau gây nhiễu
)

# ==========================================================
# CƠ CHẾ MASK LÕI PRM: CHỈ TÍNH LOSS TRÊN NHÃN ĐÁNH GIÁ (RESPONSE ONLY)
# ==========================================================
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
# ==========================================================

# 6. BẮT ĐẦU HUẤN LUYỆN
print("BẮT ĐẦU HUẤN LUYỆN...")
trainer.train()

# 7. LƯU MÔ HÌNH SAU KHI HUẤN LUYỆN
print("Đang lưu trọng số LoRA...")
model.save_pretrained("dualref_critic_lora")
tokenizer.save_pretrained("dualref_critic_lora")
print("HOÀN TẤT! Đã lưu trọng số tại thư mục 'dualref_critic_lora'.")