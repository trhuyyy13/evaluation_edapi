import json
import torch
import csv
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# THAM SỐ VÀ ĐƯỜNG DẪN CẤU HÌNH (CONFIGS)
# ==========================================
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_PATH = "deepseek-ai/deepseek-coder-1.3b-instruct"

BASE_DIR = os.path.dirname(__file__)
STATUS = "simnpo"

PRETRAINED_PATH = os.path.join(BASE_DIR, "deepseek_unlearned_simnpo")
INPUT_FILE_PATH = os.path.join(BASE_DIR, "data", "edapi_data_processed.json")

DEP_JSON = os.path.join(BASE_DIR, "data", "stuff", f"dep_api_{STATUS}.json")
REP_JSON = os.path.join(BASE_DIR, "data", "stuff", f"rep_api_{STATUS}.json")
MIS_JSON = os.path.join(BASE_DIR, "data", "stuff", f"mis_api_{STATUS}.json")

DEP_CSV = os.path.join(BASE_DIR, "csv", f"dep_api_{STATUS}.csv")
REP_CSV = os.path.join(BASE_DIR, "csv", f"rep_api_{STATUS}.csv")
MIS_CSV = os.path.join(BASE_DIR, "csv", f"mis_api_{STATUS}.csv")

# Đảm bảo các thư mục đầu ra tồn tại trước khi chạy
os.makedirs(os.path.dirname(DEP_JSON), exist_ok=True)
os.makedirs(os.path.dirname(DEP_CSV), exist_ok=True)

# Các tham số inference
BATCH_SIZE = 64  # Tùy chỉnh (4, 8, 16) tùy theo VRAM GPU của bạn
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 0.8
MAX_PROMPT_LENGTH = 2048

# ==========================================
# CÁC HÀM TIỆN ÍCH (UTILITIES)
# ==========================================
def check_api_usage(generated_line, target_api, alias_dict):
    """Kiểm tra xem API sinh ra có khớp với target hay không bằng Regex để tránh nhận diện chuỗi con."""
    for key, value in alias_dict.items():
        if value == target_api:
            pattern = r'\b' + re.escape(key) + r'\b'
            
            if re.search(pattern, generated_line):
                return True
    return False

# ==========================================
# LUỒNG CHẠY CHÍNH (MAIN FUNCTION)
# ==========================================
def main():
    # 1. Cài đặt Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang sử dụng thiết bị: {device}")

    # 2. Khởi tạo Tokenizer
    print(f"[*] Đang tải tokenizer từ: {PRETRAINED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # BẮT BUỘC: Left-padding khi chạy batch generation cho model Decoder-only
    tokenizer.padding_side = "left"

    # 3. Khởi tạo Model
    print(f"[*] Đang tải model từ: {PRETRAINED_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 4. Tải dữ liệu
    print(f"[*] Đang đọc file dữ liệu: {INPUT_FILE_PATH}")
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # data = data[:1000]

    num_samples = len(data)
    num_good = 0
    num_bad = 0
    mismatched_results = []
    old_api_cases = []
    new_api_cases = []

    # 5. Bắt đầu vòng lặp xử lý (CHẠY BATCH)
    pbar = tqdm(range(0, num_samples, BATCH_SIZE), desc="Processing Batches")

    for i in pbar:
        # Lấy ra 1 batch dữ liệu
        batch_data = data[i : i + BATCH_SIZE]
        
        batch_prompts = []
        batch_meta = []

        # Chuẩn bị prompt cho toàn bộ batch
        for item in batch_data:
            code_context = item.get("prompt", "")
            
            # Cắt ngắn nếu quá dài
            encoded_context = tokenizer.encode(code_context, add_special_tokens=False)
            if len(encoded_context) > MAX_PROMPT_LENGTH - 200:
                encoded_context = encoded_context[-(MAX_PROMPT_LENGTH - 200):]
                code_context = tokenizer.decode(encoded_context)

            # Thay thế bằng prompt mới của bạn kết hợp mẹo mở thẻ code
            prompt_text = (
                f"Complete and output the next line for the following Python function:\n"
                f"```python\n"
                f"{code_context}"
            )
            
            batch_prompts.append(prompt_text)
            batch_meta.append({
                "code_context": code_context,
                "deprecated_api": item.get("deprecated_api", ""),
                "replacement_api": item.get("replacement_api", ""),
                "alias_dict": item.get("alias_dict", {})
            })

        # Tokenize toàn bộ batch cùng lúc
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH
        ).to(device)

        # Inference cho cả batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=(TEMPERATURE > 0),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE if TEMPERATURE > 0 else 1.0, 
                top_p=TOP_P if TEMPERATURE > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Trích xuất phần text sinh ra
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Đánh giá kết quả cho từng sample trong batch
        for idx, generated_text in enumerate(generated_texts):
            meta = batch_meta[idx]
            global_id = i + idx # ID thật của sample trong data gốc
            
            clean_text = generated_text.replace("```python", "").replace("```", "").strip()
            first_line_generated = clean_text.split('\n')[0]

            # Nối dòng cuối của prompt với code gen ra để tạo context hoàn chỉnh
            last_line_of_prompt = meta["code_context"].split('\n')[-1]
            full_line_to_check = last_line_of_prompt + first_line_generated

            # Đưa full_line_to_check vào đánh giá thay vì first_line
            is_replacement = check_api_usage(full_line_to_check, meta["replacement_api"], meta["alias_dict"])
            is_deprecated = check_api_usage(full_line_to_check, meta["deprecated_api"], meta["alias_dict"])

            # Tạo record chuẩn Dictionary
            record = {
                "id": global_id,
                "prompt": meta["code_context"],
                "deprecated_api": meta["deprecated_api"],
                "replacement_api": meta["replacement_api"],
                "generated_content": full_line_to_check # Lưu lại chuỗi hoàn chỉnh để dễ debug
            }

            if is_replacement:
                num_good += 1
                new_api_cases.append(record)
            elif is_deprecated:
                num_bad += 1
                old_api_cases.append(record)
            else:
                mismatched_results.append(record)
                
        # Cập nhật thông số lên thanh tiến trình (tqdm)
        pbar.set_postfix({
            "good": num_good,
            "bad": num_bad,
            "mismatch": len(mismatched_results)
        })

    # 6. Lưu kết quả ra file
    print("\n[*] Đang lưu các file kết quả...")
    
    # Lưu định dạng JSON
    with open(MIS_JSON, "w", encoding="utf-8") as f:
        json.dump(mismatched_results, f, ensure_ascii=False, indent=4)
    
    with open(REP_JSON, "w", encoding="utf-8") as f:
        json.dump(new_api_cases, f, ensure_ascii=False, indent=4)
        
    with open(DEP_JSON, "w", encoding="utf-8") as f:
        json.dump(old_api_cases, f, ensure_ascii=False, indent=4)

    # Lưu định dạng CSV (Sử dụng DictWriter)
    csv_headers = ["id", "prompt", "deprecated_api", "replacement_api", "generated_content"]

    with open(MIS_CSV, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(mismatched_results)

    with open(REP_CSV, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(new_api_cases)
        
    with open(DEP_CSV, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(old_api_cases)

    # 7. In tổng kết
    print("\n================ TỔNG KẾT ================")
    print(f"Số lượng mẫu lệch (Mismatched)   : {len(mismatched_results)}/{num_samples}")
    print(f"Số lượng dùng API mới (Good)     : {num_good}/{num_samples}")
    print(f"Số lượng dùng API cũ (Bad)       : {num_bad}/{num_samples}")
    print("==========================================")


if __name__ == "__main__":
    main()