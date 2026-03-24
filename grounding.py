import sys
import os
import json
import re
import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- STEP 1 & CẤU HÌNH ---
TARGET_SCHEMA = {
    "company_name": "",
    "business_area": "",
    "address": "",
    "year_founded": "",
    "website": ""
}

def load_local_model():
    print("⏳ Đang tải mô hình Qwen2.5-1.5B-Instruct...")
    model_id = r"D:\workspaces\model\Qwen\Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    
    print(f"🚀 Khởi tạo thành công! Model đang chạy trên thiết bị: {model.device}")
    return tokenizer, model

# --- STEP 2: LẤY VÀ LÀM SẠCH DỮ LIỆU HTML ---
def get_website_text(url):
    print(f"🌐 Đang truy cập URL: {url}")
    html_content = ""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        print(f"⚠️ Lỗi truy cập/tải trang ({e}).")
        print("Fallback: Chuyển sang đọc mock_page.html ở local...")
        try:
            with open("mock_page.html", "r", encoding="utf-8") as f:
                html_content = f.read()
        except FileNotFoundError:
            print("❌ Không tìm thấy mock_page.html. Dừng chương trình.")
            sys.exit(1)

    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style", "nav", "footer"]):
        script.extract() 
    
    text = soup.get_text(separator=' ', strip=True)
    return text[:5000]

# --- STEP 3 & 4: DÙNG AI TRÍCH XUẤT VÀ OUTPUT JSON ---
def extract_information(text, tokenizer, model):
    print("🧠 AI đang suy luận và trích xuất thông tin...\n")
    print("-" * 50)
    
    prompt = f"""Bạn là một chuyên gia trích xuất dữ liệu. Hãy đọc đoạn văn bản sau và trích xuất các thông tin về công ty:
    1. Tên công ty (company_name)
    2. Lĩnh vực kinh doanh (business_area)
    3. Địa chỉ (address)
    4. Năm thành lập (year_founded)
    5. Website (website)
    
    Yêu cầu cực kỳ quan trọng: CHỈ trả về một block JSON hợp lệ. Không giải thích, không thêm bất kỳ văn bản nào khác ngoài JSON. Nếu không tìm thấy thông tin nào, hãy để giá trị là null.
    
    Văn bản cần xử lý:
    {text}
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful data extraction assistant that only outputs valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.1
    ) 
    
    print("✅ Đã suy luận xong!")
    print("-" * 50)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def clean_and_parse_json(ai_response):
    match = re.search(r'\{.*\}', ai_response.replace('\n', ''), re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"error": "Không thể parse JSON từ model output", "raw_output": ai_response}

# --- HÀM MỚI: GHI DỮ LIỆU VÀO INFO.MD ---
def save_to_info_md(new_data, filepath="info.md"):
    if "error" in new_data:
        print("❌ JSON không hợp lệ, bỏ qua bước lưu file.")
        return

    existing_data = []
    
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_data = json.loads(content)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data] 
        except Exception as e:
            print(f"⚠️ File {filepath} đang rỗng hoặc format không chuẩn. Tạo danh sách mới.")

    existing_data.append(new_data)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    
    print(f"💾 Đã lưu trữ thành công dữ liệu vào file '{filepath}'!")

# --- MAIN FLOW ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách sử dụng: python grounding.py {URL} data_base.md")
        sys.exit(1)
        
    url = sys.argv[1]
    db_file = sys.argv[2] 
    
    tokenizer, model = load_local_model()
    
    raw_text = get_website_text(url)
    ai_output = extract_information(raw_text, tokenizer, model)
    final_json = clean_and_parse_json(ai_output)
    
    print("\n✅ KẾT QUẢ TRÍCH XUẤT FINAL:")
    print(json.dumps(final_json, indent=4, ensure_ascii=False))
    
    print("\n🔄 Đang xử lý lưu file...")
    save_to_info_md(final_json)
