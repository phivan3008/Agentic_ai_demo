import json
import re
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH MODEL (Qwen2.5-3B-Instruct)
# ==========================================
# Thay bằng đường dẫn thư mục local chứa model của bạn nếu cần
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct" 

print("--> [System] Đang khởi tạo Model và Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto" # Tự động phân bổ lên GPU nếu có
)

# ==========================================
# 2. XÂY DỰNG DATA SPACE (TỪ HTML)
# ==========================================
def build_data_space_from_url(url):
    print(f"--> [System] Đang tải và trích xuất HTML từ: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Loại bỏ các thẻ không chứa nội dung text hữu ích
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        print("--> [System] Trích xuất HTML thành công. Data Space đã sẵn sàng.")
        return text
    except Exception as e:
        print(f"--> [Error] Không thể tải dữ liệu: {e}. Sẽ sử dụng dữ liệu giả lập (Mock Data).")
        return ""

# Cố gắng cào một URL thực tế. Nếu thất bại (do URL lỗi hoặc bị block), fallback về chuỗi text mặc định.
TARGET_URL = "https://example.com/weather-news" 
HTML_DATA_SPACE = build_data_space_from_url(TARGET_URL)

if not HTML_DATA_SPACE:
    HTML_DATA_SPACE = """
    Bản tin thời tiết thế giới hôm nay. 
    Tại Châu Á, Tokyo đang có nắng đẹp (sunny) với mức nhiệt độ vào khoảng 22 độ C. 
    Trong khi đó tại khu vực Đông Nam Á, Hanoi trời nhiều mây và có mưa rào (rainy), nhiệt độ hiện tại là 25 độ C.
    Chuyển sang Châu Âu, London tiếp tục duy trì kiểu thời tiết âm u (cloudy) với 15 độ C, thỉnh thoảng có sương mù.
    """

# ==========================================
# 3. ĐỊNH NGHĨA TOOL (MINI-RETRIEVER)
# ==========================================
def get_weather(city_name):
    """Tìm kiếm thông tin thành phố trong Data Space đã cào từ Web"""
    city = city_name.strip().lower()
    data_space_lower = HTML_DATA_SPACE.lower()
    
    # Thuật toán tìm kiếm chuỗi cơ bản
    idx = data_space_lower.find(city)
    
    if idx != -1:
        # Cắt một đoạn văn bản (context) xung quanh từ khóa tìm thấy
        start = max(0, idx - 40)
        end = min(len(HTML_DATA_SPACE), idx + 100)
        context_chunk = HTML_DATA_SPACE[start:end]
        
        return {
            "status": "success", 
            "city": city, 
            "extracted_context": f"...{context_chunk}..."
        }
    else:
        return {"status": "error", "message": "No data found in the website context"}

# ==========================================
# 4. THIẾT LẬP AGENT (SYSTEM PROMPT & REACT LOOP)
# ==========================================
system_prompt = """You are an AI assistant that can access external tools to answer user questions.
You have ONE tool available:
- Name: get_weather
- Purpose: Get the current weather and temperature context for a specific city from a database.
- JSON Call Format: {"action": "get_weather", "city": "<city_name>"}

RULES:
1. If the user asks about the weather, you MUST output ONLY the JSON Call Format to get the data. Do not add any conversational text or markdown blocks.
2. The system will provide you with the tool's result (which may contain a text snippet).
3. Analyze the result:
   - If the result contains sufficient data (weather and temperature), generate a natural language response (e.g., "Today, in Tokyo, it's sunny, temperature is 22 degree C.").
   - If the result says "error" or "No data found", or you cannot fulfill the request, you MUST reply EXACTLY with: "I don't have any data about it."
"""

def extract_json(text):
    """Bóc tách JSON thuần túy, đề phòng LLM trả về markdown block như ```json { ... } ```"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def run_weather_agent(user_query, max_iterations=3):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    for step in range(max_iterations):
        # Format input theo template chuẩn của Qwen
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Sinh text với temperature cực thấp để đảm bảo tính chính xác định dạng
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.1, 
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Ghi nhận phản hồi vào history
        messages.append({"role": "assistant", "content": response_text})
        
        # Xử lý ReAct logic
        clean_text = extract_json(response_text)
        if clean_text.startswith("{") and "action" in clean_text:
            try:
                tool_call = json.loads(clean_text)
                if tool_call.get("action") == "get_weather":
                    city_to_query = tool_call.get("city", "")
                    print(f"  [Step {step+1}] --> Agent quyết định gọi hàm get_weather cho: '{city_to_query}'")
                    
                    # Thực thi hàm Retriever
                    tool_result = get_weather(city_to_query)
                    print(f"  [Step {step+1}] --> Kết quả từ Data Space: {tool_result}")
                    
                    # Nạp kết quả vào để LLM đọc và tổng hợp
                    messages.append({
                        "role": "user", 
                        "content": f"Tool result: {json.dumps(tool_result, ensure_ascii=False)}. Now, answer the user based on this context."
                    })
                    continue # Quay lại đầu vòng lặp
            except json.JSONDecodeError:
                print(f"  [Step {step+1}] --> [Lỗi] LLM sinh ra JSON không hợp lệ: {response_text}")
                break
        else:
            # Nếu LLM không xuất JSON nữa, tức là nó đã có câu trả lời cuối cùng
            return response_text
            
    # Fallback nếu vượt quá số vòng lặp tối đa mà vẫn chưa tổng hợp được
    return "I don't have any data about it."

# ==========================================
# 5. CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("BẮT ĐẦU TEST CASE 1: CÓ THÔNG TIN TRONG DATA SPACE")
    print("="*60)
    ans1 = run_weather_agent("what is the weather in tokyo today?")
    print(f"\n=> FINAL ANSWER: {ans1}\n")

    print("\n" + "="*60)
    print("BẮT ĐẦU TEST CASE 2: KHÔNG CÓ THÔNG TIN")
    print("="*60)
    ans2 = run_weather_agent("what is the weather in new york today?")
    print(f"\n=> FINAL ANSWER: {ans2}\n")
