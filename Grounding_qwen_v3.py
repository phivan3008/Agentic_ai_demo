import json
import re
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH MODEL (Qwen2.5-3B-Instruct)
# ==========================================
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"  # Thay bằng thư mục local của bạn

print("--> [System] Đang khởi tạo Model và Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto" # Tự động phân bổ lên GPU
)

# ==========================================
# 2. ĐỊNH NGHĨA TOOL (WEB SCRAPER)
# ==========================================
def scrape_website(url):
    """
    Hàm truy cập URL, tải HTML và trích xuất nội dung văn bản.
    Để tối ưu cho LLM đọc và tìm thông tin công ty (thường nằm ở đầu hoặc cuối trang),
    hàm sẽ giữ lại phần text ở Header và Footer nếu trang web quá dài.
    """
    print(f"--> [Tool] Đang truy cập và cào dữ liệu từ: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        # Thêm timeout để tránh treo chương trình
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Xóa các thẻ không cần thiết
        for script in soup(["script", "style", "noscript", "meta", "link"]):
            script.extract()
            
        # Lấy text và làm sạch khoảng trắng
        text = soup.get_text(separator='\n', strip=True)
        
        # Tối ưu context: Nếu text quá dài, lấy 3000 ký tự đầu và 3000 ký tự cuối
        if len(text) > 6000:
            text = text[:3000] + "\n\n... [NỘI DUNG GIỮA ĐÃ ĐƯỢC LƯỢC BỎ] ...\n\n" + text[-3000:]
            
        print("--> [Tool] Trích xuất thành công nội dung trang web.")
        return {
            "status": "success",
            "url": url,
            "content": text
        }
    except Exception as e:
        print(f"--> [Error] Lỗi khi cào dữ liệu: {e}")
        return {
            "status": "error", 
            "message": f"Cannot scrape the website. Error: {str(e)}"
        }

# ==========================================
# 3. THIẾT LẬP AGENT (SYSTEM PROMPT & REACT LOOP)
# ==========================================
system_prompt = """You are an AI assistant specialized in extracting company profiles from websites.
You have ONE tool available:
- Name: scrape_website
- Purpose: Get the raw text content of a given webpage URL.
- JSON Call Format: {"action": "scrape_website", "url": "<url>"}

RULES:
1. When a user asks you to extract information from a URL, you MUST output ONLY the JSON Call Format to scrape the website. DO NOT add any conversational text or markdown.
2. The system will provide you with the tool's result, containing the website's text content.
3. Analyze the text content and extract the following basic company information:
   - Company Name
   - Address
   - Phone Number
   - Email (if available)
4. Present the extracted information clearly in natural language. If any specific information (like email or phone) is missing from the text, state that it is not available.
5. If the tool returns an error, reply: "I couldn't access the website to retrieve the information."
"""

def extract_json(text):
    """Bóc tách JSON thuần túy đề phòng LLM thêm các block markdown."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def run_company_info_agent(user_query, max_iterations=3):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    for step in range(max_iterations):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Temperature = 0.1 để output ổn định, tránh ảo giác (hallucination)
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,  # Tăng token vì kết quả trả về sẽ dài hơn bài toán thời tiết
            temperature=0.1, 
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        messages.append({"role": "assistant", "content": response_text})
        
        # Kiểm tra xem LLM có gọi Tool hay không
        clean_text = extract_json(response_text)
        if clean_text.startswith("{") and "action" in clean_text:
            try:
                tool_call = json.loads(clean_text)
                if tool_call.get("action") == "scrape_website":
                    target_url = tool_call.get("url", "")
                    print(f"  [Step {step+1}] --> Agent quyết định gọi tool cho URL: '{target_url}'")
                    
                    # Gọi hàm cào dữ liệu
                    tool_result = scrape_website(target_url)
                    
                    # Nạp kết quả text từ website vào cho LLM đọc
                    messages.append({
                        "role": "user", 
                        "content": f"Tool result: {json.dumps(tool_result, ensure_ascii=False)}. Now, extract the company information from this text."
                    })
                    continue # Lặp lại vòng lặp để tổng hợp dữ liệu
            except json.JSONDecodeError:
                print(f"  [Step {step+1}] --> [Lỗi] JSON không hợp lệ: {response_text}")
                break
        else:
            # Output không phải cấu trúc JSON -> Đây là câu trả lời cuối cùng
            return response_text
            
    return "I couldn't find all the necessary information."

# ==========================================
# 4. CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    # Test case 1: Truy xuất một trang web thực tế (Ví dụ: Trang web của FPT hoặc một doanh nghiệp bất kỳ)
    # Lưu ý: Bạn có thể thay đổi URL bên dưới sang một website doanh nghiệp khác để test.
    test_url = "https://fpt.com/"
    user_query = f"Can you extract the company name, address, and phone number from this website: {test_url}?"
    
    print("\n" + "="*60)
    print(f"BẮT ĐẦU TEST CASE: Cào dữ liệu từ {test_url}")
    print("="*60)
    
    final_answer = run_company_info_agent(user_query)
    print("\n=> FINAL ANSWER TỪ LLM:\n")
    print(final_answer)
    print("\n" + "="*60)
