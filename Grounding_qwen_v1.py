import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Khởi tạo Model và Tokenizer từ thư mục local (hoặc Hugging Face hub)
model_path = "Qwen/Qwen2.5-3B-Instruct" # Thay bằng đường dẫn folder local của bạn nếu cần
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto" # Tự động map lên GPU nếu có
)

# 2. Chuẩn bị Database (Mock Data) và Hàm Tool
WEATHER_DATA = {
    "tokyo": {"weather": "sunny", "temperature": 22},
    "hanoi": {"weather": "rainy", "temperature": 25},
    "london": {"weather": "cloudy", "temperature": 15}
}

def get_weather(city_name):
    """Hàm giả lập lấy dữ liệu thời tiết"""
    city = city_name.strip().lower()
    if city in WEATHER_DATA:
        # Trả về data thực tế điền vào JSON
        return {"status": "success", "city": city, "data": WEATHER_DATA[city]}
    else:
        # Trả về rỗng hoặc lỗi nếu không có data
        return {"status": "error", "message": "No data found"}

# 3. Khởi tạo System Prompt với các Rules nghiêm ngặt
system_prompt = """You are an AI assistant that can access external tools to answer user questions.
You have ONE tool available:
- Name: get_weather
- Purpose: Get the current weather and temperature for a specific city.
- JSON Call Format: {"action": "get_weather", "city": "<city_name>"}

RULES:
1. If the user asks about the weather, you MUST output ONLY the JSON Call Format to get the data. Do not add any conversational text.
2. The system will provide you with the tool's result.
3. Analyze the result:
   - If the result contains sufficient data (weather and temperature), generate a natural language response (e.g., "Today, in Tokyo, it's sunny, temperature is 22 degree C.").
   - If the result says "error" or "No data found", or you cannot fulfill the request, you MUST reply EXACTLY with: "I don't have any data about it."
"""

# 4. Hàm chạy vòng lặp ReAct (Tối đa N vòng)
def run_weather_agent(user_query, max_iterations=3):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    for i in range(max_iterations):
        # Prepare input format for Qwen
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Ghi nhận phản hồi của LLM vào lịch sử hội thoại
        messages.append({"role": "assistant", "content": response_text})
        
        # 5. Phân tích phản hồi xem có phải là JSON Tool Call không
        if response_text.startswith("{") and "action" in response_text:
            try:
                tool_call = json.loads(response_text)
                if tool_call.get("action") == "get_weather":
                    city_to_query = tool_call.get("city", "")
                    print(f"--> [Agent Action] Calling get_weather for city: {city_to_query}")
                    
                    # Gọi hàm thực tế
                    tool_result = get_weather(city_to_query)
                    print(f"--> [Tool Result] {tool_result}")
                    
                    # Nạp kết quả vào dưới dạng role 'user' (hoặc 'tool' nếu model support native)
                    # Qwen 2.5 hiểu tốt khi ta mớm kết quả hệ thống vào role user
                    messages.append({
                        "role": "user", 
                        "content": f"Tool result: {json.dumps(tool_result)}. Now, answer the user."
                    })
                    continue # Tiếp tục vòng lặp để LLM tổng hợp câu trả lời
            except json.JSONDecodeError:
                print("--> [Error] LLM output invalid JSON.")
                break
        else:
            # Nếu output không phải JSON, tức là LLM đã quyết định trả lời tự nhiên
            return response_text
            
    # Hết vòng lặp mà không có câu trả lời tự nhiên
    return "I don't have any data about it."

# --- CHẠY THỬ NGHIỆM ---
print("--- TEST CASE 1: CÓ DỮ LIỆU ---")
ans1 = run_weather_agent("what is the weather in tokyo today?")
print(f"\nFinal Answer: {ans1}\n")

print("--- TEST CASE 2: KHÔNG CÓ DỮ LIỆU ---")
ans2 = run_weather_agent("what is the weather in new york today?")
print(f"\nFinal Answer: {ans2}\n")
