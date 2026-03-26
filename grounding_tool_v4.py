import argparse
import requests
from bs4 import BeautifulSoup
import json
import re
import sys

class GroundingTool:
    def __init__(self, api_endpoint, api_key=""):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_name = "gpt-oss-120b"
        print(f"Loading {self.model_name} via API endpoint: {self.api_endpoint}...")

    def _call_llm(self, system_prompt, user_prompt):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0, # Đưa về 0.0 để tối đa hóa tính chính xác trong trích xuất
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

            json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
                
            return json.loads(text)
            
        except requests.exceptions.RequestException as e:
            print(f"[Error] API Connection failed: {e}")
            return {}
        except json.JSONDecodeError:
            print(f"[Warning] LLM response invalid JSON. Raw response:\n{text}")
            return {}

    # ==========================================
    # FLOW 1: Extract Company Info from URL
    # ==========================================
    def extract_from_url(self, url):
        print(f"\n[1] Access URL: {url}...")
        html_content = ""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            html_content = response.text
        except Exception as e:
            print(f"Error access ({e}). Fallback to mock_page.html")
            try:
                with open("mock_page.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
            except FileNotFoundError:
                print("File mock_page.html is not found!")
                sys.exit(1)

        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract() 
        
        text = soup.get_text(separator=' ', strip=True)[:10000] # Model 120B có thể đọc text dài hơn

        system_prompt = "You are a precise data extraction AI. Output ONLY valid JSON."
        user_prompt = f"""Extract company information from the text below into a JSON object with exactly these keys: "company_name", "industry", "address", "founded_year", "website".
        If a field is missing, set its value to null.
        
        Text:
        {text}
        """
        
        web_data = self._call_llm(system_prompt, user_prompt)
        print("Web Data Extracted:", json.dumps(web_data, indent=2, ensure_ascii=False))
        return web_data

    # ==========================================
    # FLOW 2: One-Shot Entity Resolution từ info.md
    # ==========================================
    def resolve_entity(self, md_file, target_company_name):
        print(f"\n[2] Resolving entity '{target_company_name}' directly via LLM from {md_file}...")
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except Exception as e:
            print(f"Error reading markdown file: {e}")
            return {}

        system_prompt = "You are an expert AI entity resolution engine. Output ONLY valid JSON."
        
        # Gom toàn bộ logic check mẹ/con, lấy full trường, dọn dẹp ký tự vào chung một prompt
        user_prompt = f"""
        Target Company Name: "{target_company_name}"
        
        Reference Database (Markdown Format):
        {md_content}
        
        Task:
        1. Find the exact matching company in the Reference Database.
        2. STRICT RULE: Parent corporations and subsidiaries are DIFFERENT entities (e.g., "FPT Corporation" is NOT "FPT Software"). Only match if they refer to the exact same entity.
        3. If a match is found, you MUST extract ALL its details from the markdown and return a JSON object with exactly these keys: "company_name", "industry", "address", "founded_year", "website". Do not leave out any field that exists in the markdown.
        4. Clean up the extracted values (remove markdown asterisks, fix any strange or broken encoding characters in the address).
        5. If no exact match is found, return an empty JSON object {{}}.
        """
        
        ref_data = self._call_llm(system_prompt, user_prompt)
        
        if ref_data and ref_data.get("company_name"):
            print("Matched Reference Data:", json.dumps(ref_data, indent=2, ensure_ascii=False))
            return ref_data
        else:
            print("  => No exact match found by AI.")
            return {}

    # ==========================================
    # FLOW 3: Semantic Compare
    # ==========================================
    def semantic_compare(self, web_data, ref_data):
        print("\n[3] Comparing Web Data and Reference Data...")
        system_prompt = "You are a data validation AI. Output ONLY valid JSON."
        
        user_prompt = f"""
        Compare these two JSON objects field by field.
        Web Data: {json.dumps(web_data, ensure_ascii=False)}
        Reference Data: {json.dumps(ref_data, ensure_ascii=False)}
        
        Output a JSON object where each key ("industry", "address", "founded_year", "website") contains:
        "url_value": the exact value from Web Data
        "ref_value": the exact value from Reference Data
        "similarity": a float score from 0.0 to 1.0 based on semantic similarity.
        """
        
        final_comparison = self._call_llm(system_prompt, user_prompt)
        return final_comparison

def main():
    parser = argparse.ArgumentParser(description="Grounding Tool CLI (API 120B Version)")
    parser.add_argument("url", help="URL of the company website to extract data from")
    parser.add_argument("info_md", help="Path to the info.md file containing reference company data")
    parser.add_argument("--api-url", required=True, help="Endpoint URL cho model gpt-oss-120b")
    parser.add_argument("--api-key", default="", help="API Key (nếu có)")
    
    args = parser.parse_args()

    tool = GroundingTool(api_endpoint=args.api_url, api_key=args.api_key)

    web_data = tool.extract_from_url(args.url)
    if not web_data:
        print("Failed to extract web data. Exiting.")
        return

    target_company = web_data.get("company_name")
    if not target_company:
        print("Could not identify company name from URL. Exiting.")
        return
        
    ref_data = tool.resolve_entity(args.info_md, target_company)
    if not ref_data:
        print("No matching company found in reference data. Exiting.")
        return

    final_comparison = tool.semantic_compare(web_data, ref_data)
    
    print("\n===============================")
    print("FINAL COMPARISON RESULT (JSON stdout)")
    print("===============================")
    print(json.dumps(final_comparison, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
