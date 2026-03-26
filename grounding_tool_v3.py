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
        print(f"Loading model via API endpoint: {self.api_endpoint}...")
        print("API Tool initialized successfully.\n")

    def _call_llm(self, system_prompt, user_prompt):
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            # Giả định API trả về JSON format theo chuẩn OpenAI (rất phổ biến cho các OSS API)
            result_json = response.json()
            text = result_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Trích xuất chuỗi JSON từ kết quả text
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
    
    def _parse_md_to_json_list(self, md_content):
        companies = []
        blocks = md_content.split("## ")[1:]
        for block in blocks:
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            if not lines: continue

            name_match = re.match(r'\d+\.\s*(.*)', lines[0])
            company_name = name_match.group(1).strip() if name_match else lines[0].strip()

            company_data = {
                "company_name": company_name.replace('**', ''),
                "industry": None,
                "address": None,
                "founded_year": None,
                "website": None
            }

            # Gom toàn bộ text còn lại để Regex tìm kiếm linh hoạt, tránh lỗi miss data
            block_text = "\n".join(lines[1:])
            
            match_ind = re.search(r'industry\s*[:\-]\s*(.*)', block_text, re.IGNORECASE)
            match_add = re.search(r'address\s*[:\-]\s*(.*)', block_text, re.IGNORECASE)
            match_year = re.search(r'founded_year\s*[:\-]\s*(.*)', block_text, re.IGNORECASE)
            match_web = re.search(r'website\s*[:\-]\s*(.*)', block_text, re.IGNORECASE)

            if match_ind: company_data["industry"] = match_ind.group(1).replace('**', '').strip()
            if match_add: company_data["address"] = match_add.group(1).replace('**', '').strip()
            if match_year: company_data["founded_year"] = match_year.group(1).replace('**', '').strip()
            if match_web: company_data["website"] = match_web.group(1).replace('**', '').strip()

            companies.append(company_data)
        return companies

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
            print(f"Error access ({e}).")
            print("Fallback: Use mock_page.html")
            try:
                with open("mock_page.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
            except FileNotFoundError:
                print("File mock_page.html is not found!")
                sys.exit(1)

        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract() 
        
        text = soup.get_text(separator=' ', strip=True)[:5000]

        user_prompt = f"""You are a data extraction expert. Carefully read the text below and extract the following company information:
        1. Company name (company_name)
        2. Business area (industry)
        3. Address (address)
        4. Year founded (founded_year)
        5. Website (website)

        Extremely important requirement: ONLY return a single valid JSON block.
        Do NOT provide any explanations.
        Do NOT include any text outside the JSON.
        If a piece of information cannot be found, set its value to null.

        Text to be processed:
        {text}
        """
        
        system_prompt = "You are a helpful data extraction assistant that only outputs valid JSON."
        
        web_data = self._call_llm(system_prompt, user_prompt)
        print("Web Data Extracted:", json.dumps(web_data, indent=2, ensure_ascii=False))
        return web_data

    # ==========================================
    # FLOW 2: Select Matching Company from info.md
    # ==========================================
    def resolve_entity(self, md_file, target_company_name):
        print(f"\n[2] Resolving entity '{target_company_name}' from {md_file}...")
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except Exception as e:
            print(f"Error reading markdown file: {e}")
            return {}

        ref_companies_list = self._parse_md_to_json_list(md_content)
        print(f"  -> Parsed {len(ref_companies_list)} companies from reference data.")

        system_prompt = "You are an AI entity resolution logic gate. Compare the target name with the reference data. Output ONLY a valid JSON."

        for ref_company in ref_companies_list:
            ref_name = ref_company.get('company_name')

            if target_company_name.strip().lower() == ref_name.strip().lower():
                print(f"  => Rule-based Match Confirm for '{ref_name}'!")
                print("Matched Reference Data:", json.dumps(ref_company, indent=2, ensure_ascii=False))
                return ref_company

            print(f"  -> AI Checking target against: '{ref_name}'...")
            
            user_prompt = f"""
            Target Company Name (from Web): "{target_company_name}"
            
            Reference Company Data (from DB):
            {json.dumps(ref_company, ensure_ascii=False, indent=2)}
            
            Task: Does the Target Company Name refer to the EXACT SAME company as the Reference Company Data?
            
            Strict Rules: 
            1. Parent corporations and their subsidiaries are DIFFERENT entities (e.g., "Alphabet Inc." vs "Google LLC", or "FPT Corporation" vs "FPT Software"). DO NOT MATCH THEM.
            2. Match only if they are clearly the same legal or business entity.
            
            Respond ONLY with a JSON object containing a single boolean key "is_match" (true or false).
            
            Output format:
            ```json
            {{
                "is_match": true
            }}
            ```
            """
            
            result = self._call_llm(system_prompt, user_prompt)
            
            match_val = result.get("is_match")
            if str(match_val).strip().lower() == "true":
                print(f"  => AI Match Confirm!")
                print("Matched Reference Data:", json.dumps(ref_company, indent=2, ensure_ascii=False))
                return ref_company

        print("  => No exact match found in the entire list.")
        return {}

    # ==========================================
    # FLOW 3: Compare Web Data vs Reference Data
    # ==========================================
    def semantic_compare(self, web_data, ref_data):
        print("\n[3] Comparing Web Data and Reference Data...")
        system_prompt = "You are a strict data validation AI. You must accurately extract values from the provided sources and compare their semantic meaning. Output ONLY valid JSON."
        
        user_prompt = f"""
        Task: Compare two JSON objects field by field.
        
        Web Data (Source 1):
        {json.dumps(web_data, ensure_ascii=False)}
        
        Reference Data (Source 2):
        {json.dumps(ref_data, ensure_ascii=False)}
        
        Rules:
        1. For each field (industry, address, founded_year, website), you MUST copy the EXACT string value from Source 1 into "url_value".
        2. You MUST copy the EXACT string value from Source 2 into "ref_value". Do NOT output generic words like "address" or "website".
        3. Score the semantic similarity (0.0 to 1.0). Example: "Software" and "IT Services" might be 0.8.
        
        Output format MUST be strictly like this example, using the REAL values:
        ```json
        {{
            "industry": {{"url_value": "actual source 1 value", "ref_value": "actual source 2 value", "similarity": 0.9}},
            "address": {{"url_value": "actual source 1 value", "ref_value": "actual source 2 value", "similarity": 0.8}},
            "founded_year": {{"url_value": "actual source 1 value", "ref_value": "actual source 2 value", "similarity": 1.0}},
            "website": {{"url_value": "actual source 1 value", "ref_value": "actual source 2 value", "similarity": 1.0}}
        }}
        ```
        """
        comparison_result = self._call_llm(system_prompt, user_prompt)
        return comparison_result

def main():
    parser = argparse.ArgumentParser(description="Grounding Tool CLI (API Version)")
    parser.add_argument("url", help="URL of the company website to extract data from")
    parser.add_argument("info_md", help="Path to the info.md file containing reference company data")
    
    # Bổ sung 2 tham số mới cho API
    parser.add_argument("--api-url", required=True, help="Endpoint URL cho model gpt-oss-120b")
    parser.add_argument("--api-key", default="", help="API Key (nếu server yêu cầu)")
    
    args = parser.parse_args()

    # Truyền params vào class
    tool = GroundingTool(api_endpoint=args.api_url, api_key=args.api_key)

    # Step 1
    web_data = tool.extract_from_url(args.url)
    if not web_data:
        print("Failed to extract web data. Exiting.")
        return

    # Step 2
    target_company = web_data.get("company_name")
    if not target_company:
        print("Could not identify company name from URL. Exiting.")
        return
        
    ref_data = tool.resolve_entity(args.info_md, target_company)
    if not ref_data:
        print("No matching company found in reference data. Exiting.")
        return

    # Step 3
    final_comparison = tool.semantic_compare(web_data, ref_data)
    
    print("\n===============================")
    print("FINAL COMPARISON RESULT (JSON stdout)")
    print("===============================")
    print(json.dumps(final_comparison, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
