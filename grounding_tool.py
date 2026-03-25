import argparse
import requests
from bs4 import BeautifulSoup
import json
import re
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

class GroundingTool:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        print("Model loaded successfully.\n")

    def _call_llm(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
            
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"[Warning] LLM response invalid JSON. Raw response:\n{response}")
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
                "company_name": company_name,
                "industry": None,
                "address": None,
                "founded_year": None,
                "website": None
            }

            for line in lines[1:]:
                lower_line = line.lower()
                # Parse linh hoạt hơn, chỉ tìm từ khóa thay vì khớp chính xác nguyên cụm
                if "- **industry**:" in lower_line:
                    company_data["industry"] = line.split(":", 1)[1].strip()
                elif "- **address**:" in lower_line:
                    company_data["address"] = line.split(":", 1)[1].strip()
                elif "- **founded_year**:" in lower_line:
                    company_data["founded_year"] = line.split(":", 1)[1].strip()
                elif "- **website**:" in lower_line:
                    company_data["website"] = line.split(":", 1)[1].strip()

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
        print("Web Data Extracted:", json.dumps(web_data, indent=2))
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

        # Bước 2.1: Read & Parse info.md -> JSON[]
        ref_companies_list = self._parse_md_to_json_list(md_content)
        print(f"  -> Parsed {len(ref_companies_list)} companies from reference data.")

        # Bước 2.2: Prepare Context & AI Entity Resolution (Duyệt qua từng JSON)
        system_prompt = "You are an AI entity resolution logic gate. Compare the target name with the reference data. Output ONLY a valid JSON."

        for ref_company in ref_companies_list:
            ref_name = ref_company.get('company_name')

            if target_company_name.strip().lower() == ref_name.strip().lower():
                print(f"  => Rule-based Match Confirm for '{ref_name}'!")
                print("Matched Reference Data:", json.dumps(ref_company, indent=2, ensure_ascii=False))
                return ref_company

            print(f"  -> AI Checking target against: '{ref_name}'...")
            
            # Đổi ví dụ sang Alphabet/Google để không bị nhiễu với FPT
            user_prompt = f"""
            Target Company Name (from Web): "{target_company_name}"
            
            Reference Company Data (from DB):
            {json.dumps(ref_company, ensure_ascii=False, indent=2)}
            
            Task: Does the Target Company Name refer to the EXACT SAME company as the Reference Company Data?
            
            Strict Rule: Parent corporations are DIFFERENT from their subsidiaries (e.g., "Alphabet Inc." is DIFFERENT from "Google LLC". "Vingroup" is DIFFERENT from "VinFast").
            
            Respond ONLY with a JSON object containing a single boolean key "is_match" (true or false).
            
            Output format:
            ```json
            {{
                "is_match": true
            }}
            ```
            """
            
            result = self._call_llm(system_prompt, user_prompt)
            
            # Debug: In ra kết quả raw của AI để dễ kiểm soát
            # print(f"     [Debug AI Output]: {result}")
            
            # Xử lý an toàn kiểu dữ liệu (model có thể trả về boolean True hoặc string "true", "True")
            match_val = result.get("is_match")
            if str(match_val).strip().lower() == "true":
                print(f"  => Match Confirm!")
                print("Matched Reference Data:", json.dumps(ref_company, indent=2, ensure_ascii=False))
                return ref_company

        # Chạy hết vòng for mà AI không match được cái nào
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
    parser = argparse.ArgumentParser(description="Grounding Tool CLI")
    parser.add_argument("url", help="URL of the company website to extract data from")
    parser.add_argument("info_md", help="Path to the info.md file containing reference company data")
    
    args = parser.parse_args()

    # Initialize Tool
    tool = GroundingTool(model_path=r"D:\workspaces\model\Qwen\Qwen2.5-1.5B-Instruct")

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
