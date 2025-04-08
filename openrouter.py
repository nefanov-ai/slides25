import requests
import json
from datetime import datetime
import os

def get_chat_completion(prompt: str, model: str = "meta-llama/llama-4-scout") -> dict:
    """Get chat completion from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer API KEY",
        "HTTP-Referer": "https://your-site-url.com",
        "X-Title": "Your App Name",
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def save_to_files(json_data: dict, content_md: str):
    """Save JSON and Markdown content to timestamped files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{timestamp}.json"
    md_filename = f"{timestamp}.md"
    
    # Save JSON
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    # Save Markdown
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(content_md)
    
    print(f"Files saved:\n- {os.path.abspath(json_filename)}\n- {os.path.abspath(md_filename)}")

def extract_and_format_content(json_data: dict) -> str:
    """Extract content from JSON response and format as Markdown"""
    try:
        content = json_data['choices'][0]['message']['content']
        # Format as Markdown code block if it looks like code
        if any(keyword in content.lower() for keyword in ['def ', 'class ', 'import ', 'function ']):
            return f"```python\n{content}\n```"
        return content
    except (KeyError, IndexError) as e:
        return f"# Error extracting content\n\nCould not find content in response: `{str(e)}`"

def main():
    prompt = "Write a Python function to calculate Fibonacci numbers."
    
    try:
        # Get API response
        response = get_chat_completion(prompt)
        
        # Extract and format content
        content_md = extract_and_format_content(response)
        
        # Save to files
        save_to_files(response, content_md)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"# API Request Failed\n\n`{str(e)}`"
        save_to_files({"error": str(e)}, error_msg)

if __name__ == "__main__":
    main()
