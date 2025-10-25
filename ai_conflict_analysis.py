# ai_conflict_analysis.py
import json
import requests

def analyze_conflict_with_gemini(editors, file_path, operation):
    """
    Analyze a potential conflict using Gemini or other LLMs.
    Returns a structured reasoning response.
    """
    prompt = f"""
    Two or more developers ({', '.join(editors)}) are working on the same branch.
    File affected: {file_path}.
    Operation: {operation}.
    
    As an expert software collaborator assistant, analyze:
    1. The potential risk level (low/medium/high)
    2. Why the conflict might occur
    3. What actions developers should take to resolve or prevent it.
    
    Respond in JSON with keys: risk_level, analysis, recommendations.
    """

    # Example for Gemini API (you can replace later with K2)
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            data=json.dumps(body)
        )
        result = response.json()

        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)  # Expecting valid JSON output
    except Exception as e:
        print(f"[AI ERROR] {e}")
        # fallback
        return {
            "risk_level": "medium",
            "analysis": "AI fallback: possible code overlap detected.",
            "recommendations": "Coordinate with other editors before pushing."
        }
