import os, time
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models_to_try = [
    'gemini-2.0-flash',
    'gemini-3.5-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite',
    'gemini-2.0-flash-lite',
    'gemini-3.6-flash',
]

for m in models_to_try:
    try:
        r = client.models.generate_content(model=m, contents='say hi in 3 words')
        print(f"  OK {m}: {r.text.strip()}")
    except Exception as e:
        err = str(e)[:120]
        print(f"  FAIL {m}: {type(e).__name__} - {err}")
    time.sleep(2)  # small delay between attempts
