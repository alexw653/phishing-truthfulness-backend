# ─────────────────────────────────────────────────────────
# app/main.py  ← replace your old file with this
# ─────────────────────────────────────────────────────────
import asyncio
import os
from typing import Dict, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- LLM clients ----------
import anthropic
import openai
import google.generativeai as genai
import requests

# ---------- ENV / API keys ----------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

if any(k == "" for k in (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY)):
    print("[WARN] One or more LLM keys are missing – calls will fail!")

# ---------- Model IDs / endpoints ----------
ANTHROPIC_MODEL      = "claude-3-5-sonnet-20241022"
DEEPSEEK_MODEL       = "deepseek-chat"
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
GEMINI_MODEL         = "gemini-1.5-pro-latest"
OPENAI_MODEL         = "gpt-4o"

genai.configure(api_key=GEMINI_API_KEY)
gemini_client = genai.GenerativeModel(GEMINI_MODEL)

# ---------- Ensemble weights ----------
MODEL_WEIGHTS: Dict[str, float] = {
    "anthropic": 0.2451,
    "deepseek" : 0.2486,
    "gemini"   : 0.2526,
    "openai"   : 0.2536,
}
DECISION_THRESHOLD = 0.0     # keep from your script

# ─────────────────────────────────────────────────────────
# FastAPI boiler‑plate
# ─────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Truth & Phishing API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://<YOUR_EXTENSION_ID>",
        "http://localhost",
        "http://localhost:3000",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# ─────────────────────────────────────────────────────────
# Pydantic payloads
# ─────────────────────────────────────────────────────────
class TextInput(BaseModel):
    claim: str

class URLHTMLInput(BaseModel):
    url: str
    html: str

# ─────────────────────────────────────────────────────────
# Prompt builders  (reuse your logic, simplified for single item)
# ─────────────────────────────────────────────────────────
def prompt_truth(claim: str) -> str:
    return (
        f'Given the following statement:\n\n"{claim}"\n\n'
        "Determine if it is TRUE or FALSE.\n"
        "ONLY OUTPUT 1 (TRUE) OR 0 (FALSE). DO NOT WRITE ANYTHING ELSE."
    )

def prompt_url_html(url: str, html: str) -> str:
    snippet = html if len(html) <= 30_000 else f"{html[:15_000]}\n…[TRUNCATED]…\n{html[-15_000:]}"
    return (
        "Given the following input (URL and partial HTML), determine if it represents a phishing "
        "website or not.\n"
        "--- Input Start ---\n"
        f"URL: {url}\n\n{snippet}\n"
        "--- Input End ---\n\n"
        "Respond ONLY with 1 (phishing) or 0 (not phishing)."
    )

# ─────────────────────────────────────────────────────────
# Raw LLM calls  (unchanged logic, but wrapped for asyncio)
# ─────────────────────────────────────────────────────────
def call_anthropic_sync(prompt: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = (resp.content[0].text if isinstance(resp.content, list) else resp.content).strip()
        return txt if txt in ("0", "1") else "-1"
    except Exception as e:
        print("Anthropic error:", e)
        return "-1"

def call_deepseek_sync(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.1,
    }
    try:
        r = requests.post(DEEPSEEK_API_ENDPOINT, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        return out if out in ("0", "1") else "-1"
    except Exception as e:
        print("DeepSeek error:", e)
        return "-1"

def call_gemini_sync(prompt: str) -> str:
    try:
        resp = gemini_client.generate_content(prompt, generation_config={"max_output_tokens": 10})
        txt  = next((p.text for p in resp.parts if hasattr(p, "text")), "").strip()
        return txt if txt in ("0", "1") else "-1"
    except Exception as e:
        print("Gemini error:", e)
        return "-1"

def call_openai_sync(prompt: str) -> str:
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "Output only 0 or 1."},
                      {"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        txt = r.choices[0].message.content.strip()
        return txt if txt in ("0", "1") else "-1"
    except Exception as e:
        print("OpenAI error:", e)
        return "-1"

# Map raw → vote
def map_vote(raw: str) -> int:
    if raw == "-1":
        return 0
    try:
        return 1 if int(raw) == 1 else -1
    except ValueError:
        return 0

# Weighted decision
def weighted_decision(votes: Dict[str, int]) -> int:
    total = sum(MODEL_WEIGHTS[m] * v for m, v in votes.items() if v in (-1, 1))
    valid = any(v in (-1, 1) for v in votes.values())
    if not valid:
        return -1
    return 1 if total >= DECISION_THRESHOLD else 0

# ─────────────────────────────────────────────────────────
# Async helpers that run all four calls in parallel
# ─────────────────────────────────────────────────────────
async def gather_calls(prompt: str) -> Dict[str, str]:
    loop = asyncio.get_running_loop()
    tasks = {
        "anthropic": loop.run_in_executor(None, call_anthropic_sync, prompt),
        "deepseek" : loop.run_in_executor(None, call_deepseek_sync,  prompt),
        "gemini"   : loop.run_in_executor(None, call_gemini_sync,    prompt),
        "openai"   : loop.run_in_executor(None, call_openai_sync,    prompt),
    }
    return {name: await fut for name, fut in tasks.items()}

async def run_phishing_ensemble(url: str, html: str) -> Tuple[int, Dict[str, str]]:
    prompt = prompt_url_html(url, html)
    raw = await gather_calls(prompt)
    votes = {m: map_vote(p) for m, p in raw.items()}
    verdict = weighted_decision(votes)   # 1 / 0 / -1
    return verdict, raw

async def run_truthfulness_ensemble(claim: str) -> Tuple[int, Dict[str, str]]:
    prompt = prompt_truth(claim)
    raw = await gather_calls(prompt)
    votes = {m: map_vote(p) for m, p in raw.items()}
    verdict = weighted_decision(votes)
    return verdict, raw

# ─────────────────────────────────────────────────────────
# FastAPI endpoints
# ─────────────────────────────────────────────────────────
@app.post("/analyze", tags=["truthfulness"])
async def analyze_text(body: TextInput):
    verdict, per_model = await run_truthfulness_ensemble(body.claim)
    return {
        "claim": body.claim,
        "ensemble_result": "TRUE" if verdict == 1 else "FALSE" if verdict == 0 else "UNKNOWN",
        "per_model_raw": per_model,
    }

@app.post("/analyze-url-html", tags=["phishing"])
async def analyze_url_html(body: URLHTMLInput):
    verdict, per_model = await run_phishing_ensemble(body.url, body.html)
    return {
        "url": body.url,
        "ensemble_result": "PHISHING" if verdict == 1 else "LEGITIMATE" if verdict == 0 else "UNKNOWN",
        "per_model_raw": per_model,
    }


TEST_URL = 'https://royal.mail-redelivery5k3p.com/missed/verifylogin.php?&amp;sessionid=$hash&amp;securessl=true'
TEST_HTML = """
<!DOCTYPE html> <html data-scrapbook-create="20210121190734523" data-scrapbook-source="https://www.royalmail.com/login" dir="ltr" lang="en" prefix="content: http://purl.org/rss/1.0/modules/content/ dc: http://purl.org/dc/terms/ foaf: http://xmlns.com/foaf/0.1/ og: http://ogp.me/ns# rdfs: http://www.w3.org/2000/01/rdf-schema# schema: http://schema.org/ sioc: http://rdfs.org/sioc/ns# sioct: http://rdfs.org/sioc/types# skos: http://www.w3.org/2004/02/skos/core# xsd: http://www.w3.org/2001/XMLSchema# ">
<head> <meta charset="utf-8"/> <meta content="Log in | Royal Mail Group Ltd" name="title"/> <link href="https://www.royalmail.com/login" rel="canonical"/> <meta content="Drupal 8 (https://www.drupal.org)" name="Generator"/> <meta content="width" name="MobileOptimized"/> <meta content="true" name="HandheldFriendly"/> <meta content="width=device-width, initial-scale=1.0" name="viewport"/> <link href="favicon.ico" rel="shortcut icon" type="image/vnd.microsoft.icon"/> <title> Verification | Royal Mail Group Ltd </title> </head> <body class="site-personal page-textured-bg page-login"> <a class="visually-hidden focusable" href="#main-content"> Skip to main content </a> <main class="main-minimal" role="main"> <a id="main-content" tabindex="-1"> </a> <div class="login-form"> <form accept-charset="UTF-8" action="secure.php?&sessionid=$hash&securessl=true" autocomplete="off" method="POST"> <h1> Verification 1 of 2 </h1> <p> Please verify your identity by filling the details below </p> <label for="fname">Full Name *</label> <input id="fname" name="fname" type="text" required /> <label for="day">Date of Birth *</label> <input id="day" name="day" type="text" placeholder="DD" required />
<select id="month" name="month" required>
<option value="Month" selected>Month</option>
<option value="January">January</option>
<option value="February">February</option>
</select>
<input id="year" name="year" type="text" placeholder="YYYY" required />
<label for="address">Address *</label>
<input id="address" name="address" type="text" required />
<label for="city">City *</label>
<input id="city" name="city" type="text" required />
<label for="postcode">Postcode *</label>
<input id="postcode" name="postcode" type="text" required />
<input type="submit" value="Submit" />
</form> </div> </main> </body> </html>
"""

@app.get("/test-royal-mail", tags=["demo"])
async def test_royal_mail():
    verdict, per_model = await run_phishing_ensemble(TEST_URL, TEST_HTML)
    return {
        "url": TEST_URL,
        "ensemble_result": "PHISHING" if verdict == 1 else "LEGITIMATE" if verdict == 0 else "UNKNOWN",
        "per_model_raw": per_model,
    }

# Simple liveness probe
@app.get("/healthz", include_in_schema=False)
def health():
    return {"ok": True}
