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
import time

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

# weights
MODEL_WEIGHTS: Dict[str, float] = {
    "anthropic": 0.2451,
    "deepseek" : 0.2486,
    "gemini"   : 0.2526,
    "openai"   : 0.2536,
}
DECISION_THRESHOLD = 0.0     

# FastAPI boiler‑plate
app = FastAPI(title="Hybrid Truth & Phishing API", version="1.1.0")
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

class URLInput(BaseModel):
    url: str

class TextInput(BaseModel):
    claim: str

class URLHTMLInput(BaseModel):
    url: str
    html: str

# Prompt builders 
def prompt_truth(claim: str) -> str:
    return (
        f'Given the following statement:\n\n"{claim}"\n\n'
        "Determine if it is TRUE or FALSE.\n"
        "ONLY OUTPUT 1 (TRUE) OR 0 (FALSE). DO NOT WRITE ANYTHING ELSE."
    )
def prompt_url_only(url: str) -> str:
    return (
        f"Given this URL:\n\n{url}\n\n"
        "Determine if it is a phishing website or not.\n"
        "Respond ONLY with 1 (phishing) or 0 (not phishing)."
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

# raw LLM calls
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

# Async helpers that run all four calls in parallel
async def gather_calls(prompt: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    loop = asyncio.get_running_loop()
    
    timings = {}
    
    async def timed_call(name, func, prompt):
        start = time.monotonic()
        result = await loop.run_in_executor(None, func, prompt)
        end = time.monotonic()
        timings[name] = end - start
        return result
    
    tasks = {
        "anthropic": timed_call("anthropic", call_anthropic_sync, prompt),
        "deepseek" : timed_call("deepseek",  call_deepseek_sync,  prompt),
        "gemini"   : timed_call("gemini",    call_gemini_sync,    prompt),
        "openai"   : timed_call("openai",    call_openai_sync,    prompt),
    }
    
    overall_start = time.monotonic()
    results = await asyncio.gather(*tasks.values())
    overall_end = time.monotonic()
    
    timings["all_llms_combined"] = overall_end - overall_start
    
    # map model name to result
    model_results = dict(zip(tasks.keys(), results))
    
    return model_results, timings

async def run_phishing_ensemble(url: str, html: str) -> Tuple[int, Dict[str, str], Dict[str, float]]:
    prompt = prompt_url_html(url, html)
    raw, timings = await gather_calls(prompt)
    votes = {m: map_vote(p) for m, p in raw.items()}
    verdict = weighted_decision(votes)
    return verdict, raw, timings

async def run_url_only_ensemble(url: str) -> Tuple[int, Dict[str, str], Dict[str, float]]:
    prompt = prompt_url_only(url)
    raw, timings = await gather_calls(prompt)
    votes = {m: map_vote(p) for m, p in raw.items()}
    verdict = weighted_decision(votes)
    return verdict, raw, timings


async def run_truthfulness_ensemble(claim: str) -> Tuple[int, Dict[str, str], Dict[str, float]]:
    prompt = prompt_truth(claim)
    raw, timings = await gather_calls(prompt)
    votes = {m: map_vote(p) for m, p in raw.items()}
    verdict = weighted_decision(votes)
    return verdict, raw, timings

# FastAPI endpoints
@app.post("/analyze", tags=["truthfulness"])
async def analyze_text(body: TextInput):
    verdict, per_model, timings = await run_truthfulness_ensemble(body.claim)
    print("\n================ API DEBUG (TRUTH) ==================")
    print(f"Input Claim: {body.claim}")
    print("Raw model outputs:")
    for model, raw in per_model.items():
        print(f"  {model}: {raw}")
    print("Timings (seconds):")
    for model, t in timings.items():
        print(f"  {model}: {t:.4f}s")
    print(f"Final Verdict: {'TRUE' if verdict == 1 else 'FALSE' if verdict == 0 else 'UNKNOWN'}")
    print("======================================================\n")
    return {
        "claim": body.claim,
        "ensemble_result": "TRUE" if verdict == 1 else "FALSE" if verdict == 0 else "UNKNOWN",
        "per_model_raw": per_model,
        "timings_seconds": timings, 
    }

@app.post("/analyze-url", tags=["phishing"])
async def analyze_url_only(body: URLInput):
    verdict, per_model, timings = await run_url_only_ensemble(body.url)
    print("\n================ API DEBUG (URL-ONLY) ==================")
    print(f"Input URL: {body.url}")
    print("Raw model outputs:")
    for model, raw in per_model.items():
        print(f"  {model}: {raw}")
    print("Timings (seconds):")
    for model, t in timings.items():
        print(f"  {model}: {t:.4f}s")
    print(f"Final Verdict: {'PHISHING' if verdict == 1 else 'LEGITIMATE' if verdict == 0 else 'UNKNOWN'}")
    print("========================================================\n")
    return {
        "url": body.url,
        "ensemble_result": "PHISHING" if verdict == 1 else "LEGITIMATE" if verdict == 0 else "UNKNOWN",
        "per_model_raw": per_model,
        "timings_seconds": timings, 
    }


@app.post("/analyze-url-html", tags=["phishing"])
async def analyze_url_html(body: URLHTMLInput):
    verdict, per_model, timings = await run_phishing_ensemble(body.url, body.html)
    print("\n================ API DEBUG ==================")
    print(f"Input URL: {body.url}")
    print("Raw model outputs:")
    for model, raw in per_model.items():
        print(f"  {model}: {raw}")
    print("Timings (seconds):")
    for model, t in timings.items():
        print(f"  {model}: {t:.4f}s")
    print(f"Final Ensemble Verdict: {'PHISHING' if verdict == 1 else 'LEGITIMATE' if verdict == 0 else 'UNKNOWN'}")
    print("=============================================\n")
    return {
        "url": body.url,
        "ensemble_result": "PHISHING" if verdict == 1 else "LEGITIMATE" if verdict == 0 else "UNKNOWN",
        "per_model_raw": per_model,
        "timings_seconds": timings,   # timings for each model
    }


# Simple liveness probe
@app.get("/healthz", include_in_schema=False)
def health():
    return {"ok": True}
