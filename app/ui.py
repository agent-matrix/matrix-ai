from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx, os, json

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

def _self_base_url() -> str:
    # When running inside HF Space Docker, use localhost + PORT
    port = os.getenv("PORT", "7860")
    return f"http://127.0.0.1:{port}"

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@router.get("/chat", response_class=HTMLResponse)
async def chat_get(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "answer": None})

@router.post("/chat", response_class=HTMLResponse)
async def chat_post(request: Request, question: str = Form(...)):
    # Call your /v1/chat (or return a placeholder)
    base_url = _self_base_url()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post("/v1/chat", base_url=base_url, json={"query": question})
            data = r.json()
            answer = data.get("answer", "(no answer)")
    except Exception as e:
        answer = f"Error: {e}"
    return templates.TemplateResponse("chat.html", {"request": request, "answer": answer, "question": question})

@router.get("/dev", response_class=HTMLResponse)
async def dev_get(request: Request):
    # Prefill a realistic plan request used by Matrix-Guardian
    sample = {
        "context": {
            "entity_uid": "matrix-ai",
            "health": {"score": 0.64, "status": "degraded", "last_checked": "2025-09-27T00:00:00Z"},
            "recent_checks": [
                {"check": "http", "result": "fail", "latency_ms": 900, "ts": "2025-09-27T00:00:00Z"}
            ],
        },
        "constraints": {"max_steps": 3, "risk": "low"},
    }
    return templates.TemplateResponse("dev.html", {"request": request, "sample": json.dumps(sample, indent=2)})

@router.post("/dev", response_class=HTMLResponse)
async def dev_post(request: Request, payload: str = Form(...)):
    base_url = _self_base_url()
    try:
        body = json.loads(payload)
    except Exception as e:
        return templates.TemplateResponse("dev.html", {"request": request, "sample": payload, "error": f"Invalid JSON: {e}"})
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post("/v1/plan", base_url=base_url, json=body)
            r.raise_for_status()
            data = r.json()
            pretty = json.dumps(data, indent=2)
            return templates.TemplateResponse("dev.html", {"request": request, "sample": payload, "result": pretty})
    except Exception as e:
        return templates.TemplateResponse("dev.html", {"request": request, "sample": payload, "error": str(e)})
