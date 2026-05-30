import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import Pydantic models
try:
    from models import GenerateRequest, GenerateResponse
except ImportError:
    from api.models import GenerateRequest, GenerateResponse

# Load environment variables
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "your-username/voice-forge-notion-mistral-7b")

if not HF_API_TOKEN:
    print("WARNING: HF_API_TOKEN environment variable not set. API requests to HuggingFace might fail.")

app = FastAPI(
    title="BrandVoice API Proxy",
    description="FastAPI proxy server to query HuggingFace Inference API for BrandVoice model generation."
)

# CORS Middleware setup to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    """Health check endpoint returning server status and model ID."""
    return {"status": "ok", "model": HF_REPO_ID}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generates brand-specific copy by proxying the request to the HuggingFace Inference API."""
    if not HF_API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_API_TOKEN is not configured on the server."
        )

    # Build prompt using Mistral chat template structure
    # Format content type nicely (e.g., "blog_intro" -> "blog intro", "how_to" -> "how to")
    formatted_type = request.content_type.replace("_", " ")
    instruction = f"Write a {formatted_type} about {request.topic}"
    input_context = f"Topic: {request.topic}. Audience: {request.audience}. Tone: {request.tone}"
    full_prompt = f"<s>[INST] {instruction}\n{input_context} [/INST]"

    # Endpoint URL for HuggingFace Inference API
    url = f"https://api-inference.huggingface.co/models/{HF_REPO_ID}"
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            # Query Hugging Face Inference API
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            
            # Map specific status codes
            if response.status_code == 503:
                raise HTTPException(
                    status_code=503,
                    detail="Model is warming up, retry in 20s"
                )
            elif response.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit hit, wait 60s"
                )
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"HuggingFace API error: {response.text}"
                )
                
            response_json = response.json()
            
            # Extract generated text from response
            if isinstance(response_json, list) and len(response_json) > 0:
                generated_text = response_json[0].get("generated_text", "").strip()
            elif isinstance(response_json, dict) and "generated_text" in response_json:
                generated_text = response_json.get("generated_text", "").strip()
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected HuggingFace API response format: {response_json}"
                )
                
            return GenerateResponse(
                generated_text=generated_text,
                model_id=HF_REPO_ID,
                content_type=request.content_type
            )
            
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Inference HTTP request failed: {str(exc)}"
            )
        except Exception as exc:
            # Pass other unhandled exceptions back as 500
            if isinstance(exc, HTTPException):
                raise exc
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {str(exc)}"
            )

if __name__ == "__main__":
    import uvicorn
    # Default port to 8000 as requested
    port = int(os.getenv("PORT", 8000))
    print(f"Starting FastAPI server proxy on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
