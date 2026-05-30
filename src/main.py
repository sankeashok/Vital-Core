import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.core.config import settings
from src.api.gateway import router as gateway_router, lifespan

# Configure logging at root level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VitalCore.Main")

# 1. Initialize FastAPI with lifecycle lifespan
app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Health Telemetry Wearable Analytics & LLMOps Platform",
    version="1.0.0",
    lifespan=lifespan
)

# 2. Add high-performance CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount all API Gateway endpoints
app.include_router(gateway_router)

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serves the premium glassmorphic wearable telemetry dashboard on the root route.
    Reads file directly in-memory to guarantee absolute reliability.
    """
    ui_path = os.path.join(settings.BASE_DIR, "src/ui/index.html")
    if not os.path.exists(ui_path):
        # Fallback error card if the UI template is missing
        return HTMLResponse(
            content="""
            <html>
                <body style="background:#0f172a; color:#f8fafc; font-family:sans-serif; text-align:center; padding-top:100px;">
                    <h2>❌ Vital-Core UI Template Not Found</h2>
                    <p>Make sure 'src/ui/index.html' exists in the workspace.</p>
                </body>
            </html>
            """,
            status_code=404
        )
    
    with open(ui_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Vital-Core ASGI server locally on {settings.HOST}:{settings.PORT}...")
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
