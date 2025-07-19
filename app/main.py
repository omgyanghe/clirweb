from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.main import api_router
import os
import logging
from app.core.config import LOG_FILE

# 确保日志配置已经在config.py中设置

app = FastAPI(title="Chinese-Kazakh Cross-Lingual IR System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载API路由
app.include_router(api_router)

# 挂载静态文件
# static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
# try:
#     app.mount("/static_assets", StaticFiles(directory=static_dir), name="static_assets")
#     logging.info(f"静态文件已挂载: {static_dir}")
# except RuntimeError as e:
#     logging.warning(f"静态目录未找到或未配置: {str(e)}")

# @app.get("/")
# async def index():
#     index_path = os.path.join(static_dir, "index.html")
#     return FileResponse(index_path)

@app.get("/health")
async def health():
    return {"status": "healthy"}