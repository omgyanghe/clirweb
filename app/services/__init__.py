# app/services/__init__.py
from app.core.config import KAZAKH_DOCS_PATH, KAZAKH_DOCS_INDEX_PATH
from app.services.retrieval_service import RetrievalService
from app.services.document_service import DocumentService
import logging

# 初始化服务实例
logger = logging.getLogger(__name__)
logger.info("Initializing document service")
document_service_instance = DocumentService(KAZAKH_DOCS_PATH)

logger.info("Initializing Faiss service")
retrieval_service_instance = RetrievalService(KAZAKH_DOCS_PATH, KAZAKH_DOCS_INDEX_PATH)

