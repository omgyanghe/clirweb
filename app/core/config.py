import os
from dotenv import load_dotenv
import torch
import platform
import logging
from pydantic.pydantic_settings import BaseSettings



"""
配置模块，定义了整个系统的配置参数。
该模块包含了模型路径、数据库连接、数据文件路径以及检索参数等配置项。
"""

load_dotenv()


class Settings(BaseSettings):
    DB_USER: str = PSQL_USER
    DB_PASSWORD: str = PSQL_PASSWORD
    DB_HOST: str = PSQL_HOST
    DB_PORT: int = PSQL_PORT
    DB_NAME: str = PSQL_DB_NAME

    class Config:
        env_file = ".env"

settings = Settings()


# General - 通用配置
# 设备配置，优先使用GPU，如果不可用则使用CPU
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
    

# Index Paths - 索引文件路径配置
# 索引文件保存目录
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'data', 'indices')
# 确保索引目录存在
os.makedirs(INDEX_DIR, exist_ok=True)
# 哈萨克语文档索引路径
KAZAKH_DOCS_INDEX_PATH = os.path.join(INDEX_DIR, "kazakh_docs.index")

# 修改文档路径为你的测试文档
KAZAKH_DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'data', 'documents', 'clirmatrix_test_docs.jsonl')


# 模型输入的最大序列长度
# bge-m3 模型的最大输入长度为8192
MAX_SEQ_LENGTH = 8192

# Model Paths - 模型路径配置
# BGEM3 模型路径配置
BGEM3_MODEL_PATH = os.getenv("BGEM3_MODEL_PATH", "/Users/yanghe/code/ckirweb/data/models/BAAI/bge-m3")
BGEM3_MODEL_PATH = os.getenv("BGEM3_MODEL_PATH", "C:\\Users\\yangh\\Desktop\\code\\ckirweb\\data\\models\\BAAI\\bge-m3")

# 双塔编码器模型名称或路径，默认使用BGEM3模型
DUAL_ENCODER_MODEL_NAME_OR_PATH = BGEM3_MODEL_PATH
# 双塔编码器模型权重文件路径（BGEM3不需要.pt权重）
DUAL_ENCODER_PT_PATH = ""
# 交叉编码器模型名称或路径，默认使用多语言BERT模型
CROSS_ENCODER_MODEL_NAME_OR_PATH = os.getenv("CROSS_ENCODER_MODEL_NAME_OR_PATH", "bert-base-multilingual-uncased") # 或你的本地 .pt 文件路径的加载器
# 交叉编码器模型权重文件路径
CROSS_ENCODER_PT_PATH = os.getenv("CROSS_ENCODER_PT_PATH", "data/models_pt/cross_encoder_model.pt")


# Search Params - 检索参数配置
# 默认检索结果数量 100
DEFAULT_TOP_K_RETRIEVAL = 20
# 默认分页大小
DEFAULT_PAGE_SIZE = 10

# Logging - 日志配置
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)