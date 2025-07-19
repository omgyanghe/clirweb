from modelscope.hub.snapshot_download import snapshot_download
import pathlib
# 定义要下载的模型名称
MODEL_NAME = "BAAI/bge-m3"

# 定义要忽略的文件或文件夹模式（多个模式用逗号分隔）
IGNORE_PATTERNS = ["imgs/", "onnx/"]

MODELSCOPE_CACHE = str(pathlib.Path(__file__).parent / "data" / "models" / "BAAI" / "bge-m3")
print(MODELSCOPE_CACHE)

model_dir = snapshot_download(
    MODEL_NAME,
    ignore_patterns=IGNORE_PATTERNS,
    local_dir=MODELSCOPE_CACHE
)

print(f"模型下载完成: {model_dir}")