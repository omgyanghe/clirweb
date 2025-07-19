# app/models/cross_encoder.py
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Optional
from app.core.config import CROSS_ENCODER_MODEL_NAME_OR_PATH, DEVICE

class CrossEncoder:
    def __init__(self):
        """
        Cross-Encoder模型类
        用于对查询-文档对进行相关性评分
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self):
        """加载Cross-Encoder模型"""
        if self.is_loaded:
            self.logger.info("Cross-Encoder model already loaded")
            return True
            
        try:
            self.logger.info(f"Loading Cross-Encoder model from {CROSS_ENCODER_MODEL_NAME_OR_PATH}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                CROSS_ENCODER_MODEL_NAME_OR_PATH,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                CROSS_ENCODER_MODEL_NAME_OR_PATH,
                trust_remote_code=True
            )
            
            # 移动到指定设备
            self.model.to(DEVICE)
            self.model.eval()
            
            self.is_loaded = True
            self.logger.info(f"Cross-Encoder model loaded successfully on {DEVICE}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Cross-Encoder model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            return False
    
    def unload_model(self):
        """卸载模型以释放内存"""
        if self.is_loaded:
            self.logger.info("Unloading Cross-Encoder model")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    
    def compute_scores(self, query_doc_pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        """
        计算查询-文档对的相关性分数
        
        Args:
            query_doc_pairs: 查询-文档对列表 [(query, doc), ...]
            batch_size: 批处理大小
            
        Returns:
            相关性分数列表
        """
        if not self.is_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            return [0.0] * len(query_doc_pairs)
        
        if not query_doc_pairs:
            return []
        
        try:
            all_scores = []
            
            # 批处理计算
            for i in range(0, len(query_doc_pairs), batch_size):
                batch_pairs = query_doc_pairs[i:i + batch_size]
                batch_scores = self._compute_batch_scores(batch_pairs)
                all_scores.extend(batch_scores)
            
            self.logger.debug(f"Computed scores for {len(query_doc_pairs)} query-document pairs")
            return all_scores
            
        except Exception as e:
            self.logger.error(f"Error computing cross-encoder scores: {str(e)}")
            return [0.0] * len(query_doc_pairs)
    
    def _compute_batch_scores(self, batch_pairs: List[Tuple[str, str]]) -> List[float]:
        """计算一个批次的分数"""
        try:
            # 分离查询和文档
            queries = [pair[0] for pair in batch_pairs]
            docs = [pair[1] for pair in batch_pairs]
            
            # 编码输入
            inputs = self.tokenizer(
                queries, docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到指定设备
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 处理不同的模型输出格式
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # 如果是二分类模型，取正类概率；如果是回归模型，直接使用输出
                if logits.shape[-1] == 1:
                    # 回归模型
                    scores = torch.sigmoid(logits.squeeze(-1))
                else:
                    # 分类模型，取正类概率
                    scores = torch.softmax(logits, dim=-1)[:, 1]
                
                return scores.cpu().tolist()
                
        except Exception as e:
            self.logger.error(f"Error in batch score computation: {str(e)}")
            return [0.0] * len(batch_pairs)
    
    def predict_single(self, query: str, document: str) -> float:
        """
        对单个查询-文档对进行预测
        
        Args:
            query: 查询文本
            document: 文档文本
            
        Returns:
            相关性分数
        """
        scores = self.compute_scores([(query, document)], batch_size=1)
        return scores[0] if scores else 0.0
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_path": CROSS_ENCODER_MODEL_NAME_OR_PATH,
            "is_loaded": self.is_loaded,
            "device": DEVICE,
            "model_type": "cross-encoder"
        }


# 单例实例 - 延迟加载
cross_encoder_instance = CrossEncoder()