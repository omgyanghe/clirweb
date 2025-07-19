# app/services/rerank_service.py
import logging
from typing import List, Dict, Optional
from app.models.cross_encoder import cross_encoder_instance

class RerankService:
    def __init__(self):
        """重排序服务类 - 使用Cross-Encoder对检索结果进行重排序"""
        self.logger = logging.getLogger(__name__)
        self.cross_encoder = cross_encoder_instance
    
    def ensure_model_loaded(self) -> bool:
        """确保Cross-Encoder模型已加载"""
        if not self.cross_encoder.is_loaded:
            self.logger.info("Loading Cross-Encoder model for reranking")
            return self.cross_encoder.load_model()
        return True
    
    def rerank(self, query: str, documents: List[Dict], 
               top_k: Optional[int] = None, 
               batch_size: int = 16) -> List[Dict]:
        """
        使用Cross-Encoder对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表，每个文档应包含 'title' 和 'text' 字段
            top_k: 返回前k个结果，如果为None则返回所有结果
            batch_size: Cross-Encoder批处理大小
            
        Returns:
            重排序后的文档列表，按相关性分数降序排列
        """
        if not documents:
            self.logger.warning("Empty document list provided for reranking")
            return []
        
        # 确保模型已加载
        if not self.ensure_model_loaded():
            self.logger.error("Failed to load Cross-Encoder model, returning original order")
            return documents[:top_k] if top_k else documents
        
        try:
            self.logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")
            
            # 准备查询-文档对
            query_doc_pairs = []
            for doc in documents:
                # 构建文档文本：标题 + 内容
                doc_text = self._prepare_document_text(doc)
                query_doc_pairs.append((query, doc_text))
            
            # 使用Cross-Encoder计算相关性分数
            rerank_scores = self.cross_encoder.compute_scores(
                query_doc_pairs, 
                batch_size=batch_size
            )
            
            # 为文档添加重排序分数和排名信息
            reranked_docs = []
            for i, (doc, score) in enumerate(zip(documents, rerank_scores)):
                # 创建文档副本以避免修改原始数据
                reranked_doc = doc.copy()
                reranked_doc['cross_encoder_score'] = float(score)
                reranked_doc['original_rank'] = doc.get('vector_rank', i + 1)
                reranked_docs.append(reranked_doc)
            
            # 按Cross-Encoder分数降序排序
            reranked_docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            # 更新最终排名
            for i, doc in enumerate(reranked_docs):
                doc['final_rank'] = i + 1
            
            # 返回前k个结果（如果指定）
            final_results = reranked_docs[:top_k] if top_k else reranked_docs
            
            self.logger.info(f"Successfully reranked documents. Top score: {final_results[0]['cross_encoder_score']:.4f}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {str(e)}")
            # 出错时返回原始顺序
            return documents[:top_k] if top_k else documents
    
    def _prepare_document_text(self, doc: Dict, max_length: int = 400) -> str:
        """
        准备用于Cross-Encoder的文档文本
        
        Args:
            doc: 文档字典
            max_length: 最大文本长度
            
        Returns:
            格式化的文档文本
        """
        title = doc.get('title', '').strip()
        text = doc.get('text', '').strip()
        
        # 组合标题和正文
        if title and text:
            combined = f"{title}. {text}"
        elif title:
            combined = title
        elif text:
            combined = text
        else:
            combined = "无内容"
        
        # 截断到指定长度
        if len(combined) > max_length:
            combined = combined[:max_length] + "..."
        
        return combined
    
    def get_rerank_stats(self, documents: List[Dict]) -> Dict:
        """
        获取重排序统计信息
        
        Args:
            documents: 重排序后的文档列表
            
        Returns:
            统计信息字典
        """
        if not documents:
            return {"total": 0}
        
        # 计算分数统计
        scores = [doc.get('cross_encoder_score', 0) for doc in documents]
        
        return {
            "total": len(documents),
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "score_range": max(scores) - min(scores) if scores else 0
        }
    
    def compare_rankings(self, documents: List[Dict]) -> Dict:
        """
        比较向量检索和Cross-Encoder重排序的结果差异
        
        Args:
            documents: 重排序后的文档列表
            
        Returns:
            排名比较统计
        """
        if not documents:
            return {}
        
        rank_changes = []
        for doc in documents:
            original_rank = doc.get('original_rank', 0)
            final_rank = doc.get('final_rank', 0)
            if original_rank > 0 and final_rank > 0:
                rank_changes.append(original_rank - final_rank)
        
        if not rank_changes:
            return {"no_ranking_data": True}
        
        return {
            "total_docs": len(documents),
            "avg_rank_change": sum(rank_changes) / len(rank_changes),
            "max_rank_improvement": max(rank_changes) if rank_changes else 0,
            "max_rank_decline": min(rank_changes) if rank_changes else 0,
            "docs_improved": len([x for x in rank_changes if x > 0]),
            "docs_declined": len([x for x in rank_changes if x < 0]),
            "docs_unchanged": len([x for x in rank_changes if x == 0])
        }
    
    def unload_model(self):
        """卸载Cross-Encoder模型以释放内存"""
        self.logger.info("Unloading Cross-Encoder model")
        self.cross_encoder.unload_model()


# 单例实例
rerank_service_instance = RerankService()