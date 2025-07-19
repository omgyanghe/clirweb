# app/api/routes/search.py
from fastapi import APIRouter, Query, HTTPException
from app.services import retrieval_service_instance, document_service_instance
from app.services.rerank_service import rerank_service_instance
from typing import Optional
import logging
import time

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

@router.get("/search")
async def search_documents(
    query: str,
    use_rerank: bool = Query(False, description="是否使用Cross-Encoder重排序"),
    top_k: Optional[int] = Query(100, description="候选文档数量", ge=10, le=200)
):
    """
    搜索文档API端点
    
    Args:
        query: 搜索查询
        use_rerank: 是否使用Cross-Encoder进行精确重排序
        top_k: 从向量检索中获取的候选文档数量
    
    Returns:
        搜索结果，如果使用重排序则按相关性重新排序
    """
    try:
        start_time = time.time()
        logger.info(f"搜索查询: '{query}', 使用重排序: {use_rerank}, 候选数量: {top_k}")
        
        # 第一阶段：向量检索
        vector_start_time = time.time()
        retrieved_docs = retrieval_service_instance.search(query_text=query, top_k=top_k)
        vector_time = time.time() - vector_start_time
        
        if not retrieved_docs:
            logger.info(f"查询 '{query}' 未找到匹配结果")
            return {
                "results": [], 
                "total": 0, 
                "query": query,
                "reranked": False,
                "timing": {
                    # 这两个参数可以帮助你分析和监控接口的性能瓶颈
                    # 如果vector_search_ms很高，说明向量检索慢
                    # 如果total_ms远大于vector_search_ms，说明后续处理（如重排序、数据组装等）耗时较多
                    "vector_search_ms": round(vector_time * 1000, 2),
                    "total_ms": round((time.time() - start_time) * 1000, 2)
                }
            }
        
        # 获取文档详细内容
        doc_ids = [doc['doc_id'] for doc in retrieved_docs]
        docs_data = document_service_instance.get_documents_by_ids(doc_ids)
        
        # 构建候选文档列表
        candidate_docs = []
        for doc in retrieved_docs:
            doc_id = doc['doc_id']
            doc_content = docs_data.get(doc_id)
            if doc_content:
                candidate_docs.append({
                    "doc_id": doc_id,
                    "title": doc_content.get('title', '无标题'),
                    "text": doc_content.get('text', ''),
                    "text_preview": _create_preview(doc_content.get('text', '')),
                    "vector_score": float(doc['score']),
                    "vector_rank": doc['rank']
                })
        
        # 第二阶段：Cross-Encoder重排序（如果启用）
        rerank_time = 0
        rerank_stats = {}
        ranking_comparison = {}
        
        if use_rerank and candidate_docs:
            rerank_start_time = time.time()
            logger.info(f"使用Cross-Encoder重排序 {len(candidate_docs)} 个文档")
            
            try:
                reranked_docs = rerank_service_instance.rerank(
                    query=query, 
                    documents=candidate_docs
                )
                
                # 获取重排序统计信息
                rerank_stats = rerank_service_instance.get_rerank_stats(reranked_docs)
                ranking_comparison = rerank_service_instance.compare_rankings(reranked_docs)
                
                final_results = reranked_docs
                rerank_time = time.time() - rerank_start_time
                
                logger.info(f"重排序完成，用时 {rerank_time:.2f}s，最高分: {rerank_stats.get('max_score', 0):.4f}")
                
            except Exception as e:
                logger.error(f"重排序失败: {str(e)}")
                final_results = candidate_docs
                use_rerank = False  # 标记重排序失败
        else:
            final_results = candidate_docs
        
        total_time = time.time() - start_time
        
        logger.info(f"搜索完成: 查询='{query}', 结果数={len(final_results)}, 总用时={total_time:.2f}s")
        
        # 构建响应
        response = {
            "results": final_results,
            "total": len(final_results),
            "query": query,
            "reranked": use_rerank,
            "timing": {
                "vector_search_ms": round(vector_time * 1000, 2),
                "rerank_ms": round(rerank_time * 1000, 2) if use_rerank else 0,
                "total_ms": round(total_time * 1000, 2)
            }
        }
        
        # 添加重排序统计信息（如果有）
        if use_rerank and rerank_stats:
            response["rerank_stats"] = rerank_stats
            response["ranking_comparison"] = ranking_comparison
        
        return response
        
    except Exception as e:
        logger.error(f"搜索处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索处理错误: {str(e)}")

@router.post("/rerank/unload")
async def unload_rerank_model():
    """卸载Cross-Encoder模型以释放内存"""
    try:
        rerank_service_instance.unload_model()
        logger.info("Cross-Encoder模型已卸载")
        return {"message": "Cross-Encoder模型已成功卸载"}
    except Exception as e:
        logger.error(f"卸载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")

@router.get("/rerank/status")
async def get_rerank_status():
    """获取重排序服务状态"""
    try:
        model_info = rerank_service_instance.cross_encoder.get_model_info()
        return {
            "model_loaded": model_info["is_loaded"],
            "model_path": model_info["model_path"],
            "device": model_info["device"]
        }
    except Exception as e:
        logger.error(f"获取重排序状态失败: {str(e)}")
        return {
            "model_loaded": False,
            "error": str(e)
        }

def _create_preview(text: str, max_length: int = 200) -> str:
    """创建文本预览"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # 尝试在句号处截断
    truncated = text[:max_length]
    last_period = truncated.rfind('。')
    if last_period > max_length * 0.7:  # 如果句号位置合理
        return truncated[:last_period + 1]
    else:
        return truncated + "..."