import os
import faiss
import numpy as np
import json
import logging
import jsonlines
from app.models.sentence_encoder import sentence_encoder_instance
from app.core.config import DEVICE, DEFAULT_TOP_K_RETRIEVAL

class RetrievalService:
    def __init__(self, docs_path, index_path, dimension=768):
        """
        初始化Faiss服务
        
        Args:
            docs_path: 文档数据路径
            index_path: Faiss索引文件保存路径
            dimension: 向量维度，BGE-M3 默认是1024维
        """
        self.docs_path = docs_path
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.doc_ids = []
        self.logger = logging.getLogger(__name__)
        
        # 创建索引目录
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # 加载或创建索引
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """加载现有索引或创建新索引"""
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self._create_index()
    
    def _load_index(self):
        """从文件加载Faiss索引"""
        try:
            self.logger.info(f"Loading Faiss index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            # 加载文档ID映射
            doc_ids_path = self.index_path + ".ids"
            if os.path.exists(doc_ids_path):
                with open(doc_ids_path, 'r', encoding='utf-8') as f:
                    self.doc_ids = json.load(f)
                self.logger.info(f"Loaded {len(self.doc_ids)} document IDs")
            else:
                self.logger.warning("Document IDs file not found, index may be incomplete")
                
            self.logger.info(f"Successfully loaded Faiss index with {self.index.ntotal} vectors")
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            self._create_index()
    
    def _create_index(self):
        """创建新的Faiss索引"""
        self.logger.info("Creating new Faiss index")
        
        # 加载文档数据
        documents = self._load_documents()
        if not documents:
            self.logger.error("No documents found, cannot create index")
            return
        
        # 提取文档文本和ID
        doc_texts = [doc.get("text", "") for doc in documents]
        # documents 的 id / doc_id
        # self.doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        self.doc_ids = [doc.get("doc_id") for doc in documents]

         # 判断 doc_texts 和 doc_ids 数量是否一致
        if len(doc_texts) != len(self.doc_ids):
            raise ValueError(f"doc_texts 和 doc_ids 数量不一致: {len(doc_texts)} vs {len(self.doc_ids)}")
        
        # 编码文档
        self.logger.info(f"Encoding {len(doc_texts)} documents")
        embeddings = self._encode_documents(doc_texts)
        
        # 创建和训练索引
        self.logger.info("Creating Faiss index")
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度索引
        
        # 添加向量到索引
        self.logger.info("Adding vectors to index")
        self.index = self._add_vectors_to_index(embeddings)
        
        # 保存索引
        self._save_index()
    
    def _load_documents(self):
        """加载文档数据"""
        documents = []
        try:
            self.logger.info(f"Loading documents from {self.docs_path}")
            with jsonlines.open(self.docs_path, 'r') as reader:
                for doc in reader:
                    documents.append(doc)
            self.logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
            return []
    
    def _encode_documents(self, texts):
        """使用句子编码器编码文档"""
        try:
            embeddings = sentence_encoder_instance.encode(texts)
            print(f"doc embeddings shape: {embeddings.shape}")
            print(f"doc embeddings samples: {embeddings[:5]}")
            # 确保向量是单位长度的（对于余弦相似度）
            # faiss.normalize_L2(embeddings)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error encoding documents: {str(e)}")
            return np.array([])
    
    def _add_vectors_to_index(self, embeddings):
        """将向量添加到索引中"""
        try:
            self.index.add(embeddings)
            return self.index
        except Exception as e:
            self.logger.error(f"Error adding vectors to index: {str(e)}")
            return None
    
    def _save_index(self):
        """保存索引到文件"""
        try:
            self.logger.info(f"Saving Faiss index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
            
            # 保存文档ID映射
            doc_ids_path = self.index_path + ".ids"
            with open(doc_ids_path, 'w', encoding='utf-8') as f:
                json.dump(self.doc_ids, f)
            
            self.logger.info("Successfully saved Faiss index and document IDs")
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
    
    def search(self, query_text, top_k=DEFAULT_TOP_K_RETRIEVAL):
        """搜索最相似的文档"""
        if not self.index:
            self.logger.error("Index not initialized")
            return []
        
        try:
            # 编码查询
            query_vector = sentence_encoder_instance.encode([query_text])[0]
            print(f"query vector dtype: {query_vector.dtype}")
            print(f"query vector shape: {query_vector.shape}")
            print(f"query vector sample: {query_vector}")
            # 归一化查询向量
            query_vector = query_vector / np.linalg.norm(query_vector)
            query_vector = query_vector.reshape(1, -1)
            
            # 搜索
            scores, indices = self.index.search(query_vector, top_k)
            
            # 整理结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.doc_ids) and idx >= 0:
                    doc_id = self.doc_ids[idx]
                    results.append({
                        "doc_id": doc_id,
                        "score": float(score),
                        "rank": i + 1
                    })
            
            return results
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []