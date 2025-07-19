import json
import logging
import jsonlines

class DocumentService:
    def __init__(self, docs_path):
        """
        初始化文档服务
        
        Args:
            docs_path: 文档数据路径
        """
        self.docs_path = docs_path
        self.documents = {}
        self.logger = logging.getLogger(__name__)
        self._load_documents()

    def _load_documents(self):
        """加载文档并构建内存索引"""
        try:
            self.logger.info(f"Loading documents from {self.docs_path}")
            count = 0
            with jsonlines.open(self.docs_path, 'r') as reader:
                for doc in reader:
                    # 可能 id 字段名不一致，检查常见的 id 字段名
                    doc_id = None
                    for id_field in ['id', 'doc_id', 'docid']:
                        if id_field in doc:
                            doc_id = doc[id_field]
                            break
                    
                    # 如果没有ID，则使用行号作为ID
                    if doc_id is None:
                        doc_id = str(count)
                        doc['id'] = doc_id
                    # {doc_id: doc}  # 使用文档ID作为键
                    # doc 还是一个 json 对象，包含原来文档的所有字段 {doc_id, title, text}
                    self.documents[doc_id] = doc
                    count += 1
            self.logger.info(f"Loaded {len(self.documents)} documents into memory")
        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")

    def get_document(self, doc_id):
        """获取单个文档"""
        return self.documents.get(doc_id)
    
    def get_documents_by_ids(self, doc_ids):
        """根据ID列表获取多个文档"""
        return {doc_id: self.documents.get(doc_id) for doc_id in doc_ids if doc_id in self.documents}