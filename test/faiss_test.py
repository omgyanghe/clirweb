import numpy as np
import faiss
import pathlib

dimension = 64

# 设置随机种子，保证可复现性
np.random.seed(42)


# 本地文档向量和索引
document_vectors_path = pathlib.Path(__file__).parent / "document_vectors.npy"
index_path = pathlib.Path(__file__).parent / "faiss.index"

if document_vectors_path.exists() and index_path.exists():
    # 如果文件存在，加载文档向量和索引
    print("文档向量和索引文件已存在，加载中...")
    # document_vectors = np.load(str(document_vectors_path))
    # 如果索引文件存在，加载索引
    print("加载现有索引")
    index = faiss.read_index(str(index_path))
    # 重新加载 faiss.index 文件后，不需要再执行 add。
    # 因为索引文件已经包含了所有已添加的向量数据，直接可以用来检索。

    # 如果有GPU可用，可将索引转到GPU
    if faiss.get_num_gpus() > 0:
        print(f"可用GPU数量: {faiss.get_num_gpus()}")
        res = faiss.StandardGpuResources()
        loaded_index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("未检测到可用GPU，使用CPU进行检索")

else:
    # 假设您已经编码好的文档向量和对应的文档ID
    # document_vectors = np.random.random((50000, dimension)).astype('float16')  # 80万个64维向量
    document_vectors = np.load(str(document_vectors_path)).astype('float32')
    # 使用余弦相似度需要先归一化
    faiss.normalize_L2(document_vectors)
    # document_vectors = document_vectors.astype('float16')

    # np.save(str(document_vectors_path), document_vectors)

    # IndexFlatIP 是暴力检索（Brute-force），会对所有向量都计算一次相似度，返回最精确的最近邻结果。
    # IndexIVFFlat 是倒排文件索引（Inverted File），它先将向量聚类（如 nlist=100），
    # 查询时只在部分聚类中心（通常是最相近的几个）下的向量中做精确检索。
    # 这样大大加快了速度，但牺牲了一定的检索精度，结果是近似的，不保证和暴力检索完全一致。

    # 使用 IVF 索引
    nlist = 100  # 聚类中心数量
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    # index.train(document_vectors) 是“倒排索引”类（如 IndexIVFFlat）的必要步骤，用于聚类训练。
    # 只有训练后才能 add 向量并进行检索。
    index.train(document_vectors)

    # index = faiss.IndexFlatIP(dimension)  # 内积暴力检索

    # 检查是否有可用GPU
    if faiss.get_num_gpus() > 0:
        print(f"可用GPU数量: {faiss.get_num_gpus()}")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("未检测到可用GPU，使用CPU进行检索")

    # add 就是“把向量放进索引”，让 Faiss 能检索这些向量。
    index.add(document_vectors)

    # faiss.write_index(faiss.index_gpu_to_cpu(index) if faiss.get_num_gpus() > 0 else index, str(index_path))

# 假设您已经编码好的文档向量和对应的文档ID
document_ids = list(range(50000))  # 文档ID，从0到799999
# 构建映射表
id_to_vector_index = {doc_id: i for i, doc_id in enumerate(document_ids)}
id_to_document_content = {doc_id: f"文档内容 {doc_id}" for doc_id in document_ids}  # 假设的文档内容



print("索引信息：", index)
print("索引中向量数量：", index.ntotal)
# print("第0号向量内容：", index.reconstruct(0))

# # 查看所有索引的向量值（谨慎打印）
# all_vectors = index.reconstruct_n(0, index.ntotal)
# print("前5个向量内容：", all_vectors[:5])

# 执行检索
query_vector = np.random.random((1, dimension)).astype('float16')  # 查询向量
k = 5  # 返回最近邻数量
distances, indices = index.search(query_vector, k)

# 获取检索结果对应的文档ID和内容
retrieved_doc_ids = [document_ids[idx] for idx in indices[0]]
retrieved_doc_contents = [id_to_document_content[doc_id] for doc_id in retrieved_doc_ids]

# 输出结果
print("检索到的文档ID：", retrieved_doc_ids)
print("检索到的文档内容：")
for content in retrieved_doc_contents:
    print(content)
