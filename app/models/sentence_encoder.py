from FlagEmbedding import BGEM3FlagModel
from app.core.config import BGEM3_MODEL_PATH, MAX_SEQ_LENGTH, DEVICE
import logging


class SentenceEncoder:
    def __init__(self):

        self.model = BGEM3FlagModel(BGEM3_MODEL_PATH, use_fp16=True)
        logging.info(f"Loaded BGEM3FlagModel from {BGEM3_MODEL_PATH} on {DEVICE}")

    def encode(self, texts, batch_size=32, max_length=MAX_SEQ_LENGTH):

        result = self.model.encode(texts, batch_size=batch_size, max_length=max_length)
        # 看看输出的都是什么。
        print(f"Encoded {len(texts)} texts into {len(result['dense_vecs'])} vectors")
        print(f"samlped dense_vecs: {result['dense_vecs'][:5]}")
        return result['dense_vecs']


# Singleton instance
sentence_encoder_instance = SentenceEncoder()
