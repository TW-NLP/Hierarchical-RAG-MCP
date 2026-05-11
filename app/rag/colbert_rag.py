"""High-level ColBERT-RAG wrapper.

Mirrors the public surface of :class:`app.rag.model.SimpleRagQA` but uses
ColBERT-style late-interaction (MaxSim) retrieval over per-token
embeddings instead of pooled bi-encoder cosine similarity. This is the
baseline reported in Table 2 of the Hi-RAG paper as "ColBERT-RAG".
"""

import json
import os
from typing import List, Optional

from app.rag.colbert_search import ColBERTRetriever
from config import DATA_DIR, RESULT_TOPK


class ColBERTRagQA(object):
    """ColBERT-RAG retrieval wrapper.

    Loads the same tool corpus used by Flat-RAG / Hi-RAG Stage 1, builds a
    per-token embedding index, and exposes a ``search`` method that returns
    the top-``RESULT_TOPK`` tool summaries ranked by MaxSim.
    """

    def __init__(
        self,
        data_path: str,
        embedding_name: str = "summary",
        model_name: str = "BAAI/bge-large-en-v1.5",
        cache_path: Optional[str] = None,
        max_query_len: int = 64,
        max_doc_len: int = 256,
    ):
        self.data_path = data_path
        self.embedding_name = embedding_name

        with open(data_path, "r", encoding="utf-8") as fh:
            self.data_dict = json.loads(fh.read())

        self.data_sum: List[str] = []
        for data_i in self.data_dict:
            for item in data_i.get("endpoints", []):
                if embedding_name in item:
                    self.data_sum.append(item[embedding_name])

        if cache_path is None:
            cache_dir = os.path.join(DATA_DIR, "colbert_save")
            os.makedirs(cache_dir, exist_ok=True)
            safe_model = model_name.replace("/", "_")
            cache_path = os.path.join(
                cache_dir, f"{safe_model}.{embedding_name}.pkl"
            )

        self.retriever = ColBERTRetriever(
            model_name=model_name,
            max_query_len=max_query_len,
            max_doc_len=max_doc_len,
            cache_path=cache_path,
        )
        self.retriever.index(self.data_sum)

    def search(self, query: str, top_k: int = RESULT_TOPK) -> List[str]:
        return self.retriever.search(query, top_k=top_k)

    def ask(self, query: str) -> List[str]:
        return self.search(query)
