"""ColBERT-style late-interaction retrieval (ColBERT-RAG baseline).

Implements the MaxSim score introduced by Khattab & Zaharia (2020) and used
as the strong flat-retrieval baseline in the Hi-RAG paper:

    S(q, d) = Σ_{i ∈ |q|}  max_{j ∈ |d|}  < E_q[i] , E_d[j] >

where E_q and E_d are L2-normalised per-token contextual embeddings.
Document embeddings are produced once and cached on disk; query embeddings
are produced at retrieval time.

Unlike the pooled bi-encoder used by Flat-RAG/Hi-RAG Stage 1, this baseline
keeps the full token-level representations and performs late interaction
between every query token and every document token.
"""

import os
import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class ColBERTRetriever(object):
    """Token-level late-interaction retriever (ColBERT MaxSim).

    The retriever wraps a HuggingFace encoder, produces per-token embeddings
    for the corpus once, and scores queries via MaxSim. The implementation is
    intentionally backbone-agnostic: any encoder exposed through
    ``transformers.AutoModel`` works (the paper uses ``bge-large-en``).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        max_query_len: int = 64,
        max_doc_len: int = 256,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_path = cache_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Per-document state, populated by ``index``.
        # ``doc_embeddings`` is a list of [L_d, dim] L2-normalised tensors
        # (one per document, variable length L_d ≤ max_doc_len).
        self.doc_embeddings: List[torch.Tensor] = []
        self.doc_texts: List[str] = []

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(self, texts: List[str], max_len: int) -> List[torch.Tensor]:
        """Encode a list of strings into per-token, L2-normalised embeddings.

        Returns a list of tensors, each of shape ``[L_i, hidden_dim]`` where
        ``L_i`` is the unmasked token count for the i-th input.
        """
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state  # [B, L, D]
        hidden = F.normalize(hidden, p=2, dim=-1)

        mask = batch["attention_mask"].bool()
        result: List[torch.Tensor] = []
        for i in range(hidden.size(0)):
            valid = hidden[i][mask[i]]  # [L_i, D]
            result.append(valid)
        return result

    def _encode_batched(
        self, texts: List[str], max_len: int, batch_size: int
    ) -> List[torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="ColBERT encoding",
            disable=batch_size >= len(texts),
        ):
            chunk = texts[start : start + batch_size]
            embeddings.extend(self._encode(chunk, max_len))
        return embeddings

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, corpus: List[str], batch_size: int = 16) -> None:
        """Build (or load from disk) per-token embeddings for ``corpus``."""
        self.doc_texts = list(corpus)

        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as fh:
                    payload = pickle.load(fh)
                if (
                    payload.get("model_name") == self.model_name
                    and payload.get("doc_texts") == self.doc_texts
                ):
                    self.doc_embeddings = [
                        torch.as_tensor(arr, device=self.device, dtype=torch.float32)
                        for arr in payload["doc_embeddings"]
                    ]
                    return
            except Exception:
                # Stale or corrupted cache — fall through and rebuild.
                pass

        self.doc_embeddings = self._encode_batched(
            corpus, self.max_doc_len, batch_size
        )

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            payload = {
                "model_name": self.model_name,
                "doc_texts": self.doc_texts,
                "doc_embeddings": [t.cpu().numpy() for t in self.doc_embeddings],
            }
            with open(self.cache_path, "wb") as fh:
                pickle.dump(payload, fh)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _maxsim(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """ColBERT MaxSim: sum over query tokens of max similarity to doc tokens."""
        # query_emb: [L_q, D], doc_emb: [L_d, D]
        sim = query_emb @ doc_emb.T  # [L_q, L_d]
        return sim.max(dim=1).values.sum().item()

    @torch.no_grad()
    def score(self, query: str) -> np.ndarray:
        """Return MaxSim scores for ``query`` against every indexed document."""
        if not self.doc_embeddings:
            raise RuntimeError("Call `index(corpus)` before scoring queries.")

        query_emb = self._encode([query], self.max_query_len)[0]  # [L_q, D]
        scores = np.empty(len(self.doc_embeddings), dtype=np.float32)
        for i, doc_emb in enumerate(self.doc_embeddings):
            scores[i] = self._maxsim(query_emb, doc_emb)
        return scores

    def search(self, query: str, top_k: int) -> List[str]:
        """Return the top-``k`` document texts ranked by MaxSim."""
        scores = self.score(query)
        order = np.argsort(scores)[::-1][:top_k]
        return [self.doc_texts[i] for i in order]
