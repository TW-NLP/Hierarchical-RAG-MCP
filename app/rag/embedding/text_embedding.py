from app.rag.embedding.remote_model import RemoteEmbedder,RemoteReranker

class TextEmbedding(object):
    def __init__(self):
        self.embedding_model = RemoteEmbedder()
        self.reranker = RemoteReranker()
