import requests
from config import RemoteConfig

class RemoteEmbedder:
    def __init__(self):
        self.url = RemoteConfig.embedding_config['model_url']

    def encode(self, texts):
        """embedding

        Args:
            texts (list): tesxts

        Returns:
            list: vector embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        response = requests.post(self.url, json={
            "model": RemoteConfig.embedding_config.get('model_name',''), 
            "input": texts
        })
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

class RemoteReranker:
    def __init__(self):
        self.url = RemoteConfig.rerank_config['model_url']

    def compute_score(self, query, docs):
        """重排

        Args:
            query (str): 
            docs (list): 

        Returns:
            list:score
        """

        response = requests.post(self.url, json={
            "model": RemoteConfig.rerank_config.get("model_name",""),  
            "query": query,
            "documents": docs
        })

        response.raise_for_status()
        results = response.json()["results"]

        return {item_i['document']['text']:item_i['relevance_score'] for item_i in results}
