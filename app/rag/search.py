from app.rag.keyword_search import KeyWordSearch


from app.rag.vector_search import VectorSearch
from config import RESULT_TOPK


class RagSearch(object):

    def __init__(self, faiss_path, embedding_model, rank_model):
        """

        :param faiss_path: 
        :param embedding_model: 
        :param rank_model: 
        """
        self.key_search = KeyWordSearch()
        self.vector_search = VectorSearch(faiss_path, embedding_model)
        self.rerank_model = rank_model
    def rrf_fusion(self, list1, list2, k=60, w1=0.1, w2=1.0):
       
        scores = {}

        for rank, item in enumerate(list1, start=1):
            scores[item] = scores.get(item, 0.0) + w1 * (1.0 / (k + rank))

        for rank, item in enumerate(list2, start=1):
            scores[item] = scores.get(item, 0.0) + w2 * (1.0 / (k + rank))

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in fused]

    def search(self, query, bm25, data_list,w=0.1,flat_flag=True):
        """ sum vector、bm25 、rerank search

        Args:
            query (str): question
            bm25 (object): bm25
            data_list (list): data list
        """
        search_sum = []
        vector_search_result = self.vector_search.simple_vector_search(query, data_list)
        keyword_search_result = self.key_search.keyword_search(query, bm25, data_list)

        rrf_result = self.rrf_fusion(keyword_search_result, vector_search_result, k=60,w1=w)

        if flat_flag:
            print('flat RAG search')
            return vector_search_result
        else:

            return rrf_result

    def rerank(self, query, search_sum):
        """rerank vector、bm25 results

        Args:
            query (str): question
            search_sum (list): search list

        Returns:
            list: rerank scores list
        """
        reranked_dict = self.rerank_model.compute_score(query, search_sum)

        return reranked_dict
