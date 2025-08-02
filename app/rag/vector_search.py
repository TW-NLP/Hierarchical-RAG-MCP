import numpy as np
import faiss

from config import SEARCH_TOPK


class VectorSearch(object):


    def __init__(self, faiss_path, embedding_model):
        """

        :param faiss_path: 
        :param embedding_model: 
        """

        self.faiss_path = faiss_path
        self.embedding_model = embedding_model

    def simple_vector_search(self, query, data_list):
        """use vector search

        Args:
            query (str):
            data_list (list): data sum list

        Returns:
            list: vector search list
        """
        index = faiss.read_index(self.faiss_path)

        query_embedding = self.embedding_model.encode(query)

        D, I = index.search(np.array(query_embedding), SEARCH_TOPK)

        return [data_list[i] for i in I[0]]
