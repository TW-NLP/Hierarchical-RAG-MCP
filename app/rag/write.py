import os
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import jieba



class DataWrite(object):
    """
    """

    def __init__(self, embedding_model):
        """

        :param embedding_model: 
        """
        self.embedding_model = embedding_model

    def vector_write(self, data_list, faiss_path):

        embeddings = []
        for data_i in tqdm(data_list):
            embedding_i = self.embedding_model.encode(data_i)
            embeddings.append(embedding_i[0])

        embeddings = np.array(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        faiss.write_index(index, faiss_path)

        # bm25
        tokenized_corpus = [list(jieba.cut(text)) for text in data_list]  
        bm25 = BM25Okapi(tokenized_corpus)  
        return bm25

    
