import os
from app.rag.embedding.text_embedding import TextEmbedding
from app.rag.search import RagSearch
from app.rag.write import DataWrite
import json
from config import RESULT_TOPK,FAISS_PATH

class RagQA(object):
    def __init__(self, faiss_path, data_path,embedding_name):
        """
        :param faiss_path:
        :param data_path: 
        """
        self.faiss_path = faiss_path
        self.data_path = data_path
        self.data_dict = json.loads(open(data_path, "r", encoding="utf-8").read())
        self.data_sum = []
        
        for data_i in self.data_dict:
            for item in data_i['endpoints']:
                self.data_sum.append(item[embedding_name])
        
        self.model = TextEmbedding()
        self.write_engine = DataWrite(self.model.embedding_model)
        self.search_engine = RagSearch(faiss_path, self.model.embedding_model, self.model.reranker)
        
        self.bm25_engine = None
        self._initialize_data()
    
    def _initialize_data(self):
        try:
            if os.path.exists(self.faiss_path):
                self.bm25_engine = self.write_engine.vector_write(self.data_sum, self.faiss_path)
            else:
                self.bm25_engine = self.data_save()
        except Exception as e:
            self.bm25_engine = self.data_save()
    
    def data_save(self):
        bm25 = self.write_engine.vector_write(self.data_sum, self.faiss_path)
        return bm25
    
    def search(self, query,w,flat_flag=True):
       
        if flat_flag:
            search_list = self.search_engine.search(query, self.bm25_engine, self.data_sum,w=w)
        else:
            search_list = self.search_engine.search(query, self.bm25_engine, self.data_sum,w=w,flat_flag=False)
        return search_list


class SimpleRagQA:
    
    def __init__(self, faiss_path=None, data_path=None,embedding_name=None):
        if faiss_path is None:
            faiss_path = FAISS_PATH

        self.qa_engine = RagQA(faiss_path, data_path,embedding_name)
    
    def ask(self, query):
        """
        简单的问答接口
        :param query: 
        :return: 
        """
        return self.qa_engine.search(query)