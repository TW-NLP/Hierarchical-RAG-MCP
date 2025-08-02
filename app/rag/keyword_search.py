import jieba
import numpy as np

from config import SEARCH_TOPK


class KeyWordSearch(object):
    """
    keyword search
    """

    def keyword_search(self, query, bm25, data_list):
        """use bm25 search

        Args:
            query (str): question
            bm25 (object): bm25
            data_list (list): data list

        Returns:
            list: bm25 search list
        """
        tokenized_query = list(jieba.cut(query))  
        bm25_scores = bm25.get_scores(tokenized_query)  
        top_n = np.argsort(bm25_scores)[::-1][:SEARCH_TOPK] 

        return [data_list[i] for i in top_n]
