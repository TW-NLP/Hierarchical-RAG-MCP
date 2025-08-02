import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_DIR, 'data')

SIG_TEST_DIR = os.path.join(DATA_DIR, 'test','sig_mcp_test.json')
MUL_TEST_DIR = os.path.join(DATA_DIR, 'test','mul_data_end_add.json')

SERVICE_INFO=os.path.join(DATA_DIR, 'service_info','info.txt')

FAISS_PATH = os.path.join(DATA_DIR, 'faiss_save','data.index')


SUMMARY_PATH = os.path.join(DATA_DIR, 'service_info','summary2other.json')

class LLMConfig():

    LLM_SET_Qwen3 = {
        'model': 'Qwen3-32B',
        'model_server': 'http://172.20.98.51:38084/v1',
        'api_key': 'EMPTY',
         'generate_cfg': {
        'temperature': 0,
        "max_tokens":2000
    }}

    LLM_SET_QwQ = {
        'model': 'QwQ-32B',
        'model_server': 'http://172.20.98.51:38080/v1',
        'api_key': 'EMPTY',
         'generate_cfg': {
        'temperature': 0,
        "max_tokens":2000
    }}

   

    LLM_SET_qwen3_8b = {
        'model': 'Qwen3_8B',
        'model_server': 'http://172.20.98.51:38082/v1',
        'api_key': 'EMPTY',
         'generate_cfg': {
        'temperature': 0,
        "max_tokens":2000
    }}




class RemoteConfig(object):

    embedding_config={
        'model_name':'bge',
        'model_url':'http://172.20.98.51:8081/v1/embeddings'
    }

    rerank_config={
        'model_name':'rerank_base',
        'model_url':"http://172.20.98.51:8085/rerank"
    }

    

SEARCH_TOPK=20
RESULT_TOPK=20
prompt_zh='请判断所提供的工具是否可以用来解决用户的问题。如果可以，请选择合适的函数进行调用，无需过度思考。如果不可以，请直接回答用户的问题，无需进行过度思考。'
prompt_en="Please determine whether the provided tools can be used to solve the user's problem. If they can, please select the appropriate function to call without overthinking. If they cannot, please directly answer the user's question without overthinking."