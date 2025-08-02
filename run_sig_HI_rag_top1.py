import os
from app.sig_mcp.sigmcp import SigMCP
from config import PROJECT_DIR,LLMConfig,prompt_zh,prompt_en

if __name__ == '__main__':
    sig_engine=SigMCP()
    model_name='Qwen3-32B'

    if 'Qwen3' in model_name:

        save_path=os.path.join(PROJECT_DIR,'data','infer','Qwen3-32B','sig_mcp_HIRAG.json')

        sig_engine.signal_infer(save_path,llm_set=LLMConfig.LLM_SET_Qwen3,rag_type='HIRAG',prompt=prompt_zh)


    elif 'QwQ' in model_name:

        save_path=os.path.join(PROJECT_DIR,'data','infer','QwQ','sig_mcp_HIRAG.json')

        sig_engine.signal_infer(save_path,llm_set=LLMConfig.LLM_SET_QwQ,rag_type='HIRAG',prompt=prompt_zh)

    elif 'qwen3_8b' in model_name:

        save_path=os.path.join(PROJECT_DIR,'data','infer','qwen3_8b','sig_mcp_HIRAG.json')

        sig_engine.signal_infer(save_path,llm_set=LLMConfig.LLM_SET_qwen3_8b,rag_type='HIRAG',prompt=prompt_zh)
    
    elif 'deepseek' in model_name:


        save_path=os.path.join(PROJECT_DIR,'data','infer','deepseek','sig_mcp_HIRAG.json')

        sig_engine.signal_infer(save_path,llm_set=LLMConfig.LLM_SET_deepseek,rag_type='HIRAG',prompt=prompt_zh)

    elif 'chatgpt' in model_name:
        save_path=os.path.join(PROJECT_DIR,'data','infer','chatgpt','sig_mcp_HIRAG.json')

        sig_engine.signal_infer(save_path,llm_set=LLMConfig.LLM_SET_chatgpt,rag_type='HIRAG',prompt=prompt_en)
    











