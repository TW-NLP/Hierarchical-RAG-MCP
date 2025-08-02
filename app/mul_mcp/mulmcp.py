import os
from app.rag.model import SimpleRagQA
from config import SERVICE_INFO,MUL_TEST_DIR,SUMMARY_PATH,FAISS_PATH,SIG_TEST_DIR
from qwen_agent.agents import Assistant
import json
from tqdm import tqdm



class MulMCP(object):
    pass