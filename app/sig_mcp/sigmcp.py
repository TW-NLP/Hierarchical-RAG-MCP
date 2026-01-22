import os
from app.rag.model import SimpleRagQA
from config import SIG_TEST_DIR, SERVICE_INFO
from qwen_agent.agents import Assistant
import json
from tqdm import tqdm
from config import SUMMARY_PATH, FAISS_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np


# ==================== 异构图重排序模块 ====================

class QueryGuidedToolAttention(nn.Module):
    """Query引导的工具注意力机制"""
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        self.W_q = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W_t = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * attention_dim, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, query_emb: torch.Tensor, tool_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: [hidden_dim]
            tool_embs: [num_tools, hidden_dim]
        Returns:
            attention_weights: [num_tools]
        """
        query_transformed = self.W_q(query_emb)
        tools_transformed = self.W_t(tool_embs)
        
        query_expanded = query_transformed.unsqueeze(0).expand(tool_embs.size(0), -1)
        concat_features = torch.cat([query_expanded, tools_transformed], dim=1)
        
        e = self.leaky_relu(torch.matmul(concat_features, self.a).squeeze(1))
        alpha = F.softmax(e, dim=0)
        
        return alpha


class HierarchicalServiceAggregator(nn.Module):
    """层级服务聚合器"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, service_emb: torch.Tensor, tool_embs: torch.Tensor, 
                attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            service_emb: [hidden_dim]
            tool_embs: [num_tools, hidden_dim]
            attention_weights: [num_tools]
        Returns:
            aggregated_service_emb: [hidden_dim]
        """
        transformed_tools = self.W_v(tool_embs)
        weighted_tools = attention_weights.unsqueeze(1) * transformed_tools
        aggregated_tools = torch.sum(weighted_tools, dim=0)
        updated_service = self.layer_norm(service_emb + aggregated_tools)
        
        return updated_service


class HeterogeneousGraphReranker(nn.Module):
    """异构图重排序器"""
    def __init__(self, hidden_dim: int = 768, attention_dim: int = 128, mlp_hidden: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.tool_attention = QueryGuidedToolAttention(hidden_dim, attention_dim)
        self.service_aggregator = HierarchicalServiceAggregator(hidden_dim)
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1)
        )
        
    def forward(self, query_emb: torch.Tensor, service_emb: torch.Tensor, 
                tool_embs: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Args:
            query_emb: [hidden_dim]
            service_emb: [hidden_dim]
            tool_embs: [num_tools, hidden_dim]
        Returns:
            score: float
            attention_weights: [num_tools]
        """
        attention_weights = self.tool_attention(query_emb, tool_embs)
        updated_service_emb = self.service_aggregator(service_emb, tool_embs, attention_weights)
        query_service_interaction = updated_service_emb * query_emb
        score = self.scorer(query_service_interaction).item()
        
        return score, attention_weights


class HeterogeneousGraphRerankerWrapper:
    """异构图重排序包装器"""
    def __init__(self, embedding_model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.embedding_model = embedding_model
        
        self.reranker = HeterogeneousGraphReranker(
            hidden_dim=768,
            attention_dim=128,
            mlp_hidden=256
        ).to(device)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本"""
        with torch.no_grad():
            embedding = self.embedding_model.encode(text)
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            return embedding.to(self.device)
    
    def rerank_services(self, query: str, candidate_services: List[Dict]) -> List[Tuple[Dict, float, Dict]]:
        """
        重排序服务
        
        Args:
            query: 查询字符串
            candidate_services: 候选服务列表，每个服务包含:
                - 'service_name': str
                - 'service_description': str
                - 'tools': List[Dict] 包含 'tool_name', 'tool_description'
        Returns:
            排序后的列表: (service_dict, score, attention_info)
        """
        self.reranker.eval()
        query_emb = self.encode_text(query)
        results = []
        
        with torch.no_grad():
            for service in candidate_services:
                service_desc = service.get('service_description', service.get('service_name', ''))
                service_emb = self.encode_text(service_desc)
                
                tools = service.get('tools', [])
                if not tools:
                    simple_score = F.cosine_similarity(
                        query_emb.unsqueeze(0), 
                        service_emb.unsqueeze(0)
                    ).item()
                    results.append((service, simple_score, {'attention_weights': [], 'tool_names': [], 'top_tools': []}))
                    continue
                
                tool_embs = []
                tool_names = []
                for tool in tools:
                    tool_desc = tool.get('tool_description', tool.get('tool_name', ''))
                    tool_emb = self.encode_text(tool_desc)
                    tool_embs.append(tool_emb)
                    tool_names.append(tool.get('tool_name', 'unknown'))
                
                tool_embs = torch.stack(tool_embs)
                score, attention_weights = self.reranker(query_emb, service_emb, tool_embs)
                
                attention_info = {
                    'attention_weights': attention_weights.cpu().numpy().tolist(),
                    'tool_names': tool_names,
                    'top_tools': self._get_top_tools(tool_names, attention_weights, top_k=3)
                }
                
                results.append((service, score, attention_info))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _get_top_tools(self, tool_names: List[str], attention_weights: torch.Tensor, 
                       top_k: int = 3) -> List[Tuple[str, float]]:
        """获取注意力权重最高的top-k工具"""
        weights = attention_weights.cpu().numpy()
        top_indices = np.argsort(weights)[-top_k:][::-1]
        return [(tool_names[i], float(weights[i])) for i in top_indices]


# ==================== SigMCP类 ====================

class SigMCP(object):
    def __init__(self, embedding_model=None):
        self.sig_test_dir = SIG_TEST_DIR
        self.summary2other = json.loads(open(SUMMARY_PATH, encoding="utf-8").read())
        self.simple_qa = SimpleRagQA(faiss_path=FAISS_PATH, data_path=SIG_TEST_DIR, embedding_name='summary')
        
        # 初始化异构图重排序器
        if embedding_model is not None:
            self.graph_reranker = HeterogeneousGraphRerankerWrapper(embedding_model)
        else:
            self.graph_reranker = None

    def init_agent_service(self, tools, llm_set, sys_mes=''):
        """
        初始化agent服务
        :param tools: 工具列表
        :param llm_set: LLM配置
        :param sys_mes: 系统消息
        :return: bot实例
        """
        bot = Assistant(llm=llm_set,
                        function_list=tools,
                        name='',
                        system_message=sys_mes,
                        description="I'm a roboot using the tool calling.")
        return bot

    def test(self, query: str, llm_set):
        """原始test方法"""
        tools = [{'mcpServers': {}}]
        
        for line in open(SERVICE_INFO, encoding="utf-8").readlines()[:23]:
            service_name, service_port = line.strip().split("\t")
            tools[0]['mcpServers'][service_name] = {
                'url': f"http://localhost:{service_port}/sse"
            }
        
        print("tools:", tools)
        bot = self.init_agent_service(tools, llm_set)
        
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response

    def rag_test(self, query, llm_set, w, prompt):
        """原始rag_test方法"""
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w)
        res_str = res_no_hi_des[0]
        
        service_find = self.summary2other[res_str]['service_name']
        port_find = self.summary2other[res_str]['port']
        
        tools = [{
            'mcpServers': {
                service_find: {
                    'url': f"http://localhost:{port_find}/sse"
                }
            }
        }]
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response

    def rag_flat(self, query, llm_set, prompt):
        """原始rag_flat方法"""
        res_no_hi_des = self.simple_qa.qa_engine.search(query)
        res_str = res_no_hi_des[0]
        
        service_find = self.summary2other[res_str]['service_name']
        port_find = self.summary2other[res_str]['port']
        
        tools = [{
            'mcpServers': {
                service_find: {
                    'url': f"http://localhost:{port_find}/sse"
                }
            }
        }]
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response

    def rag_test_top3(self, query, llm_set, w, prompt):
        """原始rag_test_top3方法"""
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w)
        service_find_list = []
        port_find_list = []

        for res_str_i in res_no_hi_des:
            service_find_i = self.summary2other[res_str_i]['service_name']
            port_find_i = self.summary2other[res_str_i]['port']
            if service_find_i not in service_find_list:
                service_find_list.append(service_find_i)
                port_find_list.append(port_find_i)

        tools = [{'mcpServers': {}}]

        for i in range(len(service_find_list[:3])):
            tools[0]['mcpServers'][service_find_list[i]] = {
                'url': f"http://localhost:{port_find_list[i]}/sse"
            }
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response

    def _prepare_candidate_services(self, retrieved_summaries: List[str]) -> List[Dict]:
        """
        将检索到的summaries转换为候选服务的结构化表示
        
        Args:
            retrieved_summaries: 检索到的summary列表
        Returns:
            候选服务列表
        """
        service_dict = {}
        
        for summary in retrieved_summaries:
            if summary not in self.summary2other:
                continue
                
            service_info = self.summary2other[summary]
            service_name = service_info['service_name']
            
            if service_name not in service_dict:
                service_dict[service_name] = {
                    'service_name': service_name,
                    'service_description': service_info.get('title', service_name),
                    'port': service_info['port'],
                    'type': service_info.get('type', ''),
                    'tools': []
                }
            
            service_dict[service_name]['tools'].append({
                'tool_name': summary,
                'tool_description': summary,
                'original_summary': summary
            })
        
        return list(service_dict.values())

    def hi_rag_test(self, query, llm_set, w, prompt):
        """
        使用异构图重排序的版本（Top-1）
        如果没有初始化graph_reranker，则回退到原始方法
        """
        # 初始检索
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w, flat_flag=False)
        res_no_hi_des = res_no_hi_des[:10]
        
        # 如果没有graph_reranker，使用原始方法
        if self.graph_reranker is None:
            res_no_hi_des_dict = {}
            for res_j in res_no_hi_des:
                res_no_hi_des_dict[res_j] = f"This is hierarchical information: type={self.summary2other[res_j]['type']}, service={self.summary2other[res_j]['title']}, tool={res_j}"

            value_list = list(res_no_hi_des_dict.values())
            res_no_hi_des_dict_res = {v: k for k, v in res_no_hi_des_dict.items()}
            
            results = self.simple_qa.qa_engine.search_engine.rerank(query, value_list)
            res_list = list(results.keys())
            end_summary_list = [res_no_hi_des_dict_res[k] for k in res_list]
            res_str = end_summary_list[0]
        else:
            # 使用异构图重排序
            candidate_services = self._prepare_candidate_services(res_no_hi_des)
            if not candidate_services:
                # 如果没有候选服务，回退到简单检索
                res_str = res_no_hi_des[0] if res_no_hi_des else None
                if not res_str:
                    return {}
            else:
                ranked_services = self.graph_reranker.rerank_services(query, candidate_services)
                best_service, best_score, attention_info = ranked_services[0]
                
                print(f"\n[Graph Reranking] Selected service: {best_service['service_name']}, Score: {best_score:.4f}")
                print(f"  Top tools by attention:")
                for tool_name, weight in attention_info['top_tools']:
                    print(f"    - {tool_name}: {weight:.4f}")
                
                # 从最佳服务中选择第一个工具对应的summary
                res_str = best_service['tools'][0]['original_summary']

        service_find = self.summary2other[res_str]['service_name']
        port_find = self.summary2other[res_str]['port']
        
        tools = [{
            'mcpServers': {
                service_find: {
                    'url': f"http://localhost:{port_find}/sse"
                }
            }
        }]
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response

    def filter_service(self, res):
        """过滤服务"""
        service_names = []
        for service_des in res:
            if self.summary2other[service_des]['service_name'] not in service_names:
                service_names.append(self.summary2other[service_des]['service_name'])
        if len(service_names) >= 3:
            return True
        else:
            return False

    def hi_rag_test_top3(self, query, llm_set, w, prompt):
        """
        使用异构图重排序的版本（Top-3）
        如果没有初始化graph_reranker，则回退到原始方法
        """
        # 初始检索
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w, flat_flag=False)
        res_no_hi_des = res_no_hi_des[:10]

        if not self.filter_service(res_no_hi_des):
            res_no_hi_des = res_no_hi_des[:20]

        # 如果没有graph_reranker，使用原始方法
        if self.graph_reranker is None:
            res_no_hi_des_dict = {}
            for res_j in res_no_hi_des:
                res_no_hi_des_dict[res_j] = f"This is hierarchical information: type={self.summary2other[res_j]['type']}, service={self.summary2other[res_j]['title']}, tool={res_j}"

            value_list = list(res_no_hi_des_dict.values())
            res_no_hi_des_dict_res = {v: k for k, v in res_no_hi_des_dict.items()}
            
            results = self.simple_qa.qa_engine.search_engine.rerank(query, value_list)
            res_list = list(results.keys())
            end_summary_list = [res_no_hi_des_dict_res[k] for k in res_list]
            
            filter_service_name = []
            filter_port = []
            
            for service_des in end_summary_list:
                if self.summary2other[service_des]['service_name'] not in filter_service_name:
                    filter_service_name.append(self.summary2other[service_des]['service_name'])
                    filter_port.append(self.summary2other[service_des]['port'])
        else:
            # 使用异构图重排序
            candidate_services = self._prepare_candidate_services(res_no_hi_des)
            
            if not candidate_services:
                # 如果没有候选服务，回退到简单检索
                filter_service_name = []
                filter_port = []
                for res_str in res_no_hi_des[:3]:
                    service_name = self.summary2other[res_str]['service_name']
                    if service_name not in filter_service_name:
                        filter_service_name.append(service_name)
                        filter_port.append(self.summary2other[res_str]['port'])
            else:
                ranked_services = self.graph_reranker.rerank_services(query, candidate_services)
                
                filter_service_name = []
                filter_port = []
                
                for service, score, attention_info in ranked_services[:3]:
                    service_name = service['service_name']
                    port = service['port']
                    
                    if service_name not in filter_service_name:
                        filter_service_name.append(service_name)
                        filter_port.append(port)
                    
                    print(f"\n[Graph Reranking] Service: {service_name}, Score: {score:.4f}")
                    print(f"  Top tools by attention:")
                    for tool_name, weight in attention_info['top_tools']:
                        print(f"    - {tool_name}: {weight:.4f}")

        tools = [{'mcpServers': {}}]
        
        for i in range(len(filter_service_name[:3])):
            tools[0]['mcpServers'][filter_service_name[i]] = {
                'url': f"http://localhost:{filter_port[i]}/sse"
            }
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response

    def signal_infer(self, save_path, llm_set, rag_type=None, topk=1, prompt="请判断所提供的工具是否可以用来解决用户的问题。如果可以，请选择合适的函数进行调用，无需过度思考。如果不可以，请直接回答用户的问题，无需进行过度思考。"):
        """
        信号推理方法
        
        :param save_path: 结果保存路径
        :param llm_set: 大模型配置
        :param rag_type: RAG类型 (None, 'FlatRAG', 'HIRAG')
        :param topk: top-k设置
        :param prompt: 系统提示
        :return:
        """
        label_list = []
        data_list = json.loads(open(SIG_TEST_DIR, encoding="utf-8").read())
        true_label = ''

        for line in tqdm(data_list):
            service_name = line.get('name', '')

            for tool_i in tqdm(line.get('endpoints', [])):
                try:
                    true_label = service_name + tool_i.get('path', '').replace('/', '-') + "_" + tool_i.get('path', '').replace('/', '') + "_" + tool_i.get('method', '').lower()
                    true_label = true_label.replace("-", "_").lower()
                    tool_name_list = []
                    
                    if not rag_type:
                        response = self.test(tool_i.get('query'), llm_set)
                    elif rag_type == 'FlatRAG':
                        if topk == 1:
                            response = self.rag_test(tool_i.get('query'), llm_set, w=0.1, prompt=prompt)
                        else:
                            response = self.rag_test_top3(tool_i.get('query'), llm_set, w=0.1, prompt=prompt)
                    elif rag_type == 'HIRAG':
                        if topk == 1:
                            response = self.hi_rag_test(tool_i.get('query'), llm_set, w=0.1, prompt=prompt)
                        else:
                            response = self.hi_rag_test_top3(tool_i.get('query'), llm_set, w=0.1, prompt=prompt)
                    
                    tool_i['response'] = response
                    tool_i['tool_pred'] = []
                    print("response:", response)

                    # 遍历获取tool name
                    for item in response:
                        if item.get('role', '') == 'function':
                            tool_name = item.get('name', '')
                            if tool_name:
                                tool_name_list.append(tool_name.replace("-", "_").lower())
                    
                    tool_i['tool_pred'] = tool_name_list
                    tool_i['tool_label'] = true_label
                    print("*" * 30)
                    print(tool_i['tool_pred'], tool_i['tool_label'])

                    if tool_name_list and true_label == tool_name_list[0]:
                        label_list.append(1)
                    else:
                        label_list.append(0)
                        
                except Exception as e:
                    print(f"Error processing {service_name} - {tool_i.get('path', '')}: {e}")
                    label_list.append(0)
                    continue
                    
                print(sum(label_list) / len(label_list) if label_list else 0)
        
        # ACC
        accuracy = sum(label_list) / len(label_list) if label_list else 0
        print(f"Accuracy: {accuracy:.3f}")

        with open(save_path, "w", encoding="utf-8") as file_write:
            file_write.write(json.dumps(data_list, ensure_ascii=False, indent=4))
        file_write.close()
