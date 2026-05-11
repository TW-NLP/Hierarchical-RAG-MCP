import os
from app.rag.model import SimpleRagQA
from app.rag.colbert_rag import ColBERTRagQA
from config import MUL_TEST_DIR, SERVICE_INFO
from qwen_agent.agents import Assistant
import json
from tqdm import tqdm
from config import SUMMARY_PATH, FAISS_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np


# ==================== Hi-RAG Stage 2: Type-Aware Hierarchical Re-ranking ====================

class DomainLevelGating(nn.Module):
    """域级门控机制 (Eq. 2 in paper)"""
    def __init__(self, hidden_dim: int, gate_hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # x_gate = [h_q || h_τ || (h_q ⊙ h_τ)]
        self.mlp_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query_emb: torch.Tensor, type_emb: torch.Tensor) -> torch.Tensor:
        """
        计算域级门控系数 β_s
        
        Args:
            query_emb: [hidden_dim] - query embedding h_q
            type_emb: [hidden_dim] - type embedding h_τ
        Returns:
            beta: scalar in [0, 1] - domain gate value
        """
        hadamard_product = query_emb * type_emb
        x_gate = torch.cat([query_emb, type_emb, hadamard_product], dim=0)
        beta = self.mlp_gate(x_gate)
        
        return beta.squeeze()


class TypeAugmentedToolAttention(nn.Module):
    """类型增强的工具注意力机制 (Eq. 3 in paper)"""
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # 三个变换矩阵: W_q, W_t, W_τ
        self.W_q = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W_t = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W_tau = nn.Linear(hidden_dim, attention_dim, bias=False)
        
        # 注意力向量 a
        self.a = nn.Parameter(torch.randn(3 * attention_dim, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, query_emb: torch.Tensor, tool_embs: torch.Tensor, 
                type_emb: torch.Tensor) -> torch.Tensor:
        """
        计算类型感知的工具注意力权重
        
        Args:
            query_emb: [hidden_dim] - h_q
            tool_embs: [num_tools, hidden_dim] - {h_t_i}
            type_emb: [hidden_dim] - h_τ
        Returns:
            alpha: [num_tools] - attention weights
        """
        # 变换
        query_transformed = self.W_q(query_emb)  # [attention_dim]
        tools_transformed = self.W_t(tool_embs)  # [num_tools, attention_dim]
        type_transformed = self.W_tau(type_emb)  # [attention_dim]
        
        # 扩展并拼接: [W_q h_q || W_t h_t_i || W_τ h_τ]
        query_expanded = query_transformed.unsqueeze(0).expand(tool_embs.size(0), -1)
        type_expanded = type_transformed.unsqueeze(0).expand(tool_embs.size(0), -1)
        concat_features = torch.cat([query_expanded, tools_transformed, type_expanded], dim=1)
        
        # e_i = LeakyReLU(a^T [W_q h_q || W_t h_t_i || W_τ h_τ])
        e = self.leaky_relu(torch.matmul(concat_features, self.a).squeeze(1))
        
        # α_i = softmax(e_i)
        alpha = F.softmax(e, dim=0)
        
        return alpha


class HierarchicalAggregator(nn.Module):
    """层级聚合器 (Eq. 4 in paper)"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 工具值变换矩阵 W_v
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 可学习的缩放因子 γ
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, service_emb: torch.Tensor, tool_embs: torch.Tensor, 
                attention_weights: torch.Tensor, type_emb: torch.Tensor) -> torch.Tensor:
        """
        执行层级聚合生成精化的服务表示
        
        Args:
            service_emb: [hidden_dim] - h_s
            tool_embs: [num_tools, hidden_dim] - {h_t_i}
            attention_weights: [num_tools] - {α_i}
            type_emb: [hidden_dim] - h_τ
        Returns:
            h_s': [hidden_dim] - refined service embedding
        """
        # h_{s,tools} = Σ α_i · W_v h_{t_i}
        transformed_tools = self.W_v(tool_embs)  # [num_tools, hidden_dim]
        weighted_tools = attention_weights.unsqueeze(1) * transformed_tools
        h_s_tools = torch.sum(weighted_tools, dim=0)
        
        # h_s' = LN(h_s + h_{s,tools} + γ(h_s ⊙ h_τ))
        type_interaction = self.gamma * (service_emb * type_emb)
        h_s_prime = self.layer_norm(service_emb + h_s_tools + type_interaction)
        
        return h_s_prime


class TypeAwareHierarchicalReranker(nn.Module):
    """类型感知的层级重排序器 - Hi-RAG Stage 2完整实现"""
    def __init__(self, hidden_dim: int = 768, attention_dim: int = 128, 
                 gate_hidden_dim: int = 256, mlp_hidden: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 三个核心组件
        self.domain_gate = DomainLevelGating(hidden_dim, gate_hidden_dim)
        self.tool_attention = TypeAugmentedToolAttention(hidden_dim, attention_dim)
        self.hierarchical_aggregator = HierarchicalAggregator(hidden_dim)
        
        # 最终评分MLP (Eq. 5)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1)
        )
        
    def forward(self, query_emb: torch.Tensor, service_emb: torch.Tensor, 
                tool_embs: torch.Tensor, type_emb: torch.Tensor) -> Tuple[float, torch.Tensor, float]:
        """
        执行完整的类型感知层级重排序
        
        Args:
            query_emb: [hidden_dim] - h_q
            service_emb: [hidden_dim] - h_s
            tool_embs: [num_tools, hidden_dim] - {h_t_i}
            type_emb: [hidden_dim] - h_τ
        Returns:
            score: float - final ranking score
            attention_weights: [num_tools] - tool attention weights
            beta: float - domain gate value
        """
        # Step 1: 计算域级门控 β_s (Eq. 2)
        beta = self.domain_gate(query_emb, type_emb)
        
        # Step 2: 计算类型增强的工具注意力 {α_i} (Eq. 3)
        attention_weights = self.tool_attention(query_emb, tool_embs, type_emb)
        
        # Step 3: 层级聚合得到精化的服务表示 h_s' (Eq. 4)
        refined_service_emb = self.hierarchical_aggregator(
            service_emb, tool_embs, attention_weights, type_emb
        )
        
        # Step 4: 最终评分 Score(q, s) = β_s · MLP(h_s' ⊙ h_q) (Eq. 5)
        query_service_interaction = refined_service_emb * query_emb
        mlp_score = self.scorer(query_service_interaction).squeeze()
        final_score = beta * mlp_score
        
        return final_score.item(), attention_weights, beta.item()


class HiRAGRerankerWrapper:
    """Hi-RAG重排序包装器 - 实现Algorithm 1"""
    def __init__(self, embedding_model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.embedding_model = embedding_model
        
        # 初始化类型感知层级重排序器
        self.reranker = TypeAwareHierarchicalReranker(
            hidden_dim=768,
            attention_dim=128,
            gate_hidden_dim=256,
            mlp_hidden=256
        ).to(device)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本为嵌入向量"""
        with torch.no_grad():
            embedding = self.embedding_model.encode(text)
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            return embedding.to(self.device)
    
    def rerank_services(self, query: str, candidate_services: List[Dict]) -> List[Tuple[Dict, float, Dict]]:
        """
        执行Hi-RAG Stage 2的类型感知层级重排序
        
        Args:
            query: 用户查询
            candidate_services: 候选服务列表，每个服务包含:
                - 'service_name': str
                - 'service_description': str
                - 'type': str (服务类型/Category)
                - 'tools': List[Dict] 包含工具信息
        Returns:
            排序后的列表: (service_dict, score, reranking_info)
        """
        self.reranker.eval()
        
        # 编码查询 h_q
        query_emb = self.encode_text(query)
        results = []
        
        with torch.no_grad():
            for service in candidate_services:
                # 获取服务类型 τ_s
                service_type = service.get('type', 'unknown')
                type_emb = self.encode_text(service_type)
                
                # 编码服务描述 h_s
                service_desc = service.get('service_description', service.get('service_name', ''))
                service_emb = self.encode_text(service_desc)
                
                # 获取工具列表
                tools = service.get('tools', [])
                
                if not tools:
                    # 如果没有工具，使用简化的评分
                    simple_score = F.cosine_similarity(
                        query_emb.unsqueeze(0), 
                        service_emb.unsqueeze(0)
                    ).item()
                    reranking_info = {
                        'beta': 1.0,
                        'attention_weights': [],
                        'tool_names': [],
                        'top_tools': [],
                        'type': service_type
                    }
                    results.append((service, simple_score, reranking_info))
                    continue
                
                # 编码所有工具 {h_t_i}
                tool_embs = []
                tool_names = []
                for tool in tools:
                    tool_desc = tool.get('tool_description', tool.get('tool_name', ''))
                    tool_emb = self.encode_text(tool_desc)
                    tool_embs.append(tool_emb)
                    tool_names.append(tool.get('tool_name', 'unknown'))
                
                tool_embs = torch.stack(tool_embs)
                
                # 执行类型感知的层级重排序
                score, attention_weights, beta = self.reranker(
                    query_emb, service_emb, tool_embs, type_emb
                )
                
                # 收集重排序信息
                reranking_info = {
                    'beta': beta,  # 域级门控值
                    'attention_weights': attention_weights.cpu().numpy().tolist(),
                    'tool_names': tool_names,
                    'top_tools': self._get_top_tools(tool_names, attention_weights, top_k=3),
                    'type': service_type
                }
                
                results.append((service, score, reranking_info))
        
        # 按分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _get_top_tools(self, tool_names: List[str], attention_weights: torch.Tensor, 
                       top_k: int = 3) -> List[Tuple[str, float]]:
        """获取注意力权重最高的top-k工具"""
        weights = attention_weights.cpu().numpy()
        top_indices = np.argsort(weights)[-top_k:][::-1]
        return [(tool_names[i], float(weights[i])) for i in top_indices]


# ==================== MulMCP类 ====================

class MulMCP(object):
    def __init__(self, embedding_model=None, colbert_model_name: str = "BAAI/bge-large-en-v1.5"):
        self.mul_test_dir = MUL_TEST_DIR
        self.summary2other = json.loads(open(SUMMARY_PATH, encoding="utf-8").read())

        self.simple_qa = SimpleRagQA(
            faiss_path=FAISS_PATH,
            data_path=MUL_TEST_DIR,
            embedding_name='summary'
        )

        # ColBERT-RAG baseline (lazy-loaded on first use).
        self._colbert_model_name = colbert_model_name
        self._colbert_qa = None

        # 初始化Hi-RAG重排序器
        if embedding_model is not None:
            self.hi_rag_reranker = HiRAGRerankerWrapper(embedding_model)
        else:
            self.hi_rag_reranker = None

    @property
    def colbert_qa(self) -> ColBERTRagQA:
        """Lazily instantiate the ColBERT-RAG retriever (heavy local model)."""
        if self._colbert_qa is None:
            self._colbert_qa = ColBERTRagQA(
                data_path=self.mul_test_dir,
                embedding_name='summary',
                model_name=self._colbert_model_name,
            )
        return self._colbert_qa

    def init_agent_service(self, tools, llm_set, sys_mes=''):
        """
        初始化Agent服务
        :param tools: 工具列表
        :param llm_set: LLM配置
        :param sys_mes: 系统消息
        :return: bot实例
        """
        bot = Assistant(
            llm=llm_set,
            function_list=tools,
            name='',
            system_message=sys_mes,
            description="I'm a robot using the tool calling."
        )
        return bot

    def test(self, query: str, llm_set):
        """
        使用所有可用服务进行测试（基线方法）
        :param query: 用户查询
        :param llm_set: LLM配置
        :return: 最终响应
        """
        # Define the agent
        tools = [{
            'mcpServers': {}
        }]
        
        # 加载前23个服务
        for line in open(SERVICE_INFO, encoding="utf-8").readlines()[:23]:
            service_name, service_port = line.strip().split("\t")
            tools[0]['mcpServers'][service_name] = {
                'url': f"http://localhost:{service_port}/sse"
            }
        
        print("tools:", tools)
        bot = self.init_agent_service(tools, llm_set)

        # Chat
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        # 处理最后一条完整回复
        final_response = responses[-1] if responses else {}
        return final_response

    def rag_test(self, query, llm_set, w, prompt):
        """
        使用RAG检索单个最相关的服务
        :param query: 用户查询
        :param llm_set: LLM配置
        :param w: 搜索权重
        :param prompt: 系统提示
        :return: 最终响应
        """
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w)
        res_str = res_no_hi_des[0]
        
        service_find = self.summary2other[res_str]['service_name']
        port_find = self.summary2other[res_str]['port']
        
        # Define the agent
        tools = [{
            'mcpServers': {
                service_find: {
                    'url': f"http://localhost:{port_find}/sse"
                }
            }
        }]
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        # Chat
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        # 处理最后一条完整回复
        final_response = responses[-1] if responses else {}
        return final_response

    def colbert_rag_test(self, query, llm_set, prompt):
        """ColBERT-RAG Top-1: late-interaction (MaxSim) retrieval, no hierarchy."""
        retrieved = self.colbert_qa.search(query, top_k=1)
        if not retrieved:
            return {}

        res_str = retrieved[0]
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

    def colbert_rag_test_top3(self, query, llm_set, prompt):
        """ColBERT-RAG Top-3: flat MaxSim retrieval, de-duplicated to ≤3 services."""
        retrieved = self.colbert_qa.search(query, top_k=20)

        filter_service_name: List[str] = []
        filter_port: List[int] = []
        for res_str in retrieved:
            if res_str not in self.summary2other:
                continue
            service_name = self.summary2other[res_str]['service_name']
            if service_name in filter_service_name:
                continue
            filter_service_name.append(service_name)
            filter_port.append(self.summary2other[res_str]['port'])
            if len(filter_service_name) >= 3:
                break

        tools = [{'mcpServers': {}}]
        for name, port in zip(filter_service_name, filter_port):
            tools[0]['mcpServers'][name] = {
                'url': f"http://localhost:{port}/sse"
            }

        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)

        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)

        final_response = responses[-1] if responses else {}
        return final_response

    def rag_test_top3(self, query, llm_set, w, prompt):
        """
        使用RAG检索Top3相关服务
        :param query: 用户查询
        :param llm_set: LLM配置
        :param w: 搜索权重
        :param prompt: 系统提示
        :return: 最终响应
        """
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w)
        service_find_list = []
        port_find_list = []

        for res_str_i in res_no_hi_des:
            service_find_i = self.summary2other[res_str_i]['service_name']
            port_find_i = self.summary2other[res_str_i]['port']
            if service_find_i not in service_find_list:
                service_find_list.append(service_find_i)
                port_find_list.append(port_find_i)

        # Define the agent
        tools = [{
            'mcpServers': {}
        }]

        for i in range(len(service_find_list[:3])):
            tools[0]['mcpServers'][service_find_list[i]] = {
                'url': f"http://localhost:{port_find_list[i]}/sse"
            }
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        # Chat
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        # 处理最后一条完整回复
        final_response = responses[-1] if responses else {}
        return final_response

    def filter_service(self, res):
        """
        过滤服务，确保至少有3个不同的服务
        :param res: 检索结果列表
        :return: 是否满足条件
        """
        service_names = []
        for service_des in res:
            if self.summary2other[service_des]['service_name'] not in service_names:
                service_names.append(self.summary2other[service_des]['service_name'])
        if len(service_names) >= 3:
            return True
        else:
            return False

    def _prepare_candidate_services(self, retrieved_summaries: List[str]) -> List[Dict]:
        """
        将检索到的summaries转换为候选服务的结构化表示
        实现Tool-as-Proxy策略
        
        Args:
            retrieved_summaries: 检索到的summary列表（工具描述）
        Returns:
            候选服务列表 S_cand
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
                    'type': service_info.get('type', ''),  # Category/Type τ_s
                    'tools': []
                }
            
            # 添加工具信息
            service_dict[service_name]['tools'].append({
                'tool_name': summary,
                'tool_description': summary,
                'original_summary': summary
            })
        
        return list(service_dict.values())

    def hi_rag_test(self, query, llm_set, w, prompt):
        """
        Hi-RAG方法（Top-1）- 实现Algorithm 1
        
        Stage 1: Candidate Service Retrieval (使用混合检索)
        Stage 2: Type-Aware Hierarchical Re-ranking
        """
        # ============ Stage 1: Candidate Service Retrieval ============
        # 使用混合检索获取候选工具，然后提取其父服务
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w, flat_flag=False)
        res_no_hi_des = res_no_hi_des[:10]  # Top-M候选
        
        # 如果没有Hi-RAG重排序器，使用原始方法
        if self.hi_rag_reranker is None:
            res_no_hi_des_dict = {}
            for res_j in res_no_hi_des:
                res_no_hi_des_dict[res_j] = f"This is hierarchical information: type={self.summary2other[res_j]['type']}, service={self.summary2other[res_j]['title']}, tool={res_j}"

            value_list = list(res_no_hi_des_dict.values())
            res_no_hi_des_dict_res = {v: k for k, v in res_no_hi_des_dict.items()}
            
            # 重排序
            results = self.simple_qa.qa_engine.search_engine.rerank(query, value_list)
            res_list = list(results.keys())
            end_summary_list = [res_no_hi_des_dict_res[k] for k in res_list]
            
            # 选择top 1
            res_str = end_summary_list[0]
        else:
            # ============ Stage 2: Type-Aware Hierarchical Re-ranking ============
            # 提取候选服务 S_cand = {service(t) | t ∈ Top-M}
            candidate_services = self._prepare_candidate_services(res_no_hi_des)
            
            if not candidate_services:
                res_str = res_no_hi_des[0] if res_no_hi_des else None
                if not res_str:
                    return {}
            else:
                # 执行类型感知的层级重排序
                ranked_services = self.hi_rag_reranker.rerank_services(query, candidate_services)
                best_service, best_score, reranking_info = ranked_services[0]
                
                print(f"\n[Hi-RAG Stage 2] Selected service: {best_service['service_name']}")
                print(f"  Type: {reranking_info['type']}")
                print(f"  Domain Gate (β): {reranking_info['beta']:.4f}")
                print(f"  Final Score: {best_score:.4f}")
                print(f"  Top tools by attention (α):")
                for tool_name, weight in reranking_info['top_tools']:
                    print(f"    - {tool_name}: {weight:.4f}")
                
                # 选择第一个工具对应的summary
                res_str = best_service['tools'][0]['original_summary']

        # ============ Final Inference ============
        service_find = self.summary2other[res_str]['service_name']
        port_find = self.summary2other[res_str]['port']
        
        # Define the agent
        tools = [{
            'mcpServers': {
                service_find: {
                    'url': f"http://localhost:{port_find}/sse"
                }
            }
        }]
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)

        # Chat
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        # 处理最后一条完整回复
        final_response = responses[-1] if responses else {}
        return final_response

    def hi_rag_test_top3(self, query, llm_set, w, prompt):
        """
        Hi-RAG方法（Top-3）- 实现Algorithm 1
        
        Stage 1: Candidate Service Retrieval
        Stage 2: Type-Aware Hierarchical Re-ranking
        返回Top-K (K=3) 服务
        """
        # ============ Stage 1: Candidate Service Retrieval ============
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w, flat_flag=False)
        res_no_hi_des = res_no_hi_des[:10]

        if not self.filter_service(res_no_hi_des):
            res_no_hi_des = res_no_hi_des[:20]

        # 如果没有Hi-RAG重排序器，使用原始方法
        if self.hi_rag_reranker is None:
            res_no_hi_des_dict = {}
            for res_j in res_no_hi_des:
                res_no_hi_des_dict[res_j] = f"This is hierarchical information: type={self.summary2other[res_j]['type']}, service={self.summary2other[res_j]['title']}, tool={res_j}"

            value_list = list(res_no_hi_des_dict.values())
            res_no_hi_des_dict_res = {v: k for k, v in res_no_hi_des_dict.items()}

            # 重排序
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
            # ============ Stage 2: Type-Aware Hierarchical Re-ranking ============
            candidate_services = self._prepare_candidate_services(res_no_hi_des)
            
            if not candidate_services:
                # 回退到简单检索
                filter_service_name = []
                filter_port = []
                for res_str in res_no_hi_des[:3]:
                    service_name = self.summary2other[res_str]['service_name']
                    if service_name not in filter_service_name:
                        filter_service_name.append(service_name)
                        filter_port.append(self.summary2other[res_str]['port'])
            else:
                # 执行类型感知的层级重排序
                ranked_services = self.hi_rag_reranker.rerank_services(query, candidate_services)
                
                filter_service_name = []
                filter_port = []
                
                print(f"\n[Hi-RAG Stage 2] Top-3 Services:")
                for idx, (service, score, reranking_info) in enumerate(ranked_services[:3], 1):
                    service_name = service['service_name']
                    port = service['port']
                    
                    if service_name not in filter_service_name:
                        filter_service_name.append(service_name)
                        filter_port.append(port)
                    
                    print(f"\n  Rank {idx}: {service_name}")
                    print(f"    Type: {reranking_info['type']}")
                    print(f"    Domain Gate (β): {reranking_info['beta']:.4f}")
                    print(f"    Final Score: {score:.4f}")
                    print(f"    Top tools by attention (α):")
                    for tool_name, weight in reranking_info['top_tools']:
                        print(f"      - {tool_name}: {weight:.4f}")

        # ============ Final Inference ============
        # Define the agent
        tools = [{
            'mcpServers': {}
        }]

        for i in range(len(filter_service_name[:3])):
            service_find = filter_service_name[i]
            port_find = filter_port[i]

            tools[0]['mcpServers'][service_find] = {
                'url': f"http://localhost:{port_find}/sse"
            }
        
        bot = self.init_agent_service(tools, llm_set=llm_set, sys_mes=prompt)
        
        # Chat
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        # 处理最后一条完整回复
        final_response = responses[-1] if responses else {}
        return final_response

    def mul_infer(self, save_path, llm_set, rag_type=None, topk=1, prompt="请判断所提供的工具是否可以用来解决用户的问题。如果可以,请选择合适的函数进行调用,无需过度思考。如果不可以,请直接回答用户的问题,无需进行过度思考。"):
        """
        多工具推理主函数
        :param save_path: 结果保存路径
        :param llm_set: 大模型配置
        :param rag_type: RAG类型 (None, 'FlatRAG', 'HIRAG')
        :param topk: 检索top-k个服务
        :param prompt: 系统提示
        :return: None
        """
        label_list = []
        
        data_list = json.loads(open(MUL_TEST_DIR, encoding="utf-8").read())

        for line in tqdm(data_list):
            try:
                query_i = line.get('query', '')
                true_tools = line.get('tool_list', [])
                tool_name_pred = []

                # 根据不同的RAG类型选择方法
                if not rag_type:
                    response = self.test(query_i, llm_set)
                elif rag_type == 'FlatRAG':
                    if topk == 1:
                        response = self.rag_test(query_i, llm_set, w=0.1, prompt=prompt)
                    else:
                        response = self.rag_test_top3(query_i, llm_set, w=0.1, prompt=prompt)
                elif rag_type == 'COLBERT':
                    if topk == 1:
                        response = self.colbert_rag_test(query_i, llm_set, prompt=prompt)
                    else:
                        response = self.colbert_rag_test_top3(query_i, llm_set, prompt=prompt)
                elif rag_type == 'HIRAG':
                    if topk == 1:
                        response = self.hi_rag_test(query_i, llm_set, w=0.1, prompt=prompt)
                    else:
                        response = self.hi_rag_test_top3(query_i, llm_set, w=0.1, prompt=prompt)

                line['response'] = response
                print("response:", response)

                # 遍历获取tool name
                for item in response:
                    if item.get('role', '') == 'function':
                        tool_name = item.get('name', '')
                        if tool_name:
                            tool_name_pred.append(tool_name.replace("-", "_").lower())

                line['tool_pred'] = tool_name_pred
                line['tool_label'] = true_tools

                print("=" * 60)
                print(f"Query: {query_i}")
                print(f"Predicted tools ({len(tool_name_pred)}): {tool_name_pred}")
                print(f"True tools ({len(true_tools)}): {true_tools}")

                # 评估预测结果
                # 将预测的工具名与真实工具列表进行匹配
                pred_end = []
                for true_i in true_tools:
                    for pred_i in tool_name_pred:
                        # 检查真实工具名是否在预测工具名中（考虑到可能有前缀）
                        if true_i.lower() == pred_i.lower() or true_i.lower() in pred_i.lower():
                            if true_i not in pred_end:  # 避免重复添加
                                pred_end.append(true_i)
                            break

                # 检查预测是否完全正确（所有工具都被预测到，顺序可以相反）
                if (sorted(pred_end) == sorted(true_tools)) or \
                   (pred_end == true_tools) or \
                   (pred_end[::-1] == true_tools):
                    label_list.append(1)
                else:
                    label_list.append(0)
                
                print(f"Matched tools ({len(pred_end)}): {pred_end}")
                print(f"Match result: {'✓ CORRECT' if label_list[-1] == 1 else '✗ INCORRECT'}")
                print(f"Current accuracy: {sum(label_list)}/{len(label_list)} = {sum(label_list) / len(label_list):.3f}")
                print("=" * 60)

            except Exception as e:
                print(f"Error processing query '{query_i}': {e}")
                label_list.append(0)
                continue

        # 计算最终准确率
        accuracy = sum(label_list) / len(label_list) if label_list else 0
        print(f"\nFinal Accuracy: {accuracy:.3f}")

        # 保存结果
        with open(save_path, "w", encoding="utf-8") as file_write:
            file_write.write(json.dumps(data_list, ensure_ascii=False, indent=4))
        file_write.close()

        print(f"Results saved to {save_path}")
