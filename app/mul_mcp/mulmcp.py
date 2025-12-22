import os
from app.rag.model import SimpleRagQA
from config import MUL_TEST_DIR, SERVICE_INFO
from qwen_agent.agents import Assistant
import json
from tqdm import tqdm
from config import SUMMARY_PATH, FAISS_PATH


class MulMCP(object):
    def __init__(self):
        self.mul_test_dir = MUL_TEST_DIR
        self.summary2other = json.loads(open(SUMMARY_PATH, encoding="utf-8").read())
        
        self.simple_qa = SimpleRagQA(
            faiss_path=FAISS_PATH,
            data_path=MUL_TEST_DIR,
            embedding_name='summary'
        )

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

    def hi_rag_test(self, query, llm_set, w, prompt):
        """
        使用层次化RAG检索单个最相关的服务
        :param query: 用户查询
        :param llm_set: LLM配置
        :param w: 搜索权重
        :param prompt: 系统提示
        :return: 最终响应
        """
        # 调用RAG进行工具检索
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w, flat_flag=False)
        res_no_hi_des = res_no_hi_des[:10]
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
        使用层次化RAG检索Top3相关服务
        :param query: 用户查询
        :param llm_set: LLM配置
        :param w: 搜索权重
        :param prompt: 系统提示
        :return: 最终响应
        """
        # 调用RAG进行工具检索
        res_no_hi_des = self.simple_qa.qa_engine.search(query, w, flat_flag=False)
        res_no_hi_des = res_no_hi_des[:10]

        if not self.filter_service(res_no_hi_des):
            res_no_hi_des = res_no_hi_des[:20]

        res_no_hi_des_dict = {}

        for res_j in res_no_hi_des:
            res_no_hi_des_dict[res_j] = f"This is hierarchical information: type={self.summary2other[res_j]['type']}, service={self.summary2other[res_j]['title']}, tool={res_j}"

        value_list = list(res_no_hi_des_dict.values())
        res_no_hi_des_dict_res = {v: k for k, v in res_no_hi_des_dict.items()}

        # 重排序
        results = self.simple_qa.qa_engine.search_engine.rerank(query, value_list)
        res_list = list(results.keys())
        end_summary_list = [res_no_hi_des_dict_res[k] for k in res_list]

        # Define the agent
        tools = [{
            'mcpServers': {}
        }]

        filter_service_name = []
        filter_port = []

        for service_des in end_summary_list:
            if self.summary2other[service_des]['service_name'] not in filter_service_name:
                filter_service_name.append(self.summary2other[service_des]['service_name'])
                filter_port.append(self.summary2other[service_des]['port'])

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


