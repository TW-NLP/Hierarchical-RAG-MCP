import os
from app.rag.model import SimpleRagQA
from config import SIG_TEST_DIR,SERVICE_INFO
from qwen_agent.agents import Assistant
import json
from tqdm import tqdm
from config import SUMMARY_PATH,FAISS_PATH


class SigMCP(object):
    def __init__(self,):
        self.sig_test_dir = SIG_TEST_DIR
        self.summary2other=json.loads(open(SUMMARY_PATH,encoding="utf-8").read())

        self.simple_qa = SimpleRagQA(faiss_path=FAISS_PATH,data_path=SIG_TEST_DIR,embedding_name='summary')

        
    def init_agent_service(self,tools,llm_set,sys_mes=''):
        """

        :param tools:
        :return:
        """
        bot = Assistant(llm=llm_set,
                        function_list=tools,
                        name='',
                        system_message=sys_mes,
                        description="I'm a roboot using the tool calling.")
        return bot


    def test(self,query: str,llm_set):
        pass
    
    def rag_flat(self,query,llm_set,prompt):
        pass

    def rag_test_top3(self,query,llm_set,w,prompt):
        pass
    def hi_rag_test(self,query,llm_set,w,prompt):
        res_no_hi_des=self.simple_qa.qa_engine.search(query,w,flat_flag=False)
        res_no_hi_des=res_no_hi_des[:10]
        res_no_hi_des_dict={}

        for res_j in res_no_hi_des:
            res_no_hi_des_dict[res_j]=f"This is hierarchical information: type={self.summary2other[res_j]['type']}, service={self.summary2other[res_j]['title']}, tool={res_j}"

        value_list=list(res_no_hi_des_dict.values())


        res_no_hi_des_dict_res={v:k for k,v in res_no_hi_des_dict.items()}



        results=self.simple_qa.qa_engine.search_engine.rerank(query, value_list)
        res_list=list(results.keys())
        # 恢复到原数据
        end_summary_list=[res_no_hi_des_dict_res[k] for k in res_list]
        # 选top 1
        res_str=end_summary_list[0]

        service_find=self.summary2other[res_str]['service_name']
        port_find=self.summary2other[res_str]['port']
        # Define the agent
        tools=[{
            'mcpServers': {
                service_find: {
                    'url': f"http://localhost:{port_find}/sse"
                }
            }
        }]
        
        bot = self.init_agent_service(tools,llm_set=llm_set,sys_mes=prompt)

        # Chat
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response
    def filter_service(self,res):
        service_names=[]
        for sercice_des in res:
            if self.summary2other[sercice_des]['service_name'] not in service_names:
                service_names.append(self.summary2other[sercice_des]['service_name'])
        if len(service_names)>=3:
            return True
        else:
            return False
    def hi_rag_test_top3(self,query,llm_set,w,prompt):
        res_no_hi_des=self.simple_qa.qa_engine.search(query,w,flat_flag=False)
        res_no_hi_des=res_no_hi_des[:10]

        if not self.filter_service(res_no_hi_des):
            res_no_hi_des=res_no_hi_des[:20]

        res_no_hi_des_dict={}

        for res_j in res_no_hi_des:
            res_no_hi_des_dict[res_j]=f"This is hierarchical information: type={self.summary2other[res_j]['type']}, service={self.summary2other[res_j]['title']}, tool={res_j}"

        value_list=list(res_no_hi_des_dict.values())


        res_no_hi_des_dict_res={v:k for k,v in res_no_hi_des_dict.items()}



        results=self.simple_qa.qa_engine.search_engine.rerank(query, value_list)
        res_list=list(results.keys())
        end_summary_list=[res_no_hi_des_dict_res[k] for k in res_list]

        
        # Define the agent
        tools=[{
            'mcpServers': {
                
            }
        }]

        filter_service_name=[]
        filter_port=[]


        for sercice_des in end_summary_list:
            if self.summary2other[sercice_des]['service_name'] not in filter_service_name:
                filter_service_name.append(self.summary2other[sercice_des]['service_name'])
                filter_port.append(self.summary2other[sercice_des]['port'])

        

        for i in range(len(filter_service_name[:3])):
            service_find=filter_service_name[i]
            port_find=filter_port[i]

            
            tools[0]['mcpServers'][service_find] = {
                'url': f"http://localhost:{port_find}/sse"
            }
        
        bot = self.init_agent_service(tools,llm_set=llm_set,sys_mes=prompt)
        messages = [{'role': 'user', 'content': query}]
        responses = []
        for response in bot.run(messages=messages):
            responses.append(response)
        
        final_response = responses[-1] if responses else {}
        return final_response

    

    def signal_infer(self,save_path,llm_set,rag_type=None,topk=1,prompt=""):
        """
        :param save_path:
        :param llm_set: 
        :return:
        """
        
        label_list=[]
        
        data_list=json.loads(open(SIG_TEST_DIR,encoding="utf-8").read())

        true_label=''

        for line in tqdm(data_list):

            service_name=line.get('name', '')


            for tool_i in tqdm(line.get('endpoints', [])):
                try:
                    true_label=service_name+tool_i.get('path', '').replace('/', '-')+"_"+tool_i.get('path', '').replace('/', '')+"_"+tool_i.get('method', '').lower()
                    true_label=true_label.replace("-","_").lower()
                    tool_name_list=[]
                    if not rag_type:
                        response = self.test(tool_i.get('query'),llm_set)
                    elif rag_type=='FlatRAG':
                        if topk==1:
                            response = self.rag_test(tool_i.get('query'),llm_set,w=0.1,prompt=prompt)
                        else:
                            response = self.rag_test_top3(tool_i.get('query'),llm_set,w=0.1,prompt=prompt)
                    elif rag_type=='HIRAG':
                        if topk==1:
                            response = self.hi_rag_test(tool_i.get('query'),llm_set,w=0.1,prompt=prompt)
                        else:
                            response = self.hi_rag_test_top3(tool_i.get('query'),llm_set,w=0.1,prompt=prompt)
                    tool_i['response']=response
                    tool_i['tool_pred']=[] 


                    # tool name.  arvix_mcp-list_papers_list_papers_get
                    for item in response:
                        if item.get('role', '')=='function':
                            tool_name = item.get('name', '')
                            if tool_name:
                                tool_name_list.append(tool_name.replace("-","_").lower())
                    tool_i['tool_pred']=tool_name_list
                    tool_i['tool_label']=true_label
                    print("*"*30)
                    print(tool_i['tool_pred'],tool_i['tool_label'])
                    
                
                    if tool_name_list and true_label ==tool_name_list[0]:
                        label_list.append(1)
                    else:
                        label_list.append(0)
                except Exception as e:
                    print(f"Error processing {service_name} - {tool_i.get('path', '')}: {e}")
                    label_list.append(0)
                    continue
                print(sum(label_list) / len(label_list) if label_list else 0)
        #ACC
        accuracy = sum(label_list) / len(label_list) if label_list else 0
        print(f"Accuracy: {accuracy:.3f}")

    

        with open(save_path,"w",encoding="utf-8") as file_write:
            file_write.write(json.dumps(data_list,ensure_ascii=False,indent=4))
        file_write.close()
        




