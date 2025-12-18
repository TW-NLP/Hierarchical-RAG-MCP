import os
import json
import requests
import json5
from typing import List, Dict, Any
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
from qwen_agent.gui import WebUI

from config import SERVICE_INFO


# Step 1: 动态加载 MCP 服务作为工具
def load_mcp_services() -> List[Dict[str, Any]]:
    """
    从 service_info.json 加载所有运行中的 MCP 服务信息
    """
    if not os.path.exists(SERVICE_INFO):
        print(f"警告: 未找到服务信息文件 {SERVICE_INFO}")
        return []
    
    try:
        with open(SERVICE_INFO, 'r', encoding='utf-8') as f:
            services_info = json.load(f)
        
        mcp_services = []
        for service_name, info in services_info.items():
            # 检查进程是否还在运行
            try:
                os.kill(info['pid'], 0)
                
                # 尝试从 run.py 中提取端口信息
                port = extract_port_from_runpy(info['path'])
                if port:
                    mcp_services.append({
                        'name': service_name,
                        'pid': info['pid'],
                        'port': port,
                        'base_url': f'http://localhost:{port}'
                    })
            except:
                continue
        
        print(f"✓ 加载了 {len(mcp_services)} 个 MCP 服务")
        return mcp_services
    
    except Exception as e:
        print(f"✗ 加载 MCP 服务失败: {e}")
        return []


def extract_port_from_runpy(file_path: str) -> int:
    """从 run.py 文件中提取端口号"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        match = re.search(r'mcp\.run\([^)]*port\s*=\s*(\d+)', content)
        if match:
            return int(match.group(1))
    except:
        pass
    return None


def create_mcp_tool_class(service_name: str, base_url: str, tool_description: str = None):
    """
    动态创建 MCP 服务对应的工具类
    """
    tool_name = f'mcp_{service_name}'
    
    # 在类定义之前确定描述
    if tool_description is None:
        tool_description = f'MCP service: {service_name}. Call this service to perform actions. Available at {base_url}'
    
    # 在类定义之前确定参数
    tool_parameters = [{
        'name': 'action',
        'type': 'string',
        'description': f'Action to perform in {service_name} service',
        'required': True
    }, {
        'name': 'params',
        'type': 'object',
        'description': 'Parameters for the action',
        'required': False
    }]
    
    # 保存到闭包变量
    _base_url = base_url
    _service_name = service_name
    
    @register_tool(tool_name)
    class MCPTool(BaseTool):
        # 使用外部定义的变量
        description = tool_description
        parameters = tool_parameters
        
        def call(self, params: str, **kwargs) -> str:
            try:
                parsed_params = json5.loads(params)
                action = parsed_params.get('action', '')
                action_params = parsed_params.get('params', {})
                
                # 调用 MCP 服务的 API（使用闭包变量）
                response = requests.post(
                    f'{_base_url}/{action}',
                    json=action_params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return json5.dumps(response.json(), ensure_ascii=False)
                else:
                    return json5.dumps({
                        'error': f'HTTP {response.status_code}',
                        'message': response.text
                    }, ensure_ascii=False)
            
            except Exception as e:
                return json5.dumps({
                    'error': str(e),
                    'service': _service_name
                }, ensure_ascii=False)
    
    return MCPTool


# Step 2: 加载所有 MCP 服务并注册为工具
def register_all_mcp_services():
    """
    注册所有运行中的 MCP 服务为 Qwen Agent 工具
    """
    mcp_services = load_mcp_services()
    registered_tools = []
    
    for service in mcp_services:
        service_name = service['name']
        base_url = service['base_url']
        
        # 创建并注册工具
        try:
            create_mcp_tool_class(service_name, base_url)
            tool_name = f'mcp_{service_name}'
            registered_tools.append(tool_name)
            print(f"  ✓ 注册工具: {tool_name} -> {base_url}")
        except Exception as e:
            print(f"  ✗ 注册失败: {service_name} - {e}")
    
    return registered_tools



# 配置 LLM
llm_cfg = {
    'model': 'qwen-plus',
    'model_type': 'qwen_dashscope',
    'api_key': 'sk-712e0970e24c41d390eff4bce11b1d00',
    'generate_cfg': {
        'top_p': 0.8
    }
}


# Step 5: 创建 Agent
def create_agent():
    """
    创建集成了所有 MCP 服务的 Agent
    """
    print("\n正在初始化 Agent...")
    
    # 注册所有 MCP 服务
    print("\n加载 MCP 服务:")
    mcp_tools = register_all_mcp_services()
    
    # 组合所有工具
    tools = mcp_tools
    
    print(f"\n✓ 共加载 {len(tools)} 个工具")
    print(f"  工具列表: {', '.join(tools[:10])}{'...' if len(tools) > 10 else ''}")
    
    # 系统指令
    system_instruction = f'''You are a helpful AI assistant with access to multiple tools and services.

Available tools:
- MCP services ({len(mcp_tools)} services): {', '.join(mcp_tools[:5])}{'...' if len(mcp_tools) > 5 else ''}

When using MCP services, call them with appropriate actions and parameters.
'''
    
    # 创建 Agent
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=tools,
    )
    
    return bot


# Step 6: 运行 Agent
def run_cli_mode():
    """
    命令行模式
    """
    bot = create_agent()
    
    messages = []
    print("\n" + "="*60)
    print("Agent 已启动！输入 'quit' 或 'exit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input('\n用户: ')
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not query.strip():
                continue
            
            messages.append({'role': 'user', 'content': query})
            response = []
            response_plain_text = ''
            
            print('\nAgent: ', end='')
            for response in bot.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)
            
            messages.extend(response)
            print()  # 换行
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()


def run_webui_mode():
    """
    Web UI 模式
    """
    bot = create_agent()
    print("\n启动 Web UI...")
    WebUI(bot).run()


if __name__ == "__main__":
    import sys
    
    # 检查服务是否运行
    if not os.path.exists(SERVICE_INFO):
        print("="*60)
        print("警告: 未找到 MCP 服务信息文件")
        print("请先运行: python service.py start")
        print("="*60)
    
    # 选择运行模式
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        run_webui_mode()
    else:
        run_cli_mode()