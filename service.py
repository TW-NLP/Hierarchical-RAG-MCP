import os
import subprocess
import signal
import re
import sys
import json
import time
from config import SERVICE_DIR, SERVICE_INFO

def get_conda_python():
    """
    获取当前 conda 环境的 Python 解释器路径
    """
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        # conda 环境中的 Python 路径
        python_path = os.path.join(conda_prefix, 'bin', 'python')
        if os.path.exists(python_path):
            return python_path
    
    # 如果没有找到，返回当前的 Python
    return sys.executable


def extract_port_from_runpy(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r'mcp\.run\([^)]*port\s*=\s*(\d+)', content)
    if match:
        return int(match.group(1))
    else:
        return None


def service_run():
    """
    遍历 data 目录下的所有子目录，查找并运行其中的 run.py 文件。
    """
    # 获取正确的 Python 解释器
    python_executable = get_conda_python()
    
    print("正在启动服务...")
    print(f"使用 Python: {python_executable}")
    print(f"当前环境: {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'N/A')}")
    
    # 验证 Python 环境
    try:
        result = subprocess.run(
            [python_executable, '-c', 'import fastapi; print("fastapi OK")'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ 环境验证成功: {result.stdout.strip()}")
        else:
            print(f"✗ 警告: 环境可能有问题")
            print(f"  错误: {result.stderr}")
    except Exception as e:
        print(f"✗ 警告: 无法验证环境 - {e}")
    
    print()
    
    root_dir = SERVICE_DIR
    
    # 先检查是否有正在运行的服务
    if os.path.exists(SERVICE_INFO):
        print("检测到已有服务信息文件，先停止旧服务...")
        service_stop()
        print()
    
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(SERVICE_INFO), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 存储服务信息的字典
    services_info = {}
    started_count = 0
    
    # 获取当前完整环境
    env = os.environ.copy()
    
    # 遍历所有子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "run.py" in filenames:
            run_path = os.path.join(dirpath, "run.py")
            service_name = os.path.basename(dirpath)
            
            # 创建日志文件
            log_file = os.path.join(log_dir, f"{service_name}.log")
            
            try:
                with open(log_file, 'w') as log_f:
                    # 写入启动信息到日志
                    log_f.write(f"启动命令: {python_executable} {run_path}\n")
                    log_f.write(f"工作目录: {os.path.dirname(run_path)}\n")
                    log_f.write(f"Python 路径: {python_executable}\n")
                    log_f.write(f"CONDA_PREFIX: {env.get('CONDA_PREFIX', 'N/A')}\n")
                    log_f.write(f"环境: {env.get('CONDA_DEFAULT_ENV', 'N/A')}\n")
                    log_f.write("=" * 60 + "\n")
                    log_f.flush()
                    
                    proc = subprocess.Popen(
                        [python_executable, "-u", run_path],
                        cwd=os.path.dirname(run_path),
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True
                    )
                
                # 记录服务信息
                services_info[service_name] = {
                    'pid': proc.pid,
                    'path': run_path,
                    'log': log_file,
                    'status': 'running'
                }
                
                started_count += 1
                print(f"  ✓ 启动服务: {service_name} (PID: {proc.pid})")
            except Exception as e:
                print(f"  ✗ 启动失败: {service_name} - {e}")
    
    # 等待服务启动
    print("\n等待服务启动...")
    time.sleep(3)
    
    # 验证哪些进程还在运行
    print("\n验证服务状态...")
    running_count = 0
    failed_services = []
    
    for service_name, info in list(services_info.items()):
        pid = info['pid']
        log_file = info.get('log', '')
        
        try:
            os.kill(pid, 0)
            running_count += 1
            
            # 尝试获取端口
            port = None
            try:
                port = extract_port_from_runpy(info['path'])
            except:
                pass
            
            port_info = f", Port: {port}" if port else ""
            print(f"  ✓ {service_name} (PID: {pid}{port_info})")
        except:
            print(f"  ✗ {service_name} (PID: {pid}) 启动失败")
            failed_services.append((service_name, log_file))
            
            # 显示日志的最后几行
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"     最后几行日志:")
                            for line in lines[-3:]:
                                print(f"       {line.strip()}")
                except:
                    pass
            
            del services_info[service_name]
    
    # 保存服务信息
    try:
        os.makedirs(os.path.dirname(SERVICE_INFO), exist_ok=True)
        
        with open(SERVICE_INFO, 'w', encoding='utf-8') as f:
            json.dump(services_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 服务信息已保存到: {SERVICE_INFO}")
        print(f"✓ 成功启动 {running_count}/{started_count} 个服务")
        
        if failed_services:
            print(f"\n⚠ 失败的服务 ({len(failed_services)} 个):")
            for svc, log in failed_services[:5]:  # 只显示前5个
                print(f"  - {svc}")
                print(f"    日志: {log}")
        
    except Exception as e:
        print(f"\n✗ 警告: 保存服务信息失败: {e}")
    
    print(f"\n服务启动完成")


def service_stop():
    """
    停止所有运行中的服务。
    """
    print("正在停止服务...")
    
    if not os.path.exists(SERVICE_INFO):
        print(f"  未找到服务信息文件")
        print("  没有找到运行中的服务")
        return
    
    try:
        with open(SERVICE_INFO, 'r', encoding='utf-8') as f:
            services_info = json.load(f)
    except Exception as e:
        print(f"  ✗ 读取服务信息失败: {e}")
        return
    
    if not services_info:
        print("  没有找到运行中的服务")
        try:
            os.remove(SERVICE_INFO)
        except:
            pass
        return
    
    stopped_count = 0
    not_found_count = 0
    
    for service_name, info in services_info.items():
        pid = info['pid']
        
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.1)
            
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
                print(f"  ✓ 强制终止: {service_name} (PID: {pid})")
            except:
                print(f"  ✓ 已终止: {service_name} (PID: {pid})")
            
            stopped_count += 1
            
        except ProcessLookupError:
            not_found_count += 1
            print(f"  ⚠ 进程已不存在: {service_name} (PID: {pid})")
        except Exception as e:
            print(f"  ✗ 终止失败: {service_name} - {e}")
    
    try:
        os.remove(SERVICE_INFO)
        print(f"\n✓ 服务信息文件已删除")
    except:
        pass
    
    if stopped_count > 0:
        print(f"\n服务停止完成，共停止 {stopped_count} 个服务")
    if not_found_count > 0:
        print(f"有 {not_found_count} 个服务进程已不存在")


def service_status():
    """
    查看服务运行状态
    """
    print("服务运行状态:")
    print("-" * 80)
    
    if not os.path.exists(SERVICE_INFO):
        print("  没有运行中的服务")
        print("-" * 80)
        return
    
    try:
        with open(SERVICE_INFO, 'r', encoding='utf-8') as f:
            services_info = json.load(f)
    except Exception as e:
        print(f"  读取失败: {e}")
        print("-" * 80)
        return
    
    if not services_info:
        print("  没有运行中的服务")
        print("-" * 80)
        return
    
    running_count = 0
    dead_count = 0
    
    for service_name, info in services_info.items():
        pid = info['pid']
        path = info.get('path', '')
        
        try:
            os.kill(pid, 0)
            is_running = True
        except:
            is_running = False
        
        if is_running:
            port_info = ""
            if path and os.path.exists(path):
                try:
                    port = extract_port_from_runpy(path)
                    port_info = f"Port: {port}" if port else ""
                except:
                    pass
            
            running_count += 1
            print(f"  ✓ {service_name:<30} PID: {pid:<8} {port_info}")
        else:
            dead_count += 1
            print(f"  ✗ {service_name:<30} PID: {pid:<8} (已停止)")
    
    print("-" * 80)
    print(f"总计: {running_count} 个运行, {dead_count} 个已停止")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "start":
            service_run()
        elif action == "stop":
            service_stop()
        elif action == "status":
            service_status()
        else:
            print("用法: python service.py {start|stop|status}")
            sys.exit(1)
    else:
        service_run()