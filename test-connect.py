import platform
import socket
import os
import getpass
import sys
import urllib.request


def check_environment():
    print("=" * 40)
    print("   🌍 环境检测报告 (Environment Check)")
    print("=" * 40)

    # 1. 操作系统信息 (OS Info)
    # 如果是 Windows，说明是本地；如果是 Linux，极大概率是服务器
    system_name = platform.system()
    print(f"🖥️  操作系统 (OS):      {system_name} {platform.release()}")

    # 2. 主机名 (Hostname)
    # 服务器通常有特定的主机名，比如 'aliyun-ecs' 或 'ubuntu-server'
    hostname = socket.gethostname()
    print(f"🏷️  主机名 (Hostname):    {hostname}")

    # 3. 当前登录用户 (Current User)
    # 远程通常是 'root' 或你创建的服务器特定用户
    user = getpass.getuser()
    print(f"👤 当前用户 (User):      {user}")

    # 4. 脚本运行路径 (Working Directory)
    # 远程路径通常像 '/home/user/project'，本地通常是 'C:\\Users\\...'
    cwd = os.getcwd()
    print(f"📂 运行路径 (Path):      {cwd}")

    # 5. Python 解释器路径 (Interpreter Path)
    # 确认使用的是远程环境的 Python (例如 /usr/bin/python3) 还是本地环境 (Virtualenv)
    print(f"🐍 Python 解释器:        {sys.executable}")

    print("-" * 40)

    # 6. SSH 环境变量检测 (SSH Detection)
    # 如果是通过 SSH 连接运行，通常会有 SSH_CLIENT 或 SSH_TTY 变量
    ssh_client = os.environ.get('SSH_CLIENT') or os.environ.get('SSH_CONNECTION')
    if ssh_client:
        print(f"🔒 SSH 连接检测:         ✅ 检测到 SSH 会话")
        print(f"   (源 IP -> 目标 IP):   {ssh_client.split()[:2]}")
    else:
        print(f"🔒 SSH 连接检测:         ❌ 未检测到标准 SSH 变量 (可能是本地或特殊配置)")

    print("-" * 40)

    # 7.获取公网 IP (Public IP) - 最硬核的证据
    # 如果显示的 IP 是阿里云/腾讯云/AWS 的 IP，那就是在远程
    print("🌐正在获取公网 IP (可能需要几秒)...")
    try:
        with urllib.request.urlopen('https://api.ipify.org', timeout=5) as response:
            ip = response.read().decode('utf-8')
            print(f"📌 当前机器公网 IP:      {ip}")
    except Exception as e:
        print(f"📌 获取公网 IP 失败:      {e}")

    print("=" * 40)

    # 最终判定建议
    if system_name == "Linux" or ssh_client:
        print("✅ 结论: 看起来代码正在【远程服务器】上运行！")
    elif system_name == "Windows" or system_name == "Darwin":
        print("⚠️ 结论: 看起来代码正在【本地机器】上运行！")
    else:
        print("❓ 结论: 无法确定，请检查上面的 IP 和主机名。")


if __name__ == "__main__":
    check_environment()