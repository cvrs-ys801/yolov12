import torch
import sys
import platform


def check_device():
    print("=" * 40)
    print("       🚀 PyTorch 设备检测报告")
    print("=" * 40)

    # 1. 基础环境信息
    print(f"🐍 Python 版本:     {sys.version.split()[0]}")
    print(f"🔥 PyTorch 版本:    {torch.__version__}")
    print(f"🖥️  操作系统:        {platform.system()} {platform.release()}")

    print("-" * 40)

    # 2. 检测 CUDA 是否可用
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("✅ 状态: 成功检测到 GPU (CUDA)！")

        # 获取 GPU 数量
        device_count = torch.cuda.device_count()
        print(f"🔢 GPU 数量:        {device_count}")

        # 遍历打印每个 GPU 的详细信息
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            # 获取显存信息 (转换为 GB)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"   📍 GPU {i}:          {gpu_name} ({total_memory:.2f} GB)")

        print(f"🛠️  CUDA 版本:       {torch.version.cuda}")

        # 3. 实际运算测试 (防止 is_available 为 True 但无法计算的情况)
        print("-" * 40)
        print("🧪 正在进行 Tensor 运算测试...")
        try:
            # 在 GPU 上创建张量
            x = torch.tensor([1.0, 2.0]).cuda()
            y = torch.tensor([3.0, 4.0]).cuda()
            z = x + y
            print(f"   测试计算 (1+3, 2+4): {z.cpu().numpy()}")
            print(f"   计算设备:            {z.device}")
            print("🎉 结论: GPU 配置完美，可以开始炼丹了！")
        except Exception as e:
            print(f"❌ 错误: 虽然检测到 GPU，但运算失败。可能是驱动不匹配。")
            print(f"   报错信息: {e}")

    else:
        print("⚠️ 状态: 未检测到 GPU，正在使用 CPU 跑代码！")
        print("❌ 结论: 你可能安装了 CPU 版本的 PyTorch，或者显卡驱动没装好。")
        print("   (深度学习如果用 CPU 跑，速度会非常非常慢)")

    print("=" * 40)


if __name__ == "__main__":
    check_device()