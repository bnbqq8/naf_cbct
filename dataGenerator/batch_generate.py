import os
import subprocess
from pathlib import Path

# ================= 配置区域 =================
# 1. 您的数据根目录 (存放所有 Test Set 病例的地方)
DATA_ROOT = "/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1"

# 2. 生成脚本的路径
SCRIPT_PATH = "dataGenerator/generateData.py"

# 3. 配置文件路径 (如果您要生成 2views 数据)
CONFIG_PATH = "dataGenerator/config_2views.yml" 
# 如果是其他配置，请修改这里，例如 "dataGenerator/config.yml"

# 4. 输出的 pickle 文件名 (会保存在每个病例文件夹内)
OUTPUT_NAME = "data_2views" 
# ===========================================

def main():
    # 检查路径是否存在
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 错误: 数据目录不存在 -> {DATA_ROOT}")
        return

    # 获取所有子文件夹 (即 case name)
    # 过滤掉非文件夹的项目，确保只处理目录
    all_items = os.listdir(DATA_ROOT)
    cases = [d for d in all_items if os.path.isdir(os.path.join(DATA_ROOT, d))]
    cases.sort() # 排序，方便看进度

    print(f"📂 在 {DATA_ROOT} 下共发现 {len(cases)} 个病例 (Test Set)")
    print("-" * 50)

    # 遍历每个 Case
    success_count = 0
    for i, case_name in enumerate(cases):
        print(f"🚀 [{i+1}/{len(cases)}] 正在处理: {case_name} ...")
        
        # 拼接命令
        # 相当于命令行执行: python dataGenerator/generateData.py --case xxx --dataPath xxx ...
        cmd = [
            "python", SCRIPT_PATH,
            "--dataPath", DATA_ROOT,     # 强制指定根目录，防止脚本默认值不对
            "--configPath", CONFIG_PATH, # 指定配置文件
            "--case", case_name,         # 只有这个在变
            "--outputName", OUTPUT_NAME  # 输出文件名
        ]
        
        try:
            # 执行命令，如果返回非0 (报错) 则抛出异常
            subprocess.run(cmd, check=True)
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"⚠️  处理失败: {case_name}")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")

    print("-" * 50)
    print(f"✅ 全部完成！成功: {success_count} / 总数: {len(cases)}")

if __name__ == "__main__":
    main()