#!/bin/bash
#SBATCH --job-name=semeval_task
#SBATCH --output=slurm-semeval-%j.out
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ================= 配置区 =================
ENV_NAME="semeval26_starter_pack"  
# ⚠️ 注意：请确认上面 ENV_NAME 是你刚才截图里的 "semeval26_starter_pack" 
# 还是之前的 "StableKeypoints"？请根据你实际 conda 环境名修改！
# ==========================================

# 1. 环境准备
echo "🔧 初始化环境..."
module purge
module load Miniforge3

# 定义 Python 解释器路径
PYTHON_EXEC="$HOME/.conda/envs/$ENV_NAME/bin/python"
ENV_LIB="$HOME/.conda/envs/$ENV_NAME/lib"

# 2. 环境变量设置
export LD_LIBRARY_PATH="${ENV_LIB}:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 遇到任何错误立即停止
set -e

# 3. 运行代码 (Wrapper 模式)
echo "==================================================="
echo "🚀 任务启动: SemEval 2026 Task 10 (Merged Mode)"
echo "📂 当前目录: $(pwd)"
echo "🐍 Python路径: $PYTHON_EXEC"
echo "==================================================="

# --- 阶段 A: 数据检查 ---
if [ ! -f "train_rehydrated.jsonl" ]; then
    echo "⚠️  警告: 未找到 'train_rehydrated.jsonl'。"
    echo "   如果程序报错，请先在登录节点运行 'python rehydrate_data.py' 来生成数据。"
fi

# --- 阶段 B & C: 运行全流程包装器 ---
# 这一行是关键！它替代了之前那一长串的 python 调用
# -u 参数保证 print 输出不被缓存，你能实时看到时间日志
"$PYTHON_EXEC" -u run_all_stages.py

# --- 阶段 D: 打包结果 ---
echo "📦 正在打包 submission.zip..."
if command -v zip >/dev/null 2>&1; then
    zip submission.zip submission.jsonl
else
    echo "   系统未找到 zip 命令，尝试使用 Python 打包..."
    "$PYTHON_EXEC" -c "import zipfile; with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as z: z.write('submission.jsonl')"
fi

echo "✅ 所有任务结束！请下载 submission.zip 提交。"