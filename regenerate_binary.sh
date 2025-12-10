#!/bin/bash
#SBATCH --job-name=regen_bin
#SBATCH --output=slurm-regen-bin-%j.out
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --time=02:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ================= 配置区 (与原脚本保持一致) =================
ENV_NAME="semeval26_starter_pack"
# =========================================================

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

# 🚨 核心：防止 Python 找不到当前目录下的模块
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 遇到任何错误立即停止
set -e

# 3. 运行推理
echo "==================================================="
echo "🚀 启动: 重新生成 Binary Detection 答案"
echo "📂 当前目录: $(pwd)"
echo "🐍 Python路径: $PYTHON_EXEC"
echo "==================================================="

# 运行推理脚本 (这会生成 submission.jsonl)
echo "▶️  正在运行 infer_binary.py ..."
"$PYTHON_EXEC" infer_binary.py

# 4. 重命名结果 (关键步骤)
# 将默认生成的 submission.jsonl 改名为 binary 专用名
if [ -f "submission.jsonl" ]; then
    echo "💾 正在重命名输出文件..."
    mv submission.jsonl submission_binary.jsonl
    echo "✅ 成功！结果已保存为: submission_binary.jsonl"
else
    echo "❌ 错误：未找到 submission.jsonl，推理可能失败。"
    exit 1
fi

echo "🎉 任务结束。现在你可以运行合并脚本了。"