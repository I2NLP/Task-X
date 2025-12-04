#!/bin/bash
set -e  # 遇到错误立即停止

# ================= 配置区 =================
ENV_NAME="semeval26_starter_pack"
REQ_FILE="requirements.txt"    # 修改1: 替换 yaml 文件名
PYTHON_VER="3.10"              # 修改2: 指定 Python 版本 (txt文件不包含这个信息，必须手动指定)

# 路径保持原样，无需修改
MY_CONDA_ENVS="$HOME/.conda/envs"
GHOST_DIR="$HOME/.local/lib/python3.10/site-packages"
# ==========================================

echo "🛠️  开始执行环境安装流程..."

# 1. 加载基础模块
module purge
module load Miniforge3

# 2. 清理幽灵依赖 (保持原样，这对隔离至关重要)
if [ -d "$GHOST_DIR" ]; then
    echo "🧹 删除 .local 个人库..."
    rm -rf "$GHOST_DIR"
fi

# 3. 删除旧环境 (保持原样)
if [ -d "$MY_CONDA_ENVS/$ENV_NAME" ]; then
    echo "🗑️  正在删除旧环境..."
    conda env remove -n $ENV_NAME -y
fi

# 4. 创建新环境 (修改3: 核心变动)
echo "📦 正在创建隔离环境..."

# 第一步：让 Conda 建房子 (安装 Python)
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

# 第二步：让 Pip 搬家具 (安装 txt 里的包)
# 使用绝对路径的 pip，确保 100% 安装在这个新环境里，绝不污染 base
echo "⬇️  正在安装依赖..."
"$MY_CONDA_ENVS/$ENV_NAME/bin/pip" install -r "$REQ_FILE"

# 5. 验证安装 (保持原样)
echo "✅ 验证中..."
"$MY_CONDA_ENVS/$ENV_NAME/bin/python" -c "import torch; print(f'Torch installed: {torch.__version__}')"

echo "🎉 环境搭建完成！"