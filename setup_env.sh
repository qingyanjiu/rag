#!/usr/bin/env bash
# ============================================================
# Qwen3.5 微调环境安装脚本
# 用法: bash setup_env.sh
# 说明: 在全新容器中安装所有微调依赖
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  微调环境安装脚本"
echo "  项目: $(basename "$SCRIPT_DIR")"
echo "========================================"
echo ""

# ---------- 1. 检测 Python ----------
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "❌ 未找到 Python3，请先安装 Python 3.10+"
    exit 1
fi
echo "✅ Python: $($PYTHON --version)"

# ---------- 2. 检测 / 安装 uv ----------
if command -v uv &>/dev/null; then
    echo "✅ uv: $(uv --version)"
else
    echo "📦 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 重新加载 PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ uv: $(uv --version)"
fi

# ---------- 3. 检测 CUDA ----------
if command -v nvidia-smi &>/dev/null; then
    echo "✅ GPU:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1
else
    echo "⚠️ 未检测到 nvidia-smi，请确保已安装 NVIDIA 驱动"
fi

# ---------- 4. 创建虚拟环境 ----------
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 创建虚拟环境..."
    uv venv "$VENV_DIR" --python 3.12
    echo "✅ 虚拟环境已创建: $VENV_DIR"
else
    echo "✅ 虚拟环境已存在: $VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# ---------- 4.5 配置国内镜像 ----------
# 优先使用清华源，其次阿里云镜像，可通过环境变量覆盖
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
# 如果设置了 USE_CN_MIRROR=false 则走官方源
if [ "${USE_CN_MIRROR:-true}" = "true" ]; then
    echo "📦 使用国内镜像加速: PIP_INDEX=$PIP_INDEX"
    PIP_INDEX_ARGS=(--index-url "$PIP_INDEX")
else
    echo "📦 使用 PyPI 官方源"
    PIP_INDEX_ARGS=()
fi

# ---------- 5. 安装 PyTorch (CUDA 12.8) ----------
echo "📦 安装 PyTorch (CUDA 12.8)..."
if [ "${USE_CN_MIRROR:-true}" = "true" ]; then
    # 国内镜像：PyTorch 走清华镜像（cu128 对应版本）
    echo "   使用 PyTorch 官方源（PyTorch whl 需走官方，国内镜像可能版本不全）"
    uv pip install --python "$VENV_DIR" \
        torch==2.10.0 \
        torchvision \
        --index-url "$PYTORCH_INDEX"
else
    uv pip install --python "$VENV_DIR" \
        torch==2.10.0 \
        torchvision \
        --index-url "$PYTORCH_INDEX"
fi

echo "✅ PyTorch 安装完成"

# ---------- 6. 安装微调依赖 ----------
echo "📦 安装微调依赖..."

# 按顺序安装，处理依赖关系
uv pip install --python "$VENV_DIR" \
    --upgrade \
    "${PIP_INDEX_ARGS[@]}" \
    transformers>=4.48.0 \
    datasets>=3.0.0 \
    accelerate>=1.3.0 \
    huggingface-hub>=0.28.0 \
    sentencepiece>=0.2.0 \
    tokenizers>=0.21.0 \
    pyyaml>=6.0 \
    wandb>=0.19.0

echo "✅ 基础依赖安装完成"

# ---------- 7. 安装 unsloth ----------
echo "📦 安装 unsloth + QLoRA 依赖..."
uv pip install --python "$VENV_DIR" \
    --upgrade \
    "${PIP_INDEX_ARGS[@]}" \
    "unsloth>=2025.3.0"

echo "✅ unsloth 安装完成"

# ---------- 8. 验证安装 ----------
echo ""
echo "========================================"
echo "  验证安装"
echo "========================================"
python3 -c "
import sys, torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
try:
    import unsloth
    print(f'unsloth: {unsloth.__version__}')
except Exception:
    print('unsloth: 未安装')
try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except Exception:
    pass
"

# ---------- 9. 验证模型文件 ----------
echo ""
echo "========================================"
echo "  验证模型文件"
echo "========================================"
MODEL_DIR="$SCRIPT_DIR/models/Qwen3.5-0.5B"
if [ -f "$MODEL_DIR/model.safetensors" ] || ls "$MODEL_DIR"/*.safetensors 1>/dev/null 2>&1; then
    echo "✅ 本地模型文件存在: $MODEL_DIR"
    ls -lh "$MODEL_DIR/"
else
    echo "⚠️ 未找到本地模型文件: $MODEL_DIR"
    echo "   训练时会自动从 HuggingFace 下载（使用 unsloth/Qwen3.5-0.5B）"
fi

echo ""
echo "========================================"
echo "  ✅ 环境安装完成！"
echo "========================================"
echo ""
echo "使用方式:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $SCRIPT_DIR"
echo "  python train.py --config config.yaml"
echo ""
echo "或者单步:"
echo "  cd $SCRIPT_DIR && uv run --with-venv $VENV_DIR python train.py --config config.yaml"
