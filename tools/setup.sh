#!/bin/bash
# 本地测试环境搭建脚本
# 用法: bash tools/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Agent 算法大赛 - 本地测试环境搭建 ==="
echo ""

# 1. 检查 demo/server 目录
if [ ! -d "$PROJECT_ROOT/demo/server" ]; then
    echo "❌ demo/server 目录不存在，请确保项目结构完整"
    exit 1
fi

# 2. 安装 Python 依赖
echo "📦 安装 Python 依赖..."
cd "$PROJECT_ROOT/demo/server"
pip install -r requirements.txt
echo ""

# 3. 创建配置文件（如果不存在）
if [ ! -f "$PROJECT_ROOT/demo/server/config/config.json" ]; then
    echo "📝 创建配置文件 config.json（从 config.example.json 复制）..."
    cp "$PROJECT_ROOT/demo/server/config/config.example.json" \
       "$PROJECT_ROOT/demo/server/config/config.json"
    echo "⚠️  请编辑 demo/server/config/config.json，填入你的 model_api_key"
    echo "   或设置环境变量: export DASHSCOPE_API_KEY=your_key"
else
    echo "✅ config.json 已存在"
fi
echo ""

# 4. 检查数据文件
if [ ! -f "$PROJECT_ROOT/demo/server/data/cargo_dataset.jsonl" ]; then
    echo "⚠️  缺少 cargo_dataset.jsonl，请从赛题包复制到 demo/server/data/"
fi
if [ ! -f "$PROJECT_ROOT/demo/server/data/drivers.json" ]; then
    echo "⚠️  缺少 drivers.json，请从赛题包复制到 demo/server/data/"
fi

# 5. 创建 results 目录
mkdir -p "$PROJECT_ROOT/demo/results"

echo ""
echo "=== 设置完成 ==="
echo ""
echo "🚀 运行测试:"
echo "   cd demo/server && python main.py"
echo ""
echo "📊 计算收益:"
echo "   cd demo && python calc_monthly_income.py"
echo ""
echo "📋 查看结果:"
echo "   cat demo/results/monthly_income_202603.json"
echo ""
