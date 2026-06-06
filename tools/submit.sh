#!/bin/bash
# 提交包生成脚本
# 用法: bash tools/submit.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBMIT_DIR="$PROJECT/submission_$TIMESTAMP"

echo "📦 生成提交包..."
echo ""

# 创建干净目录
rm -rf "$SUBMIT_DIR"
mkdir -p "$SUBMIT_DIR/demo/agent"
mkdir -p "$SUBMIT_DIR/demo/results"

# 复制 agent 代码
echo "→ 复制 agent/ ..."
cp "$PROJECT/agent/"*.py "$SUBMIT_DIR/demo/agent/"

# 复制 results（如果存在）
if ls "$PROJECT/demo/results/"*.jsonl 2>/dev/null 1>&2 || ls "$PROJECT/demo/results/"*.json 2>/dev/null 1>&2; then
    echo "→ 复制 results/ ..."
    cp "$PROJECT/demo/results/"*.jsonl "$SUBMIT_DIR/demo/results/" 2>/dev/null || true
    cp "$PROJECT/demo/results/"*.json "$SUBMIT_DIR/demo/results/" 2>/dev/null || true
else
    echo "⚠️  results/ 为空（初赛需要仿真产物，请先运行 tools/test.sh）"
fi

# 生成 ZIP
ZIP_NAME="submission_${TIMESTAMP}.zip"
cd "$SUBMIT_DIR"
zip -r "$PROJECT/$ZIP_NAME" demo/ -x "*.DS_Store" "*/__pycache__/*"
cd "$PROJECT"

echo ""
echo "========================================"
echo "✅ 提交包已生成: $ZIP_NAME"
echo "========================================"
echo ""
echo "📋 内容:"
unzip -l "$PROJECT/$ZIP_NAME" | tail -20
echo ""
echo "📤 上传到天池平台即可。"
echo ""
echo "⚠️ 注意:"
echo "  - 初赛: ZIP 需包含 demo/agent/ + demo/results/"
echo "  - 复赛: ZIP 只需包含 demo/agent/（删除 demo/results/）"
echo "  - 不要提交 data/ 或 server/ 或 config.json"

# 清理
rm -rf "$SUBMIT_DIR"
