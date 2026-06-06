#!/bin/bash
# 本地测试脚本 — 运行仿真并计算收益
# 用法: bash tools/test.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "  Agent 算法大赛 - 本地测试"
echo "========================================"
echo ""

# 检查 API Key
if [ -z "$DASHSCOPE_API_KEY" ] && [ -z "$TIANCHI_MODEL_API_KEY" ]; then
    echo "❌ 请先设置 API Key:"
    echo "   export DASHSCOPE_API_KEY=你的key"
    echo ""
    echo "   或者编辑 demo/server/config/config.json 填入 model_api_key"
    exit 1
fi

# 检查数据文件
if [ ! -f "$PROJECT/demo/server/data/cargo_dataset.jsonl" ]; then
    echo "❌ 缺少 cargo_dataset.jsonl"
    echo "   请从赛题包复制到 demo/server/data/"
    exit 1
fi

echo "📦 数据文件: OK"
echo "🔑 API Key: 已设置"
echo ""

# 运行仿真
echo "🚛 运行仿真..."
cd "$PROJECT/demo/server"
python3 main.py
echo ""

# 计算收益
echo "📊 计算收益..."
cd "$PROJECT/demo"
python3 calc_monthly_income.py
echo ""

# 显示结果
echo "========================================"
echo "  测试结果"
echo "========================================"
python3 -c "
import json
with open('$PROJECT/demo/results/monthly_income_202603.json') as f:
    data = json.load(f)

print()
for d in data['drivers']:
    inc = d['income']
    print(f\"{'='*50}\")
    print(f\"司机 {d['driver_id']}\")
    print(f\"  净收益: ¥{inc['net_income']:,.2f}\")
    print(f\"  毛收入: ¥{inc['gross_income']:,.2f}\")
    print(f\"  里程成本: ¥{inc['cost']:,.2f}\")
    print(f\"  偏好罚分: ¥{inc['preference_penalty']:,.2f}\")
    if d.get('validation_error'):
        print(f\"  ❌ 校验失败: {d['validation_error']}\")
    rules = d.get('preference_check', {}).get('rules', [])
    if rules:
        print(f\"  偏好明细:\")
        for r in rules:
            print(f\"    - {r['rule']}: 罚¥{r['penalty']:,.0f}\")
    print()

s = data['summary']
print(f\"{'='*50}\")
print(f\"🏆 总净收益: ¥{s['total_net_income_all_drivers']:,.2f}\")
print(f\"📉 总偏好罚分: ¥{s['total_preference_penalty']:,.2f}\")
print(f\"🔢 Token 消耗: {s['total_token_usage']}\")
print(f\"❌ 失败司机: {s['failed_driver_count']}\")
print(f\"{'='*50}\")
"
