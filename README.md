# 满帮集团 Agent 算法大赛：卡车司机连续找货决策

> 🏆 基于 Agentic AI 的卡车司机连续找货决策智能体

## 项目结构

```
├── agent/                              # 🧠 智能体核心代码（初赛 & 复赛提交物）
│   ├── __init__.py
│   ├── model_decision_service.py       # 主决策服务入口
│   ├── cargo_evaluator.py              # 货源经济评估（PPM、机会成本、偏好风险）
│   ├── preference_tracker.py           # 偏好状态追踪（跨步合规监控）
│   └── context_builder.py             # Prompt 构建（经济学框架 + 状态 + 策略）
├── demo/                               # 本地测试环境
│   ├── agent -> ../agent/              # 软链接到 agent/
│   ├── server/                         # 仿真入口（赛方提供）
│   ├── simkit/                         # 仿真引擎（赛方提供）
│   └── calc_monthly_income.py          # 收益核算脚本
├── tools/
│   └── setup.sh                        # 一键搭建本地测试环境
└── README.md
```

## 设计理念

本 Agent 融合**经济学原理**与**Agentic AI 范式**进行决策：

| 经济学概念         | 在决策中的应用                           |
|--------------------|------------------------------------------|
| 单位时间利润率(PPM) | 核心排序指标，衡量时间配置效率           |
| 机会成本           | 接一单耗用的时间本可用于接更好的单       |
| 期权价值           | 等待/空驶保留了未来接更好货源的选择权   |
| 边际分析           | 比较每次行动的边际收益与边际成本         |
| 风险规避           | 偏好罚分是非对称下行风险，应优先避免     |
| 空间套利           | 空驶到货源密集区换取更高的预期 PPM       |

## 本地测试

### 1. 环境搭建

```bash
# 一键搭建
bash tools/setup.sh

# 或手动步骤：
cd demo/server
pip install -r requirements.txt
cp config/config.example.json config/config.json
```

### 2. 配置模型

编辑 `demo/server/config/config.json`：

```json
{
  "model_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
  "model_api_key": "你的 DashScope API Key",
  "model_name": "qwen3.5-flash",
  "model_timeout_seconds": 60,
  "simulation_duration_days": 1,
  "simulation_max_steps": 20000,
  "driver_max_total_tokens": 5000000
}
```

或设置环境变量（更安全）：

```bash
export DASHSCOPE_API_KEY=your_key
```

### 3. 确保数据文件就位

```bash
# 如果 demo/server/data/ 下没有数据文件，从赛题包复制：
cp demo_docs_release_20260529/demo/server/data/cargo_dataset.jsonl demo/server/data/
cp demo_docs_release_20260529/demo/server/data/drivers.json demo/server/data/
```

### 4. 运行仿真

```bash
cd demo/server
python main.py
```

生成文件：
- `demo/results/actions_202603_D001_*.jsonl` — 每个司机的逐步动作日志
- `demo/results/run_summary_202603.json` — 仿真汇总

### 5. 计算收益与评分

```bash
cd demo
python calc_monthly_income.py
```

生成文件：
- `demo/results/monthly_income_202603.json` — 最终评分结果

### 6. 查看结果

```bash
# 查看总分
cat demo/results/monthly_income_202603.json | python -m json.tool | grep -A5 summary

# 查看某个司机的收益明细
cat demo/results/monthly_income_202603.json | python -c "
import json, sys
data = json.load(sys.stdin)
for d in data['drivers']:
    print(f\"{d['driver_id']}: net_income={d['income']['net_income']}, penalty={d['income']['preference_penalty']}, orders_completed={d['income'].get('gross_income',0)}\")
    if d.get('validation_error'):
        print(f'  ❌ {d[\"validation_error\"]}')
    if d.get('preference_check',{}).get('rules'):
        for r in d['preference_check']['rules']:
            print(f'  偏好: penalty={r[\"penalty\"]} — {r[\"rule\"]}')
"
```

## 结果解读

`monthly_income_202603.json` 关键字段：

| 字段 | 含义 |
|------|------|
| `drivers[*].income.net_income` | 净收益（核心排名指标） |
| `drivers[*].income.gross_income` | 运费毛收入 |
| `drivers[*].income.cost` | 里程成本 |
| `drivers[*].income.preference_penalty` | 偏好罚分总额 |
| `drivers[*].preference_check.rules` | 每条偏好的判定结果 |
| `drivers[*].validation_error` | 动作合法性校验失败原因 |
| `summary.total_net_income_all_drivers` | 全体净收益总和 |
| `summary.total_token_usage` | Token 消耗统计 |

## 调试技巧

1. **查看动作日志**：`cat demo/results/actions_202603_D001_*.jsonl | head -5`
2. **查看仿真日志**：`cat demo/results/logs/simulation_orchestrator.log`
3. **收益为 0**：检查 `validation_error` 字段，通常是动作校验失败
4. **偏好罚分高**：检查 `preference_check.rules` 中每条规则的明细
5. **修改仿真天数**：编辑 `config.json` 中的 `simulation_duration_days`（默认 1 天用于快速测试，正式评测是 31 天）

## 提交

### 初赛

压缩包内容：
```
demo/
├── agent/          # 本项目的 agent/ 目录
└── results/        # 仿真产物（actions_*.jsonl, run_summary_*.json）
```

### 复赛

压缩包内容：
```
demo/
└── agent/          # 仅 agent/ 目录，不含 results/
```

**注意**：
- 不要提交 `data/` 目录（数据量大，赛方提供）
- 不要提交 `config.json`（含密钥）
- 不要提交 `server/`（除非修改了赛方代码）
- 附加文件须附带说明文档（`SUBMISSION.md`）

## License

本代码为参赛作品。

---

🤖 由 Claude Code 辅助构建
