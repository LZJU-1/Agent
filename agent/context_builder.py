"""上下文构建器：构建发送给 LLM 的完整 prompt。

融合以下要素：
  1. 经济学决策框架（机会成本、边际分析、期权价值）
  2. 司机偏好规则（自然语言原文）
  3. 当前状态快照（时间、位置、收益、偏好追踪结果）
  4. 候选货源经济评估结果
  5. 策略建议（基于时间和空间分析）
"""

from __future__ import annotations

import json
from typing import Any

from .cargo_evaluator import (
    CargoEvaluationResult,
    EvaluatedCargo,
    suggest_reposition_targets,
)
from .preference_tracker import DriverStateSnapshot, PreferenceStatus


# ---------------------------------------------------------------
# System Prompt 模板
# ---------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """你是满帮货运平台的一位资深卡车司机调度智能体。

## 你的角色
你需要在模拟的 2026 年 3 月中，为一辆 4.2 米卡车做出一系列连续决策，目标是**最大化月度净收益**。

**净收益 = 成功接单的总运费收入 − 行驶里程成本 − 偏好违规罚分**

## 决策框架：经济学思维

作为理性决策者，你应当在每一步决策中运用以下经济学原理：

### 1. 单位时间利润率 (Profit per Minute, PPM)
PPM = (运费收入 − 行驶成本) ÷ 总耗时(分钟)，是衡量货源效率的**核心指标**。
- ¥5/min = ¥300/小时（优秀）
- ¥2/min = ¥120/小时（一般）
- ¥1/min = ¥60/小时（较差）
- 负 PPM = 亏本单，除非有战略意义否则不接

### 2. 机会成本
你每接一单所花费的时间，本可以用来接其他货源。因此：
- 长途单（>800 分钟）虽然总收益高，但锁定了大量时间，错失了后续机会
- 短途单灵活，但空驶占比可能偏高
- 关键比较：**该货源的 PPM 是否高于你对后续货源的预期 PPM**

### 3. 等待的期权价值
- 当前货源 PPM 偏低时，等待(wait)保留了接更好货源的权利
- 但等待太久会浪费可运营时间（"时间就是金钱"）
- 策略性休息的黄金时段：深夜/凌晨（货源少，且某些偏好要求在此时段休息）

### 4. 空驶的搜索价值
- 空驶(reposition)付出即期成本，换取进入货源更密集区域的"入场券"
- 只有当预期未来 PPM 的提升 > 空驶成本时才值得空驶
- 空驶到物流枢纽（广州、深圳、东莞、佛山）通常能获得更多货源选择

### 5. 偏好罚分 = 负收益
- ¥2400 的罚分大约相当于 8 小时优秀货源的利润
- 能把偏好当"硬约束"就尽量当硬约束，避免罚分吃掉辛苦赚的利润

### 6. 时间节奏
- 月初可以多接长途单（时间充裕）
- 月末要确保完成偏好中的天数/次数要求
- 夜间货源上新少，是执行休息偏好的最佳窗口
- 装货时间窗：到达太早要原地等待（效率损失），到达太晚则接单失败

## 三类动作

1. **take_order**：接单 {"cargo_id": "货源ID"}
   - 流程：空驶到装货点 → 装货窗等待 → 干线运输到卸货点
   - 接单后时间会大幅推进，无法中途改变主意

2. **wait**：原地休息 {"duration_minutes": 正整数}
   - 位置不变，纯粹消耗时间
   - 用于等待更好货源上线、满足休息偏好
   - 每批货源浏览耗时 = ceil(返回条数/10) 分钟

3. **reposition**：空驶到目标位置 {"latitude": 纬度, "longitude": 经度}
   - 按 60km/h 计算耗时与成本
   - 用于进入货源更密集的区域

## 重要提醒
- 货源有有效期(remove_time)，过期自动下架
- 装货时间窗(load_time)过期则接单失败（仅消耗 1 分钟）
- 只能接与你的 truck_length 匹配的货源
- 输出必须是纯 JSON，不要有任何额外文本
"""


# ---------------------------------------------------------------
# User Prompt 构建
# ---------------------------------------------------------------

def build_decision_prompt(
    snapshot: DriverStateSnapshot,
    cargo_result: CargoEvaluationResult,
    max_cargo_display: int = 20,
) -> str:
    """构建单步决策的 user prompt。"""

    parts: list[str] = []

    # ---- 1. 当前状态 ----
    parts.append(_build_status_section(snapshot))

    # ---- 2. 偏好规则与状态 ----
    parts.append(_build_preference_section(snapshot))

    # ---- 3. 候选货源 ----
    parts.append(_build_cargo_section(cargo_result, max_cargo_display))

    # ---- 4. 策略分析 ----
    parts.append(_build_strategy_section(snapshot, cargo_result))

    # ---- 5. 输出指令 ----
    parts.append(_build_output_instruction())

    return "\n\n".join(parts)


def build_system_prompt(driver_preferences_text: str = "") -> str:
    """构建 system prompt。"""
    prompt = _SYSTEM_PROMPT_TEMPLATE
    if driver_preferences_text:
        prompt += f"\n\n## 本司机的个性化偏好规则\n\n以下是该司机的个人偏好，违反将导致罚分。请在每一步决策中**严格遵守**这些偏好：\n\n{driver_preferences_text}\n\n偏好的罚分规则：每条偏好有 penalty_amount（每次/每天违规扣分）和 penalty_cap（最高累计扣分上限，null=无上限）。"
    return prompt


# ---------------------------------------------------------------
# 各 Section 构建函数
# ---------------------------------------------------------------

def _build_status_section(snapshot: DriverStateSnapshot) -> str:
    """构建当前状态部分。"""
    day = snapshot.simulation_day
    hour = snapshot.simulation_hour
    minute = snapshot.simulation_minute
    total_days_in_month = 31

    # 时间段判断
    if 6 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 14:
        period = "中午"
    elif 14 <= hour < 18:
        period = "下午"
    elif 18 <= hour < 22:
        period = "晚间"
    else:
        period = "深夜/凌晨"

    # 月进度
    month_progress = day / total_days_in_month * 100

    lines = [
        "## 📍 当前状态",
        "",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| 司机ID | {snapshot.driver_id} |",
        f"| 仿真时间 | 3月{day}日 {hour:02d}:{minute:02d}（{period}，第{day}天/{total_days_in_month}天，月进度{month_progress:.0f}%） |",
        f"| 仿真分钟 | {snapshot.simulation_minutes} |",
        f"| 当前位置 | ({snapshot.current_lat:.4f}, {snapshot.current_lng:.4f}) |",
        f"| 累计接单数 | {snapshot.completed_order_count} 单 |",
        f"| 累计毛收入 | ¥{snapshot.gross_income_so_far:,.2f} |",
        f"| 累计里程 | {snapshot.total_distance_km:.1f} km |",
        f"| 估算净收益 | ¥{snapshot.estimated_net_so_far:,.2f} |",
        f"| 活跃天数 | {snapshot.total_active_days} 天 |",
        f"| 完全休息天数 | {snapshot.total_full_rest_days} 天 |",
    ]

    # 最后动作
    if snapshot.last_action_type:
        last_status = "✅" if snapshot.last_action_success else "❌"
        lines.append(f"| 上一步动作 | {last_status} {snapshot.last_action_type} |")

    return "\n".join(lines)


def _build_preference_section(snapshot: DriverStateSnapshot) -> str:
    """构建偏好规则与状态部分。"""
    lines = [
        "## ⚠️ 偏好规则与当前状态",
        "",
        "以下是该司机的个性化偏好。**违反将导致罚分，务必在决策中遵守：**",
        "",
    ]

    has_any = False
    for ps in snapshot.preference_statuses:
        has_any = True
        # 状态图标
        if ps.status == "ok":
            icon = "✅"
        elif ps.status == "violated":
            icon = "🔴"
        else:
            icon = "⚠️"

        cap_str = f"¥{ps.penalty_cap:,.0f}" if ps.penalty_cap is not None else "无上限"
        lines.append(f"**偏好 {ps.index + 1}** {icon}")
        lines.append(f"> {ps.content}")
        lines.append(f"> 罚分: ¥{ps.penalty_amount:,.0f}/次 | 上限: {cap_str}")

        if ps.hints:
            lines.append(f"> 📊 追踪数据: {'; '.join(ps.hints)}")
        lines.append("")

    if not has_any:
        lines.append("（该司机无特殊偏好）\n")

    return "\n".join(lines)


def _build_cargo_section(
    cargo_result: CargoEvaluationResult,
    max_display: int,
) -> str:
    """构建候选货源部分。"""
    evaluated = cargo_result.evaluated[:max_display]

    lines = [
        "## 📦 候选货源（按综合评分排序，显示前{}条）".format(len(evaluated)),
        "",
        f"当前区域货源密度: {cargo_result.area_cargo_density:.2f} | "
        f"Top5 平均 PPM: ¥{cargo_result.top5_profit_per_minute:.2f}/min | "
        f"全部平均 PPM: ¥{cargo_result.avg_profit_per_minute:.2f}/min",
        f"过滤掉 {cargo_result.filtered_out_count} 条不兼容货源（车长不匹配等）",
        "",
    ]

    if not evaluated:
        lines.append("⚠️ **当前无可接货源！**建议 reposition 到货源密集区域或 wait 等待新货源上线。")
        return "\n".join(lines)

    # 表头
    lines.append(
        "| # | 货源ID | 品类 | 净利(¥) | PPM(¥/min) | 耗时(min) | "
        "空驶(km) | 干线(km) | 装货窗 | 偏好风险 | 评分 |"
    )
    lines.append(
        "|---|--------|------|----------|------------|-----------|"
        "----------|----------|--------|----------|------|"
    )

    for i, ev in enumerate(evaluated, 1):
        # 装货窗
        load_time = ev.raw.get("load_time")
        if isinstance(load_time, list) and len(load_time) == 2:
            lt_str = f"{load_time[0][-8:-3]}~{load_time[1][-8:-3]}" if len(str(load_time[0])) > 8 else str(load_time)
        else:
            lt_str = "无限制"

        # 风险标记
        if ev.preference_risk_score > 0.3:
            risk_icon = "🔴高"
        elif ev.preference_risk_score > 0.1:
            risk_icon = "🟡中"
        else:
            risk_icon = "🟢低"

        # PPM 颜色标记
        ppm = ev.profit_per_minute
        if ppm >= 3.0:
            ppm_str = f"**{ppm:.2f}** 🔥"
        elif ppm >= 1.5:
            ppm_str = f"{ppm:.2f}"
        else:
            ppm_str = f"{ppm:.2f} ⚠️"

        lines.append(
            f"| {i} | {ev.cargo_id} | {ev.cargo_name[:6]} | "
            f"{ev.net_profit:+.0f} | {ppm_str} | {ev.total_time_min} | "
            f"{ev.pickup_distance_km:.0f} | {ev.haul_distance_km:.0f} | "
            f"{lt_str} | {risk_icon} | {ev.composite_score:.0f} |"
        )

    # 如果有更多货源
    if len(cargo_result.evaluated) > max_display:
        lines.append(f"\n*(还有 {len(cargo_result.evaluated) - max_display} 条货源未显示，如需查看特定类型请告知)*")

    return "\n".join(lines)


def _build_strategy_section(
    snapshot: DriverStateSnapshot,
    cargo_result: CargoEvaluationResult,
) -> str:
    """构建策略分析部分。"""
    lines = [
        "## 🧠 策略分析",
        "",
    ]

    hour = snapshot.simulation_hour
    day = snapshot.simulation_day
    density = cargo_result.area_cargo_density
    top5_ppm = cargo_result.top5_profit_per_minute
    month_progress = day / 31.0

    # 时间段分析
    if 0 <= hour < 6:
        lines.append("- 🌙 **深夜时段 (0-6点)**：新货源上线少，多数偏好要求此时休息。**强烈建议 wait** 至早晨，一举两得。")
    elif 6 <= hour < 9:
        lines.append("- 🌅 **早晨时段 (6-9点)**：货源开始密集上线，是接单黄金窗口。")
    elif 9 <= hour < 17:
        lines.append("- ☀️ **白天运营时段 (9-17点)**：货源充足，积极接单或空驶到更优区域。")
    elif 17 <= hour < 21:
        lines.append("- 🌆 **傍晚时段 (17-21点)**：仍有货源，但开始减少。质优则接，否则考虑休息。")
    else:
        lines.append("- 🌙 **夜间时段 (21-24点)**：货源减少，检查是否有休息偏好需要满足。")

    # 货源密度分析
    if density < 0.1:
        lines.append("- 📉 **当前区域货源极度稀缺**！强烈建议 **reposition** 到物流枢纽（广州/深圳/东莞/佛山）。")
    elif density < 0.3:
        lines.append(f"- 📊 **当前区域货源偏少**（密度={density:.2f}）。如果 Top5 PPM < ¥1.5/min，建议 reposition。")
    else:
        lines.append(f"- 📈 **当前区域货源充足**（密度={density:.2f}）。可以择优接单，等待更好的机会。")

    # PPM 分析
    if top5_ppm > 3.0:
        lines.append(f"- 🔥 **货源质量优秀**（Top5 PPM=¥{top5_ppm:.2f}/min）。优先接单。")
    elif top5_ppm > 1.5:
        lines.append(f"- 👍 **货源质量尚可**（Top5 PPM=¥{top5_ppm:.2f}/min）。接高 PPM 单，观望低 PPM 单。")
    else:
        lines.append(f"- 👎 **货源质量较差**（Top5 PPM=¥{top5_ppm:.2f}/min）。除非有高 PPM 单，否则考虑 wait 或 reposition。")

    # 月末策略
    if month_progress > 0.8:
        remaining_days = 31 - day + 1
        lines.append(f"- ⏰ **月末冲刺**（仅剩{remaining_days}天）：检查偏好中的月度要求是否已满足，避免末月罚分。优先短途单以增加灵活性。")
    elif month_progress < 0.2:
        lines.append("- 🚀 **月初阶段**：时间充裕，可考虑长途高收益单。同时注意为月度偏好目标（如休息天数）打好基础。")

    return "\n".join(lines)


def _build_output_instruction() -> str:
    """构建输出格式指令。"""
    return """## 📋 输出要求

请基于以上信息，输出一个 JSON 对象表示你的决策：

```json
{"action": "take_order", "params": {"cargo_id": "货源ID"}}
```
或
```json
{"action": "wait", "params": {"duration_minutes": 正整数}}
```
或
```json
{"action": "reposition", "params": {"latitude": 纬度, "longitude": 经度}}
```

**注意：**
- 只输出 JSON，不要输出任何解释、markdown 或额外文本
- 选择 cargo_id 时确保该货源在上面的候选列表中
- wait 的 duration_minutes 至少为 1，建议每次休息 120-480 分钟
- reposition 的坐标必须在合理范围内（广东省约 lat 20-25, lng 110-117）"""
