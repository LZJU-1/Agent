"""上下文构建器：构建发送给 LLM 的完整 prompt。

融合：
  1. 经济学决策框架（PPM、机会成本、期权价值、MDP 前瞻）
  2. 司机偏好规则（自然语言原文 + 合规追踪）
  3. 当前状态快照（时间、位置、收益、偏好状态）
  4. 候选货源经济评估（PPM 排序 + 风险标记 + 状态转移）
  5. 多点侦察结果（附近枢纽货源质量对比）
  6. 策略建议（时段分析 + 月末提醒）
"""

from __future__ import annotations

from typing import Any

from .cargo_evaluator import CargoEvaluationResult
from .preference_tracker import DriverStateSnapshot


# ============================================================================
# System Prompt
# ============================================================================

_SYSTEM_PROMPT = """你是满帮货运平台的卡车司机调度智能体，需在 2026 年 3 月中做出一系列连续决策，目标是**最大化月度净收益**。

## 核心经济学原理

**1. PPM (Profit Per Minute) = 净收益 ÷ 总耗时**
这是衡量货源效率的核心 KPI。¥3/min 以上为优秀，¥1-2/min 一般，<¥1/min 较差。
永远优先选择 PPM 最高的货源——时间是最稀缺的资源。

**2. 机会成本**
接一个 800 分钟的货源，意味着放弃了 800 分钟内可能出现的好货源。
长途单总收益高但锁定时间长；短途单灵活但空驶占比高。权衡标准：PPM。

**3. 等待的期权价值**
当前货源的 PPM 偏低时，等待 (wait) 保留了接更好货源的选择权。
深夜/凌晨是等待的最佳窗口——货源上新少，且可能满足休息类偏好。

**4. 空驶的搜索价值**
空驶 (reposition) 付出即期成本，换取进入货源密集区域的"入场券"。
只有当预期 PPM 提升带来的额外收益 > 空驶成本时，空驶才是值得的。

**5. 偏好罚分 = 负收益**
一条 ¥2400 的罚分 ≈ 8 小时优秀货源的净利润。把偏好视为硬约束优先遵守。

**6. MDP 前瞻思维**
接单不只是看当前收益，还要看接单后你会出现在哪里、什么时间。
卸货地在物流枢纽附近 = 接单后有更多好选择 = 正向期权价值。
深夜到达陌生地点 = 可能被迫空驶或等待 = 负向期权价值。

## 三类动作
- `take_order` → {"cargo_id": "X"} 接单并完成运输（时间大幅推进）
- `wait` → {"duration_minutes": N} 原地休息 N 分钟
- `reposition` → {"latitude": lat, "longitude": lng} 空驶到目标位置

## 规则
- 货源有有效期，过期下架；装货窗过期则接单失败（仅耗 1 分钟）
- 只能接 truck_length 匹配的货源
- 接单后货源从池中移除，先到先得
- 输出只含 JSON，无其他文本"""


# ============================================================================
# User Prompt 构建
# ============================================================================

def build_decision_prompt(
    snapshot: DriverStateSnapshot,
    cargo_result: CargoEvaluationResult,
    scout_results: list[dict[str, Any]] | None = None,
    cost_per_km: float = 1.5,
    max_cargo_display: int = 20,
) -> str:
    """构建单步决策的 user prompt。"""
    parts: list[str] = []

    parts.append(_status_section(snapshot))
    parts.append(_preference_section(snapshot))
    parts.append(_cargo_section(cargo_result, max_cargo_display))
    parts.append(_post_delivery_section(cargo_result))
    if scout_results:
        parts.append(_scout_section(scout_results, cost_per_km))
    parts.append(_strategy_section(snapshot, cargo_result, scout_results))
    parts.append(_output_instruction())

    return "\n\n".join(parts)


def build_system_prompt(driver_preferences_text: str = "") -> str:
    """构建 system prompt，嵌入司机偏好。"""
    if driver_preferences_text:
        return (
            _SYSTEM_PROMPT
            + "\n\n## 本司机偏好规则（严格遵守！违反将扣分）\n\n"
            + driver_preferences_text
        )
    return _SYSTEM_PROMPT


# ============================================================================
# Section 构建
# ============================================================================

def _status_section(s: DriverStateSnapshot) -> str:
    """当前状态。"""
    h, m = s.simulation_hour, s.simulation_minute
    period = (
        "深夜" if h < 6 else "上午" if h < 12 else
        "下午" if h < 18 else "晚间" if h < 22 else "深夜"
    )
    return (
        f"## 状态\n"
        f"3月{s.simulation_day}日 {h:02d}:{m:02d} {period} | "
        f"第{s.simulation_day}/31天 ({s.simulation_day/31*100:.0f}%) | "
        f"位置({s.current_lat:.4f},{s.current_lng:.4f})\n"
        f"已接{s.completed_order_count}单 | "
        f"毛收入¥{s.gross_income_so_far:,.0f} | "
        f"里程{s.total_distance_km:.0f}km | "
        f"活跃{s.total_active_days}天 | 全休{s.total_full_rest_days}天"
    )


def _preference_section(s: DriverStateSnapshot) -> str:
    """偏好规则 + 今日预警。"""
    lines = ["## 偏好规则与今日预警"]

    today = s.simulation_day - 1  # 0-based day index
    today_rest_intervals = s.daily_rest_intervals.get(today, [])
    today_longest_rest = 0
    if today_rest_intervals:
        # 合并重叠区间求最长连续休息
        sorted_int = sorted(today_rest_intervals)
        merged = [sorted_int[0]]
        for start, end in sorted_int[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        today_longest_rest = max((e - s) for s, e in merged)

    today_active = s.daily_active_minutes.get(today, 0)
    hour = s.simulation_hour

    # 今日预警
    warnings = []
    if today_longest_rest > 0:
        warnings.append(f"今日已连续休息 {today_longest_rest//60}h{today_longest_rest%60}m")
    if today_active > 0:
        warnings.append(f"今日已活跃 {today_active} 分钟")
    if hour >= 20 and today_longest_rest < 480:
        warnings.append(f"⚠️ 已{hour}点，今日连续休息仅{today_longest_rest//60}h{today_longest_rest%60}m，建议尽快安排长时间休息！")

    # 月末紧迫提醒
    remaining = 31 - s.simulation_day
    for ps in s.preference_statuses:
        if "整天" in ps.content or "完全" in ps.content:
            needed = 3 if "三" in ps.content or "3" in ps.content else (2 if "两" in ps.content or "2" in ps.content else None)
            if needed and s.total_full_rest_days < needed and remaining < needed - s.total_full_rest_days + 1:
                warnings.append(f"🔴 仅剩{remaining}天！需{needed}天全休，已完成{s.total_full_rest_days}天，必须立即安排全休！")

    lines.append(f">> 今日状态: {'; '.join(warnings) if warnings else '正常'}")

    for ps in s.preference_statuses:
        icon = {"ok": "✅", "violated": "🔴", "needs_attention": "⚠️"}.get(ps.status, "❓")
        lines.append(f"{icon} P{ps.index+1}: {ps.content[:80]}{'...' if len(ps.content)>80 else ''}")
        if ps.hints:
            lines.append(f"   → {'; '.join(ps.hints[:3])}")
    if not s.preference_statuses:
        lines.append("(无特殊偏好)")
    return "\n".join(lines)


def _cargo_section(cr: CargoEvaluationResult, max_display: int) -> str:
    """候选货源表格（紧凑格式）。"""
    evs = cr.evaluated[:max_display]
    lines = [
        f"## 候选货源 (当前区域密度{cr.area_cargo_density:.2f}, 过滤{cr.filtered_out_count}条)",
        "",
    ]
    if not evs:
        lines.append("⚠️ 无可接货源！建议 reposition 或 wait。")
        return "\n".join(lines)

    # 紧凑表头
    lines.append("|#|ID|品类|净利¥|PPM|耗时|空驶|干线|装货窗|风险|评分|")
    lines.append("|-|--|----|----:|---|----|---:|---:|------|----|----|")

    for i, ev in enumerate(evs, 1):
        # 装货窗缩写
        lt = ev.raw.get("load_time")
        if isinstance(lt, list) and len(lt) == 2:
            lt_str = f"{str(lt[0])[-8:-3]}~{str(lt[1])[-8:-3]}"
        else:
            lt_str = "-"

        risk = "高" if ev.preference_risk_score > 0.3 else ("中" if ev.preference_risk_score > 0.1 else "低")
        ppm = f"**{ev.profit_per_minute:.2f}**" if ev.profit_per_minute >= 3.0 else (
            f"{ev.profit_per_minute:.2f}⚠️" if ev.profit_per_minute < 1.0 else f"{ev.profit_per_minute:.2f}"
        )

        lines.append(
            f"|{i}|{ev.cargo_id}|{ev.cargo_name[:5]}|{ev.net_profit:+.0f}|"
            f"{ppm}|{ev.total_time_min}|{ev.pickup_distance_km:.0f}|"
            f"{ev.haul_distance_km:.0f}|{lt_str}|{risk}|{ev.composite_score:.0f}|"
        )

    if len(cr.evaluated) > max_display:
        lines.append(f"\n*(+{len(cr.evaluated)-max_display}条未显示)*")
    return "\n".join(lines)


def _post_delivery_section(cr: CargoEvaluationResult) -> str:
    """接单后状态转移分析（Top-3 货源）。"""
    pda = cr.post_delivery_analysis[:3]
    if not pda:
        return ""

    lines = ["## 接单后状态预估 (Top3)", ""]
    lines.append("|ID|卸货地|到达时间|到达时段|净利¥|PPM|距枢纽|")
    lines.append("|-|------|--------|--------|----:|---|------|")

    for p in pda:
        night = "🌙夜" if p["is_night_arrival"] else "☀️昼"
        month = "⚠️月末" if p["is_month_end"] else ""
        exceed = "❌超月" if p.get("exceeds_month") else ""
        tag = " ".join(filter(None, [night, month, exceed]))
        lines.append(
            f"|{p['cargo_id']}|{p['end_city'][:6]}|"
            f"3/{p['finish_day']} {p['finish_hour']:02d}:00|{tag}|"
            f"{p['net_profit']:+.0f}|{p['profit_per_minute']:.2f}|{p['end_hub_distance_km']:.0f}km|"
        )

    return "\n".join(lines)


def _scout_section(scout_results: list[dict[str, Any]], cost_per_km: float) -> str:
    """多点侦察结果。"""
    if not scout_results:
        return ""

    lines = ["## 附近枢纽侦察", ""]
    lines.append("|枢纽|距离km|空驶成本¥|耗时min|货源数|Top5PPM|")
    lines.append("|-|------|--------|------|------|-------|")

    for sr in scout_results:
        ppm = f"**{sr['top5_ppm']:.2f}**" if sr['top5_ppm'] >= 2.0 else f"{sr['top5_ppm']:.2f}"
        lines.append(
            f"|{sr['hub_name']}|{sr['distance_km']:.0f}|"
            f"{sr['repos_cost']:.0f}|{sr['repos_time_min']}|"
            f"{sr['cargo_count']}|{ppm}|"
        )

    # 空驶是否值得的判断
    best = max(scout_results, key=lambda s: s["top5_ppm"])
    current_ppm = scout_results[0].get("_current_top5_ppm", 0) if scout_results else 0
    if best["top5_ppm"] > 2.0:
        lines.append(f"\n💡 {best['hub_name']} 货源质量明显更好(Top5 PPM={best['top5_ppm']:.2f})，空驶过去需要 {best['repos_time_min']} 分钟，成本 ¥{best['repos_cost']:.0f}。")

    return "\n".join(lines)


def _strategy_section(
    snapshot: DriverStateSnapshot,
    cargo_result: CargoEvaluationResult,
    scout_results: list[dict[str, Any]] | None,
) -> str:
    """策略分析。"""
    h = snapshot.simulation_hour
    day = snapshot.simulation_day
    ppm = cargo_result.top5_profit_per_minute
    density = cargo_result.area_cargo_density

    hints = ["## 策略提示"]

    # 时段策略
    if h < 6:
        hints.append("- 🌙 深夜：货源少，建议 wait 至早晨。如偏好要求此时休息，务必遵守。")
    elif h < 9:
        hints.append("- 🌅 早晨：货源开始密集上线，积极接单。")
    elif h < 17:
        hints.append("- ☀️ 白天运营：择优接单，PPM < ¥2 可等等看。")
    elif h < 21:
        hints.append("- 🌆 傍晚：货源减少，有好单快接。")
    else:
        hints.append("- 🌙 夜间：检查休息偏好，考虑 wait 至明天。")

    # PPM 分析
    if ppm >= 3:
        hints.append(f"- 🔥 Top5 PPM=¥{ppm:.2f}/min 优秀，优先接高 PPM 单。")
    elif ppm >= 1.5:
        hints.append(f"- 👍 Top5 PPM=¥{ppm:.2f}/min 尚可，择优接单。")
    else:
        hints.append(f"- 👎 Top5 PPM=¥{ppm:.2f}/min 偏低。{'考虑 reposition 到侦察到的枢纽。' if scout_results else '考虑 wait 等待更好货源或 reposition 到枢纽。'}")

    # 月末提醒
    remaining = 31 - day + 1
    if remaining <= 3:
        hints.append(f"- ⏰ 仅剩{remaining}天！检查偏好月度要求是否满足。避免超长单。")
    elif remaining <= 7:
        hints.append(f"- 📅 月末临近（剩{remaining}天），注意偏好要求。")

    # 侦察引导
    if scout_results:
        hints.append("- 🔭 已侦察附近枢纽（见上表），空驶前比较预期收益。")

    return "\n".join(hints)


def _output_instruction() -> str:
    """输出格式。"""
    return (
        "## 输出\n"
        '{"action":"take_order","params":{"cargo_id":"X"}}\n'
        '{"action":"wait","params":{"duration_minutes":N}}\n'
        '{"action":"reposition","params":{"latitude":lat,"longitude":lng}}\n'
        "只输出JSON，无其他文本。"
    )
