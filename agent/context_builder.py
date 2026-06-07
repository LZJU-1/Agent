"""上下文构建器：构建发送给 LLM 的完整 prompt。

设计原则：
  1. 偏好合规（尤其是每日休息）> 一切，铁律级别
  2. 经济学原理辅助决策，但永远不凌驾于偏好之上
  3. 休息状态以最醒目的方式呈现，无法忽视
"""

from __future__ import annotations

from typing import Any

from .cargo_evaluator import CargoEvaluationResult
from .preference_tracker import DriverStateSnapshot


# ============================================================================
# System Prompt
# ============================================================================

_SYSTEM_PROMPT = """你是满帮货运平台的卡车司机调度智能体，在 2026 年 3 月中做出一系列连续决策。

## 🚨 铁律：偏好合规 > 利润

偏好违规的罚分极其惨重（¥2,400/天），接一整天货都赚不回来。
**每一步决策前，首先检查：今天是否已经满足了所有休息类偏好？**

- 如果今天还没休息够 → **必须 wait**，不要接单
- 如果当前时间在偏好要求的休息窗口内（如 0-6点）→ **必须 wait**
- 只有今天休息已达标，才考虑接单赚钱
- 接单后时间会大幅推进，可能跨过休息窗口，导致当天违规！接了长途单大概率会毁掉当天的休息

## 偏好罚分换算（记住这些数字）
- ¥2,400 罚分 ≈ 接 3-5 单的净利润
- ¥1,800 罚分 ≈ 接 2-3 单的净利润
- 为了赚 ¥500 而触发 ¥2,400 罚分 = 净亏 ¥1,900

## 三类动作
- `wait` → {"duration_minutes": N} 原地休息（最重要、最安全的动作）
- `take_order` → {"cargo_id": "X"} 接单运输（时间大幅推进，有风险）
- `reposition` → {"latitude": lat, "longitude": lng} 空驶

## 规则
- 货源有有效期，过期下架
- 装货窗过期则接单失败（仅耗 1 分钟）
- 只能接 truck_length 匹配的货源
- 输出只含 JSON，无其他文本
- wait 时 duration_minutes 至少 60，建议 180-480"""


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

    # 🚨 休息状态放在最前面！无法忽视
    parts.append(_rest_alert_section(snapshot))
    parts.append(_status_section(snapshot))
    parts.append(_preference_section(snapshot))
    parts.append(_cargo_section(cargo_result, max_cargo_display))
    parts.append(_post_delivery_section(cargo_result, snapshot))
    if scout_results:
        parts.append(_scout_section(scout_results, cost_per_km))
    parts.append(_strategy_section(snapshot, cargo_result))
    parts.append(_output_instruction())

    return "\n\n".join(parts)


def build_system_prompt(driver_preferences_text: str = "") -> str:
    """构建 system prompt，嵌入司机偏好。"""
    if driver_preferences_text:
        return (
            _SYSTEM_PROMPT
            + "\n\n## 本司机偏好规则\n\n"
            + driver_preferences_text
            + "\n\n**以上每一条都是铁律。违反 = 扣钱 = 白干。**"
        )
    return _SYSTEM_PROMPT


# ============================================================================
# 🚨 休息状态警报 — 最高优先级，放在 prompt 最前面
# ============================================================================

def _rest_alert_section(s: DriverStateSnapshot) -> str:
    """分析今日休息状态，给出明确的行为指令。"""
    today = s.simulation_day - 1
    h, m = s.simulation_hour, s.simulation_minute
    today_minutes = h * 60 + m

    # 合并今日休息区间求最长连续休息
    intervals = s.daily_rest_intervals.get(today, [])
    today_longest_rest = _longest_merged_rest(intervals)

    lines = [
        "=" * 50,
        "🚨 今日休息状态 (最高优先级！)",
        "=" * 50,
    ]

    # 检查每个偏好中的休息要求
    has_rest_requirement = False
    for ps in s.preference_statuses:
        text = ps.content

        # ---- 连续休息要求（如"每天至少连续休息8小时"） ----
        if _is_continuous_rest_pref(text):
            has_rest_requirement = True
            required_h = _extract_rest_hours(text)
            required_min = required_h * 60 if required_h else 480

            rest_h = today_longest_rest // 60
            rest_min = today_longest_rest % 60
            deficit = required_min - today_longest_rest

            lines.append(f"📋 要求: {text[:60]}...")
            lines.append(f"   需要: {required_h}小时连续休息")

            if today_longest_rest >= required_min:
                lines.append(f"   ✅ 已达标！今日最长连续休息 {rest_h}h{rest_min}m")
            else:
                deficit_h = deficit // 60
                deficit_m = deficit % 60
                # 判断今天是否还有救
                remaining_today = 1440 - today_minutes
                if remaining_today >= deficit:
                    lines.append(f"   🔴 未达标！今日仅休息 {rest_h}h{rest_min}m，还差 {deficit_h}h{deficit_m}m")
                    lines.append(f"   ⚡ 今天还剩 {remaining_today//60}h{remaining_today%60}m，来得及！**立即 wait {deficit} 分钟！**")
                else:
                    lines.append(f"   💀 今日已无法补救（仅剩{remaining_today//60}h，需{deficit_h}h）。但仍应立即休息，避免连续违规。")

        # ---- 定时休息要求（如"零点到早上六点睡觉"） ----
        elif _is_scheduled_rest_pref(text):
            has_rest_requirement = True
            start_h, end_h = _extract_rest_window(text)
            lines.append(f"📋 要求: {text[:60]}...")
            lines.append(f"   休息窗口: {start_h:02d}:00 - {end_h:02d}:00")

            # 检查是否在当前窗口内
            in_window = _in_time_window(h, start_h, end_h)
            if in_window:
                lines.append(f"   🔴 当前 {h:02d}:{m:02d} 正在休息窗口内！**必须 wait！不要接单！**")
            else:
                # 检查今天窗口是否已经过了
                if h >= end_h:
                    # 检查今天是否在窗口内休息过
                    window_rest = _rest_during_window(intervals, today, start_h, end_h)
                    if window_rest > 0:
                        lines.append(f"   ✅ 今日休息窗口已满足（休息了 {window_rest}min）")
                    else:
                        lines.append(f"   💀 今日休息窗口({start_h:02d}-{end_h:02d})已过，未休息！明天必须遵守。")
                else:
                    # 窗口还没到
                    lines.append(f"   ⏰ 休息窗口 {start_h:02d}:00 开始，现在是 {h:02d}:{m:02d}，到点必须 wait")

    # 没有检测到休息偏好的情况
    if not has_rest_requirement:
        # 通用休息建议
        if today_longest_rest > 0:
            lines.append(f"今日最长连续休息: {today_longest_rest//60}h{today_longest_rest%60}m")
        if h < 6:
            lines.append(f"当前凌晨 {h:02d}:{m:02d}，建议休息至早晨")

    # ---- 日期和地点提醒 ----
    import re
    for ps in s.preference_statuses:
        text = ps.content
        # 特殊日期提醒
        dates = re.findall(r'(?:三月|3月)?\s*(\d+)\s*[号日]', text)
        for d_str in dates:
            d = int(d_str)
            if d == s.simulation_day:
                lines.append(f"🔴 今天3月{d}号！必须执行：{text[:100]}")
            elif d == s.simulation_day + 1:
                lines.append(f"⚠️ 明天3月{d}号！提前规划：{text[:100]}")

        # 地点要求 + 完成进度
        city_matches = re.findall(r'(?:在|去|到)\s*([一-鿿]{2,4})(?:区|市|县|镇)', text)
        day_matches = re.findall(r'(\d+)\s*(?:个|天)', text)
        if city_matches and day_matches:
            target = int(day_matches[0])
            city = city_matches[0]
            # 统计已完成天数
            done = sum(1 for r in s.accepted_cargo_regions if city in r)
            if done < target:
                lines.append(f"🔴 需{target}天到{city}，已完成{done}天！请主动找{city}的货源！")
            else:
                lines.append(f"✅ {city}已满足：{done}/{target}天")

    lines.append("=" * 50)
    return "\n".join(lines)


# ============================================================================
# 状态
# ============================================================================

def _status_section(s: DriverStateSnapshot) -> str:
    h, m = s.simulation_hour, s.simulation_minute
    period = "深夜" if h < 6 else "上午" if h < 12 else "下午" if h < 18 else "晚间" if h < 22 else "深夜"
    return (
        f"## 状态\n"
        f"3月{s.simulation_day}日 {h:02d}:{m:02d} {period} | "
        f"位置({s.current_lat:.4f},{s.current_lng:.4f}) | "
        f"已接{s.completed_order_count}单 | "
        f"毛收入¥{s.gross_income_so_far:,.0f} | "
        f"活跃{s.total_active_days}天 | 全休{s.total_full_rest_days}天"
    )


# ============================================================================
# 偏好
# ============================================================================

def _preference_section(s: DriverStateSnapshot) -> str:
    lines = ["## 所有偏好规则"]
    for ps in s.preference_statuses:
        icon = {"ok": "✅", "violated": "🔴", "needs_attention": "⚠️"}.get(ps.status, "❓")
        lines.append(f"{icon} P{ps.index+1}: {ps.content}")
        if ps.hints:
            lines.append(f"   → {'; '.join(ps.hints[:3])}")
    if not s.preference_statuses:
        lines.append("(无特殊偏好)")
    return "\n".join(lines)


# ============================================================================
# 货源
# ============================================================================

def _cargo_section(cr: CargoEvaluationResult, max_display: int) -> str:
    evs = cr.evaluated[:max_display]
    lines = [
        f"## 候选货源 (密度{cr.area_cargo_density:.2f}, 过滤{cr.filtered_out_count}条, Top5PPM=¥{cr.top5_profit_per_minute:.2f})",
        "",
    ]
    if not evs:
        lines.append("⚠️ 无可接货源")
        return "\n".join(lines)

    lines.append("|#|ID|品类|净利¥|PPM|耗时|空驶|干线|装货窗|风险|")
    lines.append("|-|--|----|----:|---|----|---:|---:|------|----|")
    for i, ev in enumerate(evs, 1):
        lt = ev.raw.get("load_time")
        lt_str = f"{str(lt[0])[-8:-3]}" if isinstance(lt, list) and len(lt) == 2 else "-"
        risk = "🔴" if ev.preference_risk_score > 0.3 else ("🟡" if ev.preference_risk_score > 0.1 else "🟢")
        ppm = f"**{ev.profit_per_minute:.2f}**" if ev.profit_per_minute >= 3.0 else f"{ev.profit_per_minute:.2f}"
        lines.append(
            f"|{i}|{ev.cargo_id}|{ev.cargo_name[:5]}|{ev.net_profit:+.0f}|"
            f"{ppm}|{ev.total_time_min}|{ev.pickup_distance_km:.0f}|"
            f"{ev.haul_distance_km:.0f}|{lt_str}|{risk}|"
        )
    return "\n".join(lines)


# ============================================================================
# 接单后状态
# ============================================================================

def _post_delivery_section(cr: CargoEvaluationResult, s: DriverStateSnapshot) -> str:
    pda = cr.post_delivery_analysis[:3]
    if not pda:
        return ""

    lines = ["## 接单后预估", ""]
    lines.append("|ID|卸货地|到达时间|时段|耗时|净利¥|")
    lines.append("|-|------|--------|----|----|----:|")

    for p in pda:
        night = "🌙夜" if p["is_night_arrival"] else "☀️昼"
        # 检查到达时间是否会破坏休息
        warn = ""
        if p["is_night_arrival"]:
            warn = "⚠️夜间到达"
        lines.append(
            f"|{p['cargo_id']}|{p['end_city'][:6]}|"
            f"3/{p['finish_day']} {p['finish_hour']:02d}:00|{night}{warn}|"
            f"{p['total_time_spent']}min|{p['net_profit']:+.0f}|"
        )

    # 警告：接单会导致休息不足
    h, _ = s.simulation_hour, s.simulation_minute
    for p in pda:
        if p["total_time_spent"] > 300 and h >= 18:
            lines.append(f"\n⚠️ 注意：接单耗时 {p['total_time_spent']}min，可能跨过休息窗口！")
            break

    return "\n".join(lines)


# ============================================================================
# 侦察
# ============================================================================

def _scout_section(scout_results: list[dict[str, Any]], cost_per_km: float) -> str:
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
    return "\n".join(lines)


# ============================================================================
# 策略
# ============================================================================

def _strategy_section(
    snapshot: DriverStateSnapshot,
    cargo_result: CargoEvaluationResult,
) -> str:
    h = snapshot.simulation_hour
    day = snapshot.simulation_day
    ppm = cargo_result.top5_profit_per_minute

    hints = ["## 决策指南"]

    # 优先级1: 检查休息需求（从 rest_alert 中已经强调过，这里简化）
    if h < 6:
        hints.append("- 🔴 深夜时段：默认 wait，除非今天休息已完全达标")
    elif h >= 21:
        hints.append("- 🟠 夜间 {h}点：优先确保休息达标，再考虑接单")
    else:
        # 白天：只有休息达标了才建议接单
        hints.append("- 🟢 白天运营：休息达标后可接单。优先短途单（<300min），避免跨过夜间休息窗口")

    # PPM 参考
    if ppm >= 3:
        hints.append(f"- PPM=¥{ppm:.2f}/min 优秀，但不要为了高 PPM 牺牲休息")
    elif ppm < 1.5:
        hints.append(f"- PPM=¥{ppm:.2f}/min 偏低，如果没有好的短途单，建议 wait")

    # 月末
    remaining = 31 - day + 1
    if remaining <= 3:
        hints.append(f"- ⏰ 仅剩{remaining}天！确保月度偏好已达标")

    return "\n".join(hints)


def _output_instruction() -> str:
    return (
        "## 输出\n"
        "只输出 JSON，无其他文字：\n"
        '{"action":"wait","params":{"duration_minutes":N}}  ← 休息是默认选择\n'
        '{"action":"take_order","params":{"cargo_id":"X"}}  ← 仅当休息已达标\n'
        '{"action":"reposition","params":{"latitude":lat,"longitude":lng}}'
    )


# ============================================================================
# 辅助函数
# ============================================================================

def _longest_merged_rest(intervals: list[tuple[int, int]]) -> int:
    """合并重叠休息区间，返回最长连续休息分钟数。"""
    if not intervals:
        return 0
    sorted_int = sorted(intervals)
    merged = [sorted_int[0]]
    for s, e in sorted_int[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return max((e - s) for s, e in merged)


def _is_continuous_rest_pref(text: str) -> bool:
    """判断偏好是否要求连续休息（如'每天至少连续休息X小时'）。"""
    return any(kw in text for kw in ["连续", "休息满", "休息满", "熄火"]) and \
           any(kw in text for kw in ["每天", "每日"]) and \
           not _is_scheduled_rest_pref(text)


def _is_scheduled_rest_pref(text: str) -> bool:
    """判断偏好是否要求特定时段休息（如'零点到早上六点睡觉'）。"""
    import re
    return bool(re.search(r'(?:零点|晚上|夜间|凌晨|早上|\d+点).*(?:到|至|～).*(?:点|早上|凌晨)', text)) and \
           any(kw in text for kw in ["睡觉", "休息", "停着", "熄火"])


def _extract_rest_hours(text: str) -> int | None:
    """提取连续休息要求的小时数。"""
    import re
    m = re.search(r'(\d+)\s*小时', text)
    if m:
        return int(m.group(1))
    return None


def _extract_rest_window(text: str) -> tuple[int, int]:
    """提取定时休息窗口 (start_hour, end_hour)。"""
    import re
    # 匹配如 "零点...到...六点" "23:00...到...04:00"
    nums = re.findall(r'(\d+)\s*点', text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    # 零点特殊处理
    if "零点" in text or "0点" in text or "00:00" in text:
        return 0, 6  # 默认 0-6
    return 0, 6


def _in_time_window(hour: int, start: int, end: int) -> bool:
    """判断当前小时是否在 [start, end) 窗口内。"""
    if start < end:
        return start <= hour < end
    else:  # 跨日窗口如 23-4
        return hour >= start or hour < end


def _rest_during_window(
    intervals: list[tuple[int, int]],
    day: int,
    start_h: int,
    end_h: int,
) -> int:
    """计算在指定时间窗口内的休息分钟数。"""
    day_start = day * 1440
    window_start = day_start + start_h * 60
    window_end = day_start + end_h * 60
    if end_h <= start_h:
        window_end += 1440

    total = 0
    for s, e in intervals:
        overlap_start = max(s, window_start)
        overlap_end = min(e, window_end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total
