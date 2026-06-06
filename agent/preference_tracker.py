"""偏好追踪器：跨步骤追踪司机偏好满足状态。

设计原则：
  - 不硬编码任何偏好规则（偏好以自然语言文本形式存在，由 LLM 理解）
  - 追踪通用指标：每日休息、活动天数、接单品类/区域、空驶距离等
  - 将追踪到的状态以结构化方式呈现给 LLM，辅助其判断偏好合规性
  - 给出"风险提示"但不由代码做确定性判断
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreferenceStatus:
    """单条偏好的追踪状态。"""
    index: int                          # 偏好序号（0-based）
    content: str                        # 偏好原文
    penalty_amount: float               # 每次违规罚分
    penalty_cap: float | None           # 罚分上限
    # 追踪指标（由历史动作推算）
    status: str = "unknown"             # "ok" | "at_risk" | "violated"
    hints: list[str] = field(default_factory=list)  # 面向 LLM 的状态提示


@dataclass
class DriverStateSnapshot:
    """司机当前状态的完整快照（供 prompt 构建使用）。"""
    driver_id: str
    simulation_minutes: int
    simulation_day: int                 # 当月第几天（1-31）
    simulation_hour: int                # 当天小时（0-23）
    simulation_minute: int              # 当天分钟（0-59）
    current_lat: float
    current_lng: float
    completed_order_count: int
    gross_income_so_far: float          # 截至目前累计毛收入（元）
    total_distance_km: float            # 累计行驶里程
    estimated_net_so_far: float         # 估算净收益
    # 偏好状态
    preference_statuses: list[PreferenceStatus] = field(default_factory=list)
    # 每日统计（按 day_index 0-30）
    daily_order_count: dict[int, int] = field(default_factory=dict)
    daily_active_minutes: dict[int, int] = field(default_factory=dict)
    daily_rest_intervals: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    # 接单品类与区域
    accepted_categories: list[str] = field(default_factory=list)
    accepted_cargo_regions: list[str] = field(default_factory=list)
    # 活跃天数
    total_active_days: int = 0
    total_full_rest_days: int = 0       # 完全不出车的天数
    # 最后动作
    last_action_type: str = ""
    last_action_params: dict[str, Any] = field(default_factory=dict)
    last_action_success: bool = True


class PreferenceTracker:
    """偏好状态追踪器。

    每步决策前调用 update() 更新状态，
    然后通过 snapshot() 获取用于决策的当前状态快照。
    """

    def __init__(self, driver_id: str) -> None:
        self._driver_id = driver_id
        self._preferences: list[dict[str, Any]] = []
        self._pref_statuses: list[PreferenceStatus] = []
        self._history_records: list[dict[str, Any]] = []
        self._step_count = 0

        # 每日追踪
        self._daily_orders: dict[int, int] = {}          # day -> count
        self._daily_active_min: dict[int, int] = {}      # day -> active minutes
        self._daily_rest: dict[int, list[tuple[int, int]]] = {}  # day -> [(start_min, end_min)]
        self._accepted_categories: list[str] = []
        self._accepted_regions: list[str] = []
        self._completed_order_count = 0
        self._gross_income = 0.0
        self._total_distance = 0.0
        self._total_cost = 0.0
        self._last_action_type = ""
        self._last_action_params: dict[str, Any] = {}
        self._last_action_success = True

        # 当前状态
        self._sim_minutes = 0
        self._current_lat = 0.0
        self._current_lng = 0.0

    # ---- 公开方法 ----

    def set_preferences(self, preferences: list[dict[str, Any]]) -> None:
        """设置司机的偏好列表。"""
        self._preferences = preferences
        self._pref_statuses = []
        for i, pref in enumerate(preferences):
            self._pref_statuses.append(PreferenceStatus(
                index=i,
                content=str(pref.get("content", "")),
                penalty_amount=float(pref.get("penalty_amount", 0) or 0),
                penalty_cap=float(pref.get("penalty_cap", 0)) if pref.get("penalty_cap") is not None else None,
            ))

    def update(
        self,
        simulation_minutes: int,
        current_lat: float,
        current_lng: float,
        decision_history: list[dict[str, Any]] | None = None,
    ) -> None:
        """根据最新仿真状态和历史记录更新追踪状态。"""
        self._sim_minutes = simulation_minutes
        self._current_lat = current_lat
        self._current_lng = current_lng

        if decision_history:
            new_records = decision_history[len(self._history_records):]
            for record in new_records:
                self._process_record(record)
            self._history_records = list(decision_history)

        # 重新评估每条偏好的状态
        self._reassess_preferences()

    def snapshot(self, cost_per_km: float = 0.0) -> DriverStateSnapshot:
        """生成当前状态快照。"""
        day = self._sim_minutes // 1440
        day_minutes = self._sim_minutes % 1440
        hour = day_minutes // 60
        minute = day_minutes % 60

        estimated_net = self._gross_income - self._total_cost

        return DriverStateSnapshot(
            driver_id=self._driver_id,
            simulation_minutes=self._sim_minutes,
            simulation_day=day + 1,
            simulation_hour=hour,
            simulation_minute=minute,
            current_lat=self._current_lat,
            current_lng=self._current_lng,
            completed_order_count=self._completed_order_count,
            gross_income_so_far=round(self._gross_income, 2),
            total_distance_km=round(self._total_distance, 2),
            estimated_net_so_far=round(estimated_net, 2),
            preference_statuses=list(self._pref_statuses),
            daily_order_count=dict(self._daily_orders),
            daily_active_minutes=dict(self._daily_active_min),
            daily_rest_intervals=dict(self._daily_rest),
            accepted_categories=list(self._accepted_categories),
            accepted_cargo_regions=list(self._accepted_regions),
            total_active_days=sum(1 for v in self._daily_active_min.values() if v > 0),
            total_full_rest_days=sum(
                1 for d in range(day + 1)
                if self._daily_active_min.get(d, 0) == 0
            ),
            last_action_type=self._last_action_type,
            last_action_params=self._last_action_params,
            last_action_success=self._last_action_success,
        )

    @property
    def step_count(self) -> int:
        return self._step_count

    # ---- 内部方法 ----

    def _process_record(self, record: dict[str, Any]) -> None:
        """处理一条历史动作记录，更新追踪指标。"""
        self._step_count += 1

        action_obj = record.get("action", {})
        action_name = str(action_obj.get("action", "")).strip().lower()
        params = action_obj.get("params", {}) or {}
        result = record.get("result", {}) or {}

        self._last_action_type = action_name
        self._last_action_params = dict(params)
        self._last_action_success = bool(result.get("accepted", True))

        # 解析时间区间
        step_elapsed = int(record.get("step_elapsed_minutes", 0))
        query_scan = int(record.get("query_scan_cost_minutes", 0))
        end_minutes = int(result.get("simulation_progress_minutes", 0))
        action_start = end_minutes - step_elapsed + query_scan
        action_end = end_minutes
        action_exec_cost = step_elapsed - query_scan

        action_day = action_start // 1440

        if action_name == "take_order":
            accepted = bool(result.get("accepted", False))
            if accepted:
                self._completed_order_count += 1
                cargo_id = str(params.get("cargo_id", ""))
                self._last_action_success = True

                # 里程与收益（从 result 中获取）
                pickup_km = float(result.get("pickup_deadhead_km", 0) or 0)
                haul_km = float(result.get("haul_distance_km", 0) or 0)
                self._total_distance += pickup_km + haul_km

                # 收益估算：用 result 中的信息
                income_eligible = bool(result.get("income_eligible", True))

            # 记录活跃分钟（接单尝试也算活跃）
            if action_exec_cost > 0:
                self._daily_active_min[action_day] = (
                    self._daily_active_min.get(action_day, 0) + action_exec_cost
                )

        elif action_name == "reposition":
            # 空驶：记录里程与活跃分钟
            distance_km = float(result.get("distance_km", 0) or 0)
            self._total_distance += distance_km
            if action_exec_cost > 0:
                self._daily_active_min[action_day] = (
                    self._daily_active_min.get(action_day, 0) + action_exec_cost
                )

        elif action_name == "wait":
            duration = int(params.get("duration_minutes", 0))
            if duration > 0:
                # 记录为休息区间
                if action_day not in self._daily_rest:
                    self._daily_rest[action_day] = []
                self._daily_rest[action_day].append((action_start, action_end))
                # 休息不算活跃

        # 更新日订单数
        if action_name == "take_order" and self._last_action_success:
            self._daily_orders[action_day] = self._daily_orders.get(action_day, 0) + 1

    def _reassess_preferences(self) -> None:
        """重新评估每条偏好的满足状态，生成面向 LLM 的提示。

        不做确定性判断（偏好是自然语言），但计算具体指标供 LLM 决策。
        """
        current_day = self._sim_minutes // 1440
        total_days = current_day + 1
        today_minutes = self._sim_minutes % 1440
        hour = today_minutes // 60

        for ps in self._pref_statuses:
            text = ps.content
            hints: list[str] = []

            # ---- 每日休息类偏好：精准计算今日状态 ----
            if any(kw in text for kw in ["每天", "每日"]) and \
               any(kw in text for kw in ["休息", "睡觉", "熄火", "停车"]):
                # 今日最长连续休息
                intervals = self._daily_rest.get(current_day, [])
                if intervals:
                    merged = self._merge_intervals(intervals)
                    today_longest = max((e - s) for s, e in merged)
                    hints.append(f"今日最长连续休息={today_longest//60}h{today_longest%60}m")

                    # 尝试提取要求的小时数
                    import re
                    nums = re.findall(r'(\d+)\s*小时', text)
                    if nums:
                        required = int(nums[0]) * 60
                        deficit = required - today_longest
                        if deficit > 0:
                            hints.append(f"⚠️ 今日还差{deficit//60}h{deficit%60}m才能满足[{text[:30]}...]")
                            ps.status = "violated"
                        else:
                            hints.append(f"✅ 今日休息已达标")
                            ps.status = "ok"
                    else:
                        ps.status = "needs_attention"
                else:
                    hints.append("⚠️ 今日尚未休息！")
                    ps.status = "violated"

                # 月度统计（仅含已过去的整天）
                past_days_violations = 0
                for d in range(current_day):
                    d_intervals = self._daily_rest.get(d, [])
                    if d_intervals:
                        merged = self._merge_intervals(d_intervals)
                        longest = max((e - s) for s, e in merged)
                        import re
                        nums = re.findall(r'(\d+)\s*小时', text)
                        required = int(nums[0]) * 60 if nums else 480
                        if longest < required:
                            past_days_violations += 1
                if past_days_violations > 0:
                    hints.append(f"本月过往{past_days_violations}天未达标")

            # ---- 定时休息类偏好（如0-6点休息） ----
            elif any(kw in text for kw in ["点", "睡觉", "停着", "熄火"]) and \
                 any(kw in text for kw in ["到", "至", "～"]) and \
                 not any(kw in text for kw in ["每天", "每日"]):
                hints.append(f"当前时间={hour:02d}:{self._sim_minutes%1440%60:02d}")

                # 检查今日休息窗口内的休息情况
                import re
                window_nums = re.findall(r'(\d+)\s*点', text)
                if len(window_nums) >= 2:
                    ws, we = int(window_nums[0]), int(window_nums[1])
                    intervals = self._daily_rest.get(current_day, [])
                    window_rest = sum(
                        max(0, min(e, current_day*1440+we*60) - max(s, current_day*1440+ws*60))
                        for s, e in intervals
                    ) if intervals else 0
                    if window_rest > 0:
                        hints.append(f"今日{ws:02d}-{we:02d}窗口已休息{window_rest}min")
                        ps.status = "ok"
                    elif hour >= we:
                        hints.append(f"⚠️ 今日{ws:02d}-{we:02d}窗口已过，未休息！明天必须遵守")
                        ps.status = "violated"
                    elif ws <= hour < we:
                        hints.append(f"🔴 当前正在休息窗口({ws:02d}-{we:02d})内！必须 wait！")
                        ps.status = "violated"
                    else:
                        hints.append(f"今日休息窗口{ws:02d}-{we:02d}尚未开始")
                        ps.status = "needs_attention"

            # ---- 禁接品类 ----
            elif any(kw in text for kw in ["不接", "推掉", "一律不", "干不了", "不能接"]):
                categories = list(set(self._accepted_categories))
                if categories:
                    hints.append(f"已接品类: {', '.join(categories[-10:])}")
                    ps.status = "needs_attention"
                else:
                    ps.status = "ok"

            # ---- 区域限制 ----
            elif any(kw in text for kw in ["惠州", "深圳", "广州", "东莞", "佛山", "增城", "珠海", "汕头"]):
                regions = list(set(self._accepted_regions))
                if regions:
                    hints.append(f"已涉及区域: {', '.join(regions[-10:])}")
                    ps.status = "needs_attention"
                else:
                    ps.status = "ok"

            # ---- 月度天数要求 ----
            elif any(kw in text for kw in ["每月", "整月", "本月", "起码", "至少"]):
                active_days = sum(1 for v in self._daily_active_min.values() if v > 0)
                full_rest_days = sum(1 for d in range(current_day + 1) if self._daily_active_min.get(d, 0) == 0)
                hints.append(f"本月{total_days}天中，活跃{active_days}天，全休{full_rest_days}天")

                import re
                nums = re.findall(r'(\d+)\s*(?:天|个)', text)
                if nums:
                    target = int(nums[0])
                    if "不出车" in text or "歇着" in text or "停驶" in text or "完全" in text:
                        if full_rest_days >= target:
                            ps.status = "ok"
                            hints.append(f"✅ 已满足{target}天全休")
                        else:
                            ps.status = "violated"
                            hints.append(f"⚠️ 需要{target}天全休，已完成{full_rest_days}天")

            # ---- 特定日期 ----
            elif any(kw in text for kw in ["号", "日"]):
                hints.append(f"当前日期: 3月{current_day+1}日")
                ps.status = "needs_attention"

            # ---- 空驶距离 ----
            elif any(kw in text for kw in ["空驶", "公里", "km"]):
                hints.append(f"累计里程={self._total_distance:.0f}km")
                ps.status = "needs_attention"

            # ---- 默认 ----
            else:
                ps.status = "ok"

            ps.hints = hints

    def generate_proactive_hints(self) -> list[str]:
        """生成前瞻性建议——告诉 LLM 当前需要做什么来满足偏好。

        这些提示基于对偏好文本的通用分析（不硬编码具体规则），
        结合追踪到的状态给 LLM 策略性建议。
        """
        hints: list[str] = []
        current_day = self._sim_minutes // 1440
        hour = (self._sim_minutes % 1440) // 60
        total_days = current_day + 1
        remaining_days = 31 - current_day

        for ps in self._pref_statuses:
            text = ps.content

            # 休息类偏好：检查今天是否已满足
            if any(kw in text for kw in ["休息", "睡觉", "熄火", "停车"]):
                # 检查当前时间是否在常见休息窗口
                if hour >= 22 or hour < 6:
                    today_rest = self._daily_rest.get(current_day, [])
                    if not today_rest:
                        hints.append(f"💤 偏好P{ps.index+1}要求休息，当前为夜间({hour}点)，建议现在 wait。")

            # 天数类偏好：检查完成进度
            if any(kw in text for kw in ["每月", "整月", "本月", "起码", "至少"]):
                # 尝试从文本中提取数字
                import re
                numbers = re.findall(r'(\d+)\s*(?:天|个|次)', text)
                if numbers:
                    target = int(numbers[0])
                    active_days = sum(1 for v in self._daily_active_min.values() if v > 0)
                    if "不出车" in text or "歇着" in text or "停驶" in text or "完全" in text:
                        full_rest_days = sum(
                            1 for d in range(current_day + 1)
                            if self._daily_active_min.get(d, 0) == 0
                        )
                        if full_rest_days < target and remaining_days <= target - full_rest_days + 2:
                            hints.append(
                                f"⚠️ 偏好P{ps.index+1}：需{target}天完全休息，已完成{full_rest_days}天，"
                                f"仅剩{remaining_days}天！尽快安排全天休息。"
                            )

            # 日期特定偏好
            if any(kw in text for kw in ["号", "日"]):
                import re
                dates = re.findall(r'(\d+)\s*号', text)
                for d in dates:
                    target_day = int(d)
                    if target_day == current_day + 1:
                        hints.append(f"📅 偏好P{ps.index+1}：今天是3月{target_day}号，请检查是否有特殊要求！")
                    elif target_day == current_day + 2:
                        hints.append(f"📅 偏好P{ps.index+1}：明天是3月{target_day}号，提前规划路线。")

        return hints

    def record_order_details(
        self,
        cargo_name: str = "",
        start_city: str = "",
        end_city: str = "",
    ) -> None:
        """从外部（决策服务）记录接单的品类和城市信息。

        由于 tracker 只能看到 action 记录而看不到 cargo 详情，
        这个方法是决策服务在接单成功后调用来补充信息的。
        """
        if cargo_name:
            self._accepted_categories.append(cargo_name)
        if start_city:
            self._accepted_regions.append(start_city)
        if end_city:
            self._accepted_regions.append(end_city)

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """合并重叠区间。"""
        if not intervals:
            return []
        sorted_intervals = sorted(intervals)
        merged = [sorted_intervals[0]]
        for s, e in sorted_intervals[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))
        return merged
