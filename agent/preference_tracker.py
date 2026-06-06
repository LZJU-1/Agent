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

        重要：这里不做确定性判断（因为偏好是自然语言的），
        而是计算相关指标，提供信息给 LLM 决策。
        """
        current_day = self._sim_minutes // 1440
        total_days = current_day + 1

        for ps in self._pref_statuses:
            text = ps.content
            hints: list[str] = []

            # ---- 通用指标计算（不依赖硬编码规则） ----

            # 休息相关指标
            if any(kw in text for kw in ["休息", "睡觉", "熄火", "停车", "停驶", "停着"]):
                # 统计本月每天最长连续休息
                for day in range(current_day + 1):
                    intervals = self._daily_rest.get(day, [])
                    if intervals:
                        merged = self._merge_intervals(intervals)
                        longest = max((e - s) for s, e in merged)
                        longest_h = longest / 60.0
                        hints.append(f"第{day+1}天最长连续休息={longest_h:.1f}小时")

                # 本月完全休息天数
                full_rest = sum(
                    1 for d in range(current_day + 1)
                    if self._daily_active_min.get(d, 0) == 0
                )
                hints.append(f"本月完全不出车天数={full_rest}")

            # 禁接品类相关
            if any(kw in text for kw in ["不接", "推掉", "一律不", "干不了", "不能接"]):
                categories = list(set(self._accepted_categories))
                if categories:
                    hints.append(f"已接品类: {', '.join(categories[-10:])}")

            # 区域限制相关
            if any(kw in text for kw in ["惠州", "深圳", "广州", "东莞", "佛山", "增城", "珠海", "汕头", "不往", "不进"]):
                regions = list(set(self._accepted_regions))
                if regions:
                    hints.append(f"已涉及区域: {', '.join(regions[-10:])}")

            # 每月天数要求
            if any(kw in text for kw in ["每月", "整月", "本月", "这个月", "三月", "起码", "至少", "不少于"]):
                hints.append(f"本月已过{total_days}天，活跃天数={sum(1 for v in self._daily_active_min.values() if v > 0)}")

            # 特定日期要求
            if any(kw in text for kw in ["号", "日"]):
                hints.append(f"当前日期: 3月{current_day+1}日")

            # 空驶距离限制
            if any(kw in text for kw in ["空驶", "公里", "km", "KM"]):
                hints.append(f"累计行驶里程={self._total_distance:.0f}km")

            # 夜间时间限制
            if any(kw in text for kw in ["零点", "晚上", "夜间", "早上", "凌晨", "点以后", "点到"]):
                h = self._sim_minutes % 1440 // 60
                hints.append(f"当前时间={h:02d}:{self._sim_minutes%1440%60:02d}")

            # ---- 状态判断 ----
            ps.hints = hints
            # 状态由 LLM 根据提示自行判断，这里只标记"需要关注"
            if hints:
                ps.status = "needs_attention"
            else:
                ps.status = "ok"

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
