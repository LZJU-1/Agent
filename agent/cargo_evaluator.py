"""货源经济评估器：基于经济学原理对候选货源进行筛选、评分与排序。

核心经济学概念：
  - 单位时间利润率 (profit_per_minute)：衡量时间配置效率的核心指标
  - 机会成本：接单耗用的时间本可用于其他货源或策略性等待
  -  reservation price：低于最低可接受 PPM 的货源不值得接
  - 空间套利：空驶成本 vs 目的地货源密度预期
  - 偏好兼容性：违反偏好的罚分应视为负收益纳入决策
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------
# Haversine 距离（与 simkit 一致）
# ---------------------------------------------------------------

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * radius_km * math.asin(math.sqrt(h))


def distance_to_minutes(distance_km: float, speed_km_per_hour: float = 60.0) -> int:
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


# ---------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------

@dataclass
class EvaluatedCargo:
    """一条货源的完整经济评估结果。"""
    # 原始字段
    cargo_id: str
    cargo_name: str = ""
    price_yuan: float = 0.0         # 价格（元）
    # 距离
    pickup_distance_km: float = 0.0  # 空驶到装货地（km）
    haul_distance_km: float = 0.0    # 干线运输里程（km）
    total_distance_km: float = 0.0   # 总里程
    # 时间（分钟）
    pickup_time_min: int = 0
    wait_time_min: int = 0           # 装货窗等待
    haul_time_min: int = 0
    total_time_min: int = 0
    # 经济指标
    pickup_cost: float = 0.0
    haul_cost: float = 0.0
    total_cost: float = 0.0
    net_profit: float = 0.0
    profit_per_minute: float = 0.0   # 核心 KPI：每分钟净收益（元）
    profit_per_km: float = 0.0
    # 偏好风险
    preference_risk_score: float = 0.0  # 越高越危险（0=安全）
    preference_risk_reasons: list[str] = field(default_factory=list)
    # 定位价值
    end_position_value_score: float = 0.0  # 卸货地货源密度预期
    # 综合评分（0-100）
    composite_score: float = 0.0
    # 原始数据
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CargoEvaluationResult:
    """批量评估结果。"""
    evaluated: list[EvaluatedCargo]     # 按 composite_score 降序排列
    filtered_out_count: int = 0         # 被过滤掉的货源数
    area_cargo_density: float = 0.0     # 当前区域货源密度估计
    avg_profit_per_minute: float = 0.0  # 当前区域平均 PPM
    top5_profit_per_minute: float = 0.0 # Top5 平均 PPM
    # 状态转移分析（接单后的状态）
    post_delivery_analysis: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PostDeliveryState:
    """接单完成后的状态预估——帮助 LLM 做 MDP 式的前瞻决策。"""
    cargo_id: str
    end_lat: float                # 卸货地纬度
    end_lng: float                # 卸货地经度
    end_city: str                 # 卸货城市
    finish_minute: int            # 预计完成时刻（仿真分钟）
    finish_day: int               # 预计完成日期（1-31）
    finish_hour: int              # 预计完成小时（0-23）
    is_night_arrival: bool        # 是否夜间到达（21-6点）
    is_month_end: bool            # 是否月末到达（>25天）
    total_time_spent: int         # 从当前到卸货总耗时
    net_profit: float             # 净收益
    profit_per_minute: float      # PPM


# ---------------------------------------------------------------
# 货源经济评估引擎
# ---------------------------------------------------------------

# 广东省主要物流枢纽坐标（用于目的地价值评估）
_LOGISTICS_HUBS: list[tuple[float, float, str]] = [
    (23.13, 113.26, "广州"),
    (22.54, 114.06, "深圳"),
    (23.02, 113.75, "东莞"),
    (22.84, 113.21, "佛山"),
    (23.18, 113.50, "广州黄埔"),
    (22.87, 113.83, "深圳宝安"),
    (22.94, 114.09, "东莞清溪"),
    (23.09, 113.19, "佛山南海"),
    (22.56, 113.31, "中山"),
    (23.36, 116.68, "汕头"),
    (21.92, 112.02, "阳江"),
    (23.68, 115.10, "河源"),
]


class CargoEvaluator:
    """货源经济评估器。

    评估流程：
    1. 基础过滤（车长不匹配等）
    2. 经济指标计算（PPM、净收益等）
    3. 偏好风险评估
    4. 目的地定位价值评估
    5. 综合评分与排序
    """

    def __init__(
        self,
        driver_lat: float,
        driver_lng: float,
        cost_per_km: float,
        truck_length: str,
        cargo_view_batch_size: int = 10,
    ) -> None:
        self._driver_lat = driver_lat
        self._driver_lng = driver_lng
        self._cost_per_km = cost_per_km
        self._truck_length = truck_length
        self._cargo_view_batch_size = cargo_view_batch_size

    # ---------- 公开方法 ----------

    def evaluate(
        self,
        cargo_items: list[dict[str, Any]],
        simulation_minutes: int = 0,
    ) -> CargoEvaluationResult:
        """对一批候选货源执行完整经济评估。"""
        evaluated: list[EvaluatedCargo] = []
        filtered_out = 0

        for item in cargo_items:
            cargo = item.get("cargo", {})
            distance_km = float(item.get("distance_km", 0))

            # 1. 基础过滤
            if not self._passes_basic_filter(cargo):
                filtered_out += 1
                continue

            # 2. 经济计算（含装货窗等待时间）
            ev = self._compute_economics(cargo, distance_km, simulation_minutes)
            if ev.total_time_min <= 0:
                filtered_out += 1
                continue

            # 3. 偏好风险评估
            self._assess_preference_risk(ev)

            # 4. 目的地价值评估
            self._assess_end_position_value(ev)

            # 5. 综合评分
            ev.composite_score = self._compute_composite_score(ev)
            evaluated.append(ev)

        # 排序
        evaluated.sort(key=lambda e: e.composite_score, reverse=True)

        # 汇总统计
        density = self._estimate_area_density(cargo_items)
        pps = [e.profit_per_minute for e in evaluated if e.profit_per_minute > 0]
        avg_ppm = sum(pps) / len(pps) if pps else 0.0
        top5_ppm = sum(sorted(pps, reverse=True)[:5]) / min(5, len(pps)) if pps else 0.0

        return CargoEvaluationResult(
            evaluated=evaluated,
            filtered_out_count=filtered_out,
            area_cargo_density=density,
            avg_profit_per_minute=avg_ppm,
            top5_profit_per_minute=top5_ppm,
        )

    def analyze_post_delivery(
        self,
        evaluated_cargos: list[EvaluatedCargo],
        simulation_minutes: int,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """对 Top-N 货源做状态转移分析：计算接单后的位置和时间。

        这帮助 LLM 进行 MDP 式的前瞻决策——不仅看当前收益，
        还看接单后"你会出现在哪里，是什么时间"。
        """
        results: list[dict[str, Any]] = []
        month_end = 31 * 24 * 60  # 44640 min

        for ev in evaluated_cargos[:top_n]:
            cargo = ev.raw
            end = cargo.get("end", {})
            end_lat = float(end.get("lat", 0))
            end_lng = float(end.get("lng", 0))
            end_city = str(end.get("city", "未知"))

            finish_min = simulation_minutes + ev.total_time_min
            finish_day = finish_min // 1440 + 1
            finish_hour = (finish_min % 1440) // 60

            results.append({
                "cargo_id": ev.cargo_id,
                "end_lat": round(end_lat, 4),
                "end_lng": round(end_lng, 4),
                "end_city": end_city,
                "finish_minute": finish_min,
                "finish_day": finish_day,
                "finish_hour": finish_hour,
                "is_night_arrival": finish_hour >= 21 or finish_hour < 6,
                "is_month_end": finish_day > 25,
                "exceeds_month": finish_min > month_end,
                "total_time_spent": ev.total_time_min,
                "net_profit": ev.net_profit,
                "profit_per_minute": ev.profit_per_minute,
                "end_hub_distance_km": round(min(
                    haversine_km(end_lat, end_lng, hlat, hlng)
                    for hlat, hlng, _ in _LOGISTICS_HUBS
                ), 1),
            })
        return results

    def get_cargo_by_id(self, evaluated: list[EvaluatedCargo], cargo_id: str) -> EvaluatedCargo | None:
        """按 cargo_id 查找已评估的货源。"""
        for ev in evaluated:
            if ev.cargo_id == cargo_id:
                return ev
        return None

    # ---------- 内部方法 ----------

    def _passes_basic_filter(self, cargo: dict[str, Any]) -> bool:
        """基础过滤：车长匹配等。"""
        allowed_lengths = cargo.get("truck_length", [])
        if isinstance(allowed_lengths, list) and allowed_lengths:
            if self._truck_length not in allowed_lengths:
                return False
        return True

    def _compute_economics(
        self,
        cargo: dict[str, Any],
        distance_to_pickup: float,
        simulation_minutes: int = 0,
    ) -> EvaluatedCargo:
        """计算货源的经济指标。"""
        cargo_id = str(cargo.get("cargo_id", ""))
        cargo_name = str(cargo.get("cargo_name", ""))
        # 价格：原始数据单位为分，此处已是元（simkit 中 normalize 已除以100）
        price_yuan = float(cargo.get("price", 0))

        start = cargo.get("start", {})
        end = cargo.get("end", {})
        start_lat = float(start.get("lat", 0))
        start_lng = float(start.get("lng", 0))
        end_lat = float(end.get("lat", 0))
        end_lng = float(end.get("lng", 0))

        # 距离
        pickup_distance_km = distance_to_pickup
        haul_distance_km = haversine_km(start_lat, start_lng, end_lat, end_lng)
        total_distance_km = pickup_distance_km + haul_distance_km

        # 时间
        pickup_time_min = distance_to_minutes(pickup_distance_km)
        haul_time_min = int(cargo.get("cost_time_minutes", 0))
        wait_time_min = self._compute_wait_time(cargo, pickup_time_min, simulation_minutes)

        total_time_min = pickup_time_min + wait_time_min + haul_time_min

        # 成本（仅计算行驶成本，装/卸/干线是服务过程）
        pickup_cost = pickup_distance_km * self._cost_per_km
        haul_cost = haul_distance_km * self._cost_per_km
        total_cost = total_distance_km * self._cost_per_km

        # 收益
        net_profit = price_yuan - total_cost
        profit_per_minute = net_profit / total_time_min if total_time_min > 0 else 0.0
        profit_per_km = net_profit / total_distance_km if total_distance_km > 0 else 0.0

        return EvaluatedCargo(
            cargo_id=cargo_id,
            cargo_name=cargo_name,
            price_yuan=round(price_yuan, 2),
            pickup_distance_km=round(pickup_distance_km, 2),
            haul_distance_km=round(haul_distance_km, 2),
            total_distance_km=round(total_distance_km, 2),
            pickup_time_min=pickup_time_min,
            wait_time_min=wait_time_min,
            haul_time_min=haul_time_min,
            total_time_min=total_time_min,
            pickup_cost=round(pickup_cost, 2),
            haul_cost=round(haul_cost, 2),
            total_cost=round(total_cost, 2),
            net_profit=round(net_profit, 2),
            profit_per_minute=round(profit_per_minute, 4),
            profit_per_km=round(profit_per_km, 4),
            raw=cargo,
        )

    # 仿真纪元: 2026-03-01 00:00:00
    _SIM_EPOCH = __import__("datetime").datetime(2026, 3, 1, 0, 0, 0)

    @classmethod
    def _compute_wait_time(
        cls,
        cargo: dict[str, Any],
        pickup_time_min: int,
        simulation_minutes: int = 0,
    ) -> int:
        """计算装货窗等待时间（分钟）。

        仿真时间线：simulation_minutes=0 对应 2026-03-01 00:00:00。
        到达装货地时间 = simulation_minutes + pickup_time_min（忽略 query_scan）。
        等待时间 = max(0, 装货窗开始 - 到达时间)。
        若到达时间晚于装货窗结束，接单会失败，返回极大值。
        """
        load_time = cargo.get("load_time")
        if not isinstance(load_time, list) or len(load_time) != 2:
            return 0

        try:
            lt_start_str = str(load_time[0]).strip().replace(" ", "T")
            lt_end_str = str(load_time[1]).strip().replace(" ", "T")
            lt_start = cls._SIM_EPOCH.fromisoformat(lt_start_str)
            lt_end = cls._SIM_EPOCH.fromisoformat(lt_end_str)
        except (ValueError, TypeError):
            return 0

        lt_start_min = int((lt_start - cls._SIM_EPOCH).total_seconds() // 60)
        lt_end_min = int((lt_end - cls._SIM_EPOCH).total_seconds() // 60)

        if lt_end_min < lt_start_min:
            return 0  # 无效时间窗

        arrival_min = simulation_minutes + pickup_time_min

        if arrival_min > lt_end_min:
            # 到达时已过装货窗 — 接单必然失败
            return 999999

        return max(0, lt_start_min - arrival_min)

    def _assess_preference_risk(self, ev: EvaluatedCargo) -> None:
        """评估货源与已知偏好类型的潜在冲突风险。

        注意：此处仅为初步标记，不做确定性的偏好违规判断。
        实际的偏好判断由 LLM 结合偏好原文完成。
        """
        risk = 0.0
        reasons: list[str] = []

        cargo_name = ev.cargo_name
        cargo = ev.raw
        start = cargo.get("start", {})
        end = cargo.get("end", {})
        start_city = str(start.get("city", ""))
        end_city = str(end.get("city", ""))

        # 标记可能有偏好风险的货源类型（中性标记，由 LLM 最终判断）
        high_risk_categories = {"机械设备", "蔬菜", "鲜活水产品", "玉米"}
        if cargo_name in high_risk_categories:
            risk += 0.15
            reasons.append(f"品类[{cargo_name}]可能有偏好限制")

        # 长途单接单后耗时很长
        if ev.total_time_min > 1000:
            risk += 0.05
            reasons.append(f"超长单({ev.total_time_min}min)，可能影响日程安排")

        # 装货/卸货地涉及特定城市（D001 偏好中涉及惠州等）
        sensitive_cities = {"惠州", "深圳", "珠海", "汕头"}
        for city in sensitive_cities:
            if city in start_city or city in end_city:
                risk += 0.05
                reasons.append(f"涉及[{city}]，可能有地理偏好限制")

        # 空驶距离过长（D002 偏好：空驶 >55km 扣分）
        if ev.pickup_distance_km > 50:
            risk += 0.1
            reasons.append(f"空驶距离较长({ev.pickup_distance_km:.0f}km)")

        ev.preference_risk_score = min(1.0, risk)
        ev.preference_risk_reasons = reasons

    def _assess_end_position_value(self, ev: EvaluatedCargo) -> None:
        """评估卸货地的战略定位价值。

        卸货地靠近物流枢纽时，后续更容易找到好货源，具有正向的"定位期权价值"。
        """
        end = ev.raw.get("end", {})
        end_lat = float(end.get("lat", 0))
        end_lng = float(end.get("lng", 0))

        # 计算到最近物流枢纽的距离
        min_dist = min(
            haversine_km(end_lat, end_lng, hub_lat, hub_lng)
            for hub_lat, hub_lng, _ in _LOGISTICS_HUBS
        )
        # 越近越好：0km → 1.0, 100km → 0.0
        value = max(0.0, 1.0 - min_dist / 100.0)
        ev.end_position_value_score = round(value, 4)

    def _compute_composite_score(self, ev: EvaluatedCargo) -> float:
        """计算货源的综合评分（0-100）。

        评分维度与权重：
        - 单位时间利润率 (PPM)：50% — 核心经济效率指标
        - 净收益绝对值：20% — 大单有规模效应
        - 偏好兼容性：15% — 安全货源优先
        - 定位价值：10% — 后续机会的期权价值
        - 时效性：5% — 短期能完成的单灵活性更好
        """
        score = 0.0

        # 1. PPM 评分 (0-50)：以 ¥5/min 为满分基准
        ppm = ev.profit_per_minute
        ppm_score = min(50.0, (ppm / 5.0) * 50.0) if ppm > 0 else 0.0
        score += ppm_score

        # 2. 净收益评分 (0-20)：以 ¥2000 为满分基准
        net = ev.net_profit
        net_score = min(20.0, (net / 2000.0) * 20.0) if net > 0 else 0.0
        score += net_score

        # 3. 偏好兼容性 (0-15)：风险低的满分
        risk = ev.preference_risk_score
        risk_score = (1.0 - risk) * 15.0
        score += risk_score

        # 4. 定位价值 (0-10)
        score += ev.end_position_value_score * 10.0

        # 5. 时效性 (0-5)：耗时越短越好
        total_hours = ev.total_time_min / 60.0
        time_score = max(0.0, 5.0 - total_hours * 0.5)  # 10小时以上得0分
        score += min(5.0, time_score)

        return round(score, 2)

    @staticmethod
    def _estimate_area_density(cargo_items: list[dict[str, Any]]) -> float:
        """估算当前区域的货源密度（0-1）。"""
        n = len(cargo_items)
        if n == 0:
            return 0.0
        # 基于返回数量和距离分布估算
        distances = [float(item.get("distance_km", 0)) for item in cargo_items]
        avg_dist = sum(distances) / len(distances) if distances else 100.0
        # 距离近 + 数量多 = 密度高
        dist_factor = max(0.0, 1.0 - avg_dist / 200.0)
        count_factor = min(1.0, n / 100.0)
        return round(0.5 * dist_factor + 0.5 * count_factor, 4)


# ---------------------------------------------------------------
# 推荐热点位置
# ---------------------------------------------------------------

def suggest_reposition_targets(
    current_lat: float,
    current_lng: float,
    cargo_items: list[dict[str, Any]],
    max_suggestions: int = 5,
) -> list[dict[str, Any]]:
    """根据当前货源分布，推荐空驶目标位置。

    逻辑：分析查询到的货源起点分布，识别货源密集的聚类中心。
    同时结合广东省物流枢纽，给出空驶建议。
    """
    suggestions: list[dict[str, Any]] = []

    # 策略1：货源起点聚类中心
    if cargo_items:
        # 收集货源起点坐标
        start_points: list[tuple[float, float, float]] = []  # (lat, lng, price)
        for item in cargo_items[:200]:  # 分析前200条
            cargo = item.get("cargo", {})
            start = cargo.get("start", {})
            slat = float(start.get("lat", 0))
            slng = float(start.get("lng", 0))
            price = float(cargo.get("price", 0))
            if slat and slng:
                start_points.append((slat, slng, price))

        if start_points:
            # 简单网格聚类：将广东省划分为网格，找货源最密集的网格中心
            grid: dict[tuple[int, int], list[float]] = {}
            for slat, slng, price in start_points:
                # 0.5度 ≈ 55km 网格
                grid_lat = int(slat / 0.5)
                grid_lng = int(slng / 0.5)
                key = (grid_lat, grid_lng)
                if key not in grid:
                    grid[key] = []
                grid[key].append(price)

            # 按货源数量和平均价格排序
            grid_scores: list[tuple[float, float, int, float]] = []
            for (glat, glng), prices in grid.items():
                center_lat = (glat + 0.5) * 0.5
                center_lng = (glng + 0.5) * 0.5
                count = len(prices)
                avg_price = sum(prices) / count if count else 0
                dist = haversine_km(current_lat, current_lng, center_lat, center_lng)
                score = count * avg_price / (1 + dist / 50)
                grid_scores.append((score, center_lat, center_lng, count))

            grid_scores.sort(key=lambda x: x[0], reverse=True)
            for _, lat, lng, count in grid_scores[:max_suggestions]:
                suggestions.append({
                    "latitude": round(lat, 4),
                    "longitude": round(lng, 4),
                    "reason": f"货源密集区（约{count}条在架货源）",
                    "estimated_cargo_count": count,
                })

    # 策略2：补充物流枢纽
    hub_suggestions = []
    for hub_lat, hub_lng, name in _LOGISTICS_HUBS:
        dist = haversine_km(current_lat, current_lng, hub_lat, hub_lng)
        if dist > 10:  # 不推荐当前位置
            hub_suggestions.append({
                "latitude": round(hub_lat, 4),
                "longitude": round(hub_lng, 4),
                "reason": f"物流枢纽[{name}]",
                "estimated_cargo_count": 0,
                "distance_km": round(dist, 1),
            })

    hub_suggestions.sort(key=lambda h: h.get("distance_km", 999))
    # 合并结果，取 max_suggestions 个
    seen = set()
    for s in suggestions + hub_suggestions:
        key = (s["latitude"], s["longitude"])
        if key not in seen:
            seen.add(key)
            if len([x for x in seen]) > max_suggestions:
                break

    return (suggestions + hub_suggestions)[:max_suggestions]
