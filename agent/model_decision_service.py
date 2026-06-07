"""决策服务 v5 — PreferenceManager 驱动。

流程:
  1. 获取状态 → 解析偏好
  2. 硬约束检查 → 需休息? → 直接 wait
  3. 查询货源 → 过滤禁品/禁区/超时 → 只剩安全选项
  4. 无安全货源 → wait; 有 → LLM 选最优
"""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort

from .cargo_evaluator import CargoEvaluator
from .preference_manager import PreferenceManager
from .preference_tracker import PreferenceTracker


class ModelDecisionService:

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        self._trackers: dict[str, PreferenceTracker] = {}
        self._mgrs: dict[str, PreferenceManager] = {}
        self._max_cargo_display = 20
        self._query_cargo_k = 300  # 扩大查询范围，增加找到近期装货窗货源的概率

    # ================================================================
    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        sim_minutes = int(status.get("simulation_progress_minutes", 0))
        cost_per_km = float(status.get("cost_per_km", 1.5))
        truck_length = str(status.get("truck_length", "4.2米"))
        preferences = status.get("preferences", []) or []

        hour = (sim_minutes % 1440) // 60
        minute = sim_minutes % 1440 % 60
        day = sim_minutes // 1440 + 1

        # 1. 解析偏好
        mgr = self._get_mgr(driver_id)
        mgr.parse(preferences)

        # 2. 更新追踪
        tracker = self._get_tracker(driver_id)
        tracker.set_preferences(preferences)
        hist = self._api.query_decision_history(driver_id, -1)
        tracker.update(sim_minutes, lat, lng, hist.get("records", []))
        snap = tracker.snapshot(cost_per_km)

        today = day - 1
        mgr.update_tracking(
            day=day,
            full_rest_days=snap.total_full_rest_days,
            visited_regions=snap.accepted_cargo_regions,
            today_active=snap.daily_active_minutes.get(today, 0) > 0,
            today_rest=snap.daily_rest_intervals.get(today, []),
        )

        # 3. 铁律 1+5+6: 需要休息？
        rest = mgr.should_rest_now(hour, minute)
        if rest:
            self._logger.info("REST: %s d%d %02d:%02d → wait %dmin",
                            driver_id, day, hour, minute, rest["params"]["duration_minutes"])
            return rest

        # 3.5 铁律 9: 特殊日期主动导航 (goto_place / route)
        special = mgr.get_special_date_action(hour, minute, lat, lng)
        if special:
            self._logger.info("SPECIAL: %s d%d %02d:%02d → %s %s",
                            driver_id, day, hour, minute,
                            special["action"], special.get("params", {}))
            return special

        # 4. 查询 + 评估货源（传入仿真时间以正确计算装货窗等待）
        items = self._query(driver_id, lat, lng)
        ev = CargoEvaluator(driver_lat=lat, driver_lng=lng,
                            cost_per_km=cost_per_km, truck_length=truck_length)
        result = ev.evaluate(items, simulation_minutes=sim_minutes)

        # 4.5 如果有关键区域偏好未达标，额外查询该区域附近货源
        extra_items = self._query_required_regions(driver_id, mgr)
        if extra_items:
            extra_result = ev.evaluate(extra_items, simulation_minutes=sim_minutes)
            # 合并货源（去重），将额外货源插入结果前面
            seen_ids = {ec.cargo_id for ec in result.evaluated}
            for ec in extra_result.evaluated:
                if ec.cargo_id not in seen_ids:
                    result.evaluated.insert(0, ec)
                    seen_ids.add(ec.cargo_id)
            self._logger.info("REGION_SCOUT: %s found %d extra cargos near required regions",
                            driver_id, len(extra_result.evaluated))

        # 5. 铁律 2+3+4: 过滤货源
        simple = [_cargo_dict(ec, sim_minutes) for ec in result.evaluated]
        safe = mgr.filter_cargos(simple, sim_minutes % 1440)

        # 6. 无安全货源 → 休息一段时间后重试
        if not safe:
            # 如果今天有休息需求未满足，休息所需时长；否则休息 180min
            rest_dur = 180
            for r in mgr.daily_rests:
                if r.rest_type == "continuous":
                    already = mgr._longest_rest()
                    still_need = max(0, r.required_hours * 60 - already)
                    if still_need > 0:
                        rest_dur = still_need
                        break
            self._logger.info("NO_SAFE: %s d%d %02d:%02d → wait %dmin",
                            driver_id, day, hour, minute, rest_dur)
            return {"action": "wait", "params": {"duration_minutes": rest_dur}}

        # 7. LLM 选择
        action = self._llm_decide(driver_id, snap, safe, mgr, cost_per_km)

        # 7.5. 验证 reposition 目标不在禁区内
        if action["action"] == "reposition":
            target_lat = float(action["params"].get("latitude", 0))
            target_lng = float(action["params"].get("longitude", 0))
            forbidden, reason = mgr.is_reposition_forbidden(target_lat, target_lng)
            if forbidden:
                self._logger.warning("REPOS_FORBIDDEN: %s → %s, falling back to wait", driver_id, reason)
                # 回退到等待（休息到能安全行动的时间）
                remaining = 24 * 60 - (sim_minutes % 1440)
                if remaining < 60:
                    remaining = 60
                action = {"action": "wait", "params": {"duration_minutes": min(remaining, 240)}}

        # 8. 记录接单信息
        if action["action"] == "take_order":
            cid = action["params"]["cargo_id"]
            for c in safe:
                if c["cargo_id"] == cid:
                    mgr.record_cargo(c["cargo_name"], c["start_city"], c["end_city"])
                    break

        self._logger.info("DECIDE: %s → %s %s", driver_id,
                         action["action"], action.get("params", {}))
        return action

    # ================================================================
    def _llm_decide(self, did, snap, safe, mgr, cost_per_km):
        sys = _SYSTEM_PROMPT.replace("{context}", mgr.get_prompt_context())
        usr = _build_user(snap, safe)

        for attempt in range(3):
            try:
                resp = self._api.model_chat_completion({
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": usr},
                    ],
                    "response_format": {"type": "json_object"},
                })
                return self._parse(resp)
            except Exception as e:
                self._logger.warning("LLM retry %d: %s", attempt + 1, e)

        return {"action": "take_order", "params": {"cargo_id": safe[0]["cargo_id"]}}

    @staticmethod
    def _parse(resp):
        c = resp["choices"][0]["message"]["content"].strip()
        if c.startswith("```"): c = "\n".join(l for l in c.split("\n") if "```" not in l)
        a = json.loads(c)
        n = a["action"].strip().lower()
        p = a.get("params", {})
        if n == "take_order":
            return {"action": "take_order", "params": {"cargo_id": str(p["cargo_id"])}}
        if n == "reposition":
            return {"action": "reposition", "params": {"latitude": float(p["latitude"]), "longitude": float(p["longitude"])}}
        d = int(p.get("duration_minutes", 240))
        return {"action": "wait", "params": {"duration_minutes": max(60, min(d, 720))}}

    def _query(self, did, lat, lng):
        try:
            return self._api.query_cargo(driver_id=did, latitude=lat, longitude=lng, k=self._query_cargo_k).get("items", [])
        except Exception:
            return []

    def _get_tracker(self, did):
        if did not in self._trackers:
            self._trackers[did] = PreferenceTracker(did)
        return self._trackers[did]

    def _get_mgr(self, did):
        if did not in self._mgrs:
            self._mgrs[did] = PreferenceManager()
        return self._mgrs[did]

    # 关键城市坐标（用于针对性地查询货源）
    _CITY_COORDS = {
        "增城": (23.15, 113.67),
        "四会": (23.32, 112.83),
        "深圳": (22.54, 114.06),
        "广州": (23.13, 113.26),
        "东莞": (23.02, 113.75),
        "惠州": (23.09, 114.40),
        "佛山": (22.84, 113.21),
        "珠海": (22.27, 113.55),
        "汕头": (23.36, 116.68),
        "中山": (22.56, 113.31),
    }

    def _query_required_regions(self, driver_id: str, mgr: PreferenceManager) -> list[dict]:
        """对未达标的偏好区域做定向货源查询。"""
        extra: list[dict] = []
        for r in mgr.required_regions:
            if r.city and r.city in self._CITY_COORDS:
                # 检查是否已达标
                done = sum(1 for v in mgr._visited_regions if r.city in v)
                if done < r.min_days:
                    clat, clng = self._CITY_COORDS[r.city]
                    try:
                        result = self._api.query_cargo(
                            driver_id=driver_id, latitude=clat, longitude=clng, k=50
                        )
                        items = result.get("items", []) if result else []
                        extra.extend(items)
                    except Exception:
                        pass
        return extra


# ================================================================
_SYSTEM_PROMPT = """你是卡车货运调度员。系统已确保你的休息和偏好合规。

## 偏好状态
{context}

## 任务
从候选货源中选最好的接单。没有好货时 reposition 到物流枢纽（广州 23.13,113.26 / 深圳 22.54,114.06 / 东莞 23.02,113.75 / 佛山 22.84,113.21）。

## 原则
- 优先高 PPM（每分钟净收益）
- PPM 相近选耗时短的
- 注意装货窗

## 输出 (只输出 JSON)
{{"action":"take_order","params":{{"cargo_id":"X"}}}}
{{"action":"reposition","params":{{"latitude":lat,"longitude":lng}}}}"""


def _build_user(snap, safe):
    h, m = snap.simulation_hour, snap.simulation_minute
    lines = [
        f"3月{snap.simulation_day}日 {h:02d}:{m:02d} | "
        f"({snap.current_lat:.4f},{snap.current_lng:.4f}) | "
        f"已接{snap.completed_order_count}单 | 净利≈¥{snap.estimated_net_so_far:,.0f}",
        "",
        f"## 候选货源 ({len(safe)}条，已过滤不安全项)",
        "",
        "|#|ID|品类|净利¥|PPM|总耗时|空驶|干线|装货窗|",
        "|-|--|----|----:|---|------|---:|---:|------|",
    ]
    for i, c in enumerate(safe[:20], 1):
        lt = c.get("load_time", "-")
        if isinstance(lt, list) and len(lt) == 2:
            # 显示装货窗开始时间（简短格式）
            lt_str = str(lt[0])[-11:-3] if len(str(lt[0])) > 11 else str(lt[0])[-8:-3]
        else:
            lt_str = "即刻"
        ppm = c["profit_per_minute"]
        ppm_s = f"**{ppm:.2f}**" if ppm >= 2 else f"{ppm:.2f}"
        total_h = c["total_time_min"] / 60
        time_s = f"{total_h:.1f}h" if total_h >= 1 else f"{c['total_time_min']}min"
        lines.append(f"|{i}|{c['cargo_id']}|{c['cargo_name'][:6]}|{c['net_profit']:+.0f}|"
                     f"{ppm_s}|{time_s}|{c['pickup_distance_km']:.0f}km|"
                     f"{c['haul_distance_km']:.0f}km|{lt_str}|")
    lines.append("")
    lines.append("## 输出 | take_order 或 reposition 到物流枢纽")
    lines.append("总耗时含空驶+装货窗等待+干线运输，已确保在休息窗口前完成。")
    return "\n".join(lines)


def _cargo_dict(ev, simulation_minutes: int = 0) -> dict:
    cargo = ev.raw
    # 使用 evaluator 计算的精确总时间（含装货窗等待）
    total_time = ev.total_time_min
    # 如果 evaluator 已经计算了正确的总时间（含装货窗等待），直接使用
    # 否则回退到旧估算
    if total_time <= 0 or ev.wait_time_min > 0:
        # 已正确计算
        pass
    else:
        # 回退估算
        raw_cost = int(cargo.get("cost_time_minutes", 0) or 0)
        pickup_time = ev.pickup_time_min
        total_time = raw_cost + pickup_time

    return {
        "cargo_id": ev.cargo_id,
        "cargo_name": ev.cargo_name,
        "start_city": str(cargo.get("start", {}).get("city", "")),
        "end_city": str(cargo.get("end", {}).get("city", "")),
        "total_time_min": total_time,
        "pickup_time_min": ev.pickup_time_min,   # 空驶时间
        "haul_time_min": ev.haul_time_min,        # 干线运输时间
        "wait_time_min": ev.wait_time_min,        # 装货窗等待时间
        "net_profit": ev.net_profit,
        "profit_per_minute": ev.profit_per_minute,
        "pickup_distance_km": ev.pickup_distance_km,
        "haul_distance_km": ev.haul_distance_km,
        "load_time": cargo.get("load_time"),
        "preference_risk_score": ev.preference_risk_score,
    }
