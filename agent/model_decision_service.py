"""增强版模型决策服务。

基于经济学原理与学术研究成果：
  - 多点侦察（Hybrid mechanism）：不只查当前位置，也查附近物流枢纽
  - MDP 思维：每次接单后的状态转移（终点位置、时间推进）
  - 单位时间利润率 (PPM)：核心排序指标
  - 偏好关键字过滤：标记明确冲突但不硬编码规则
  - 带重试+fallback 的鲁棒 LLM 调用

依赖 `simkit.ports.SimulationApiPort`，由评测进程注入。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort

from .cargo_evaluator import (
    CargoEvaluator,
    CargoEvaluationResult,
    EvaluatedCargo,
    haversine_km,
    suggest_reposition_targets,
)
from .context_builder import build_decision_prompt, build_system_prompt
from .preference_tracker import PreferenceTracker


# ---- 广东省主要物流枢纽（用于侦察） ----
_SCOUT_HUBS: list[tuple[float, float, str]] = [
    (23.13, 113.26, "广州"),
    (22.54, 114.06, "深圳"),
    (23.02, 113.75, "东莞"),
    (22.84, 113.21, "佛山"),
    (22.56, 113.31, "中山"),
    (23.36, 116.68, "汕头"),
]

# 触发侦察的条件
_SCOUT_PPM_THRESHOLD = 2.0       # Top5 PPM 低于此值才侦察
_SCOUT_DENSITY_THRESHOLD = 0.3   # 货源密度低于此值才侦察
_SCOUT_MIN_DISTANCE_KM = 30.0    # 侦察点距当前位置至少多远


class ModelDecisionService:
    """增强版决策服务：多点侦察 + 经济分析 + 偏好追踪 + 智能 prompt。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")

        self._trackers: dict[str, PreferenceTracker] = {}

        # 可调参数
        self._max_cargo_display = 20       # 送入 LLM 的最大货源数
        self._query_cargo_k = 100          # 每次查询货源条数
        self._scout_k = 50                 # 侦察查询条数（比主查询少，省时间）
        self._max_scout_locations = 2      # 最多侦察几个枢纽
        self._max_retries = 2              # LLM 最大重试次数
        self._default_wait_minutes = 240   # fallback 休息时长

    # ================================================================
    # 主入口
    # ================================================================

    def decide(self, driver_id: str) -> dict[str, Any]:
        """单步决策主入口。

        流程：
        1. 获取状态 → 2. 主查询货源 → 3. 更新偏好追踪 →
        4. 经济评估 → 5. (条件)多点侦察 → 6. 构建 prompt →
        7. LLM 决策 → 8. 解析（失败时 fallback）
        """
        # 1. 获取司机状态
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        sim_minutes = int(status.get("simulation_progress_minutes", 0))
        cost_per_km = float(status.get("cost_per_km", 1.5))
        truck_length = str(status.get("truck_length", "4.2米"))
        preferences = status.get("preferences", []) or []

        self._logger.info(
            "decision: driver=%s day=%s time=%smin pos=(%.4f,%.4f)",
            driver_id,
            sim_minutes // 1440 + 1,
            sim_minutes,
            lat,
            lng,
        )

        # 2. 主查询：当前位置货源
        cargo_items = self._query_cargo_safe(driver_id, lat, lng, self._query_cargo_k)

        # 3. 更新偏好追踪器
        tracker = self._get_or_create_tracker(driver_id)
        tracker.set_preferences(preferences)
        hist_resp = self._api.query_decision_history(driver_id, -1)
        hist_records = hist_resp.get("records", []) if hist_resp else []
        tracker.update(
            simulation_minutes=sim_minutes,
            current_lat=lat,
            current_lng=lng,
            decision_history=hist_records,
        )
        snapshot = tracker.snapshot(cost_per_km=cost_per_km)

        # 4. 经济评估当前位置货源
        evaluator = CargoEvaluator(
            driver_lat=lat, driver_lng=lng,
            cost_per_km=cost_per_km, truck_length=truck_length,
        )
        cargo_result = evaluator.evaluate(cargo_items)

        # 5. 偏好关键字过滤（标记明确冲突）
        cargo_result = self._apply_preference_keyword_filter(cargo_result, preferences)

        self._logger.info(
            "eval: %s candidates, top5_ppm=¥%.2f, density=%.2f",
            len(cargo_result.evaluated),
            cargo_result.top5_profit_per_minute,
            cargo_result.area_cargo_density,
        )

        # 6. 条件侦察：当前货源差时，查附近枢纽
        scout_results: list[dict[str, Any]] = []
        if self._should_scout(cargo_result):
            scout_results = self._scout_hubs(
                driver_id, lat, lng, cost_per_km, truck_length,
            )
            self._logger.info("scout: queried %s hubs", len(scout_results))

        # 6.5 状态转移分析（Top-5 货源接单后的位置/时间）
        post_delivery = evaluator.analyze_post_delivery(
            cargo_result.evaluated,
            simulation_minutes=sim_minutes,
            top_n=5,
        )
        cargo_result.post_delivery_analysis = post_delivery

        # 7. 构建 prompt
        system_prompt = build_system_prompt(
            driver_preferences_text=self._format_preferences_for_prompt(preferences),
        )
        user_prompt = build_decision_prompt(
            snapshot=snapshot,
            cargo_result=cargo_result,
            scout_results=scout_results,
            cost_per_km=cost_per_km,
            max_cargo_display=self._max_cargo_display,
        )

        # 8. LLM 决策
        action = self._call_llm_with_retry(
            system_prompt, user_prompt,
            cargo_result, scout_results,
        )

        self._logger.info(
            "output: driver=%s action=%s params=%s",
            driver_id,
            action.get("action"),
            json.dumps(action.get("params", {}), ensure_ascii=False),
        )
        return action

    # ================================================================
    # 多点侦察
    # ================================================================

    def _should_scout(self, cargo_result: CargoEvaluationResult) -> bool:
        """判断是否需要侦察其他位置。"""
        if cargo_result.top5_profit_per_minute >= _SCOUT_PPM_THRESHOLD:
            return False
        if cargo_result.area_cargo_density >= _SCOUT_DENSITY_THRESHOLD:
            return False
        return len(cargo_result.evaluated) < 30

    def _scout_hubs(
        self,
        driver_id: str,
        lat: float, lng: float,
        cost_per_km: float,
        truck_length: str,
    ) -> list[dict[str, Any]]:
        """在附近物流枢纽查询货源，比较各位置的货源质量。"""
        # 选最近的几个枢纽
        hubs_with_dist = []
        for hub_lat, hub_lng, name in _SCOUT_HUBS:
            d = haversine_km(lat, lng, hub_lat, hub_lng)
            if d > _SCOUT_MIN_DISTANCE_KM:
                hubs_with_dist.append((d, hub_lat, hub_lng, name))
        hubs_with_dist.sort()
        selected = hubs_with_dist[:self._max_scout_locations]

        results: list[dict[str, Any]] = []
        for dist, hub_lat, hub_lng, name in selected:
            items = self._query_cargo_safe(driver_id, hub_lat, hub_lng, self._scout_k)
            evaluator = CargoEvaluator(
                driver_lat=hub_lat, driver_lng=hub_lng,
                cost_per_km=cost_per_km, truck_length=truck_length,
            )
            hub_result = evaluator.evaluate(items)

            # 移动到该枢纽的空驶成本
            repos_time_min = max(1, int(dist / 60.0 * 60 + 0.999))  # ceil
            repos_cost = dist * cost_per_km

            results.append({
                "hub_name": name,
                "latitude": round(hub_lat, 4),
                "longitude": round(hub_lng, 4),
                "distance_km": round(dist, 1),
                "repos_time_min": repos_time_min,
                "repos_cost": round(repos_cost, 2),
                "cargo_count": len(hub_result.evaluated),
                "top5_ppm": round(hub_result.top5_profit_per_minute, 2),
                "avg_ppm": round(hub_result.avg_profit_per_minute, 2),
                "density": round(hub_result.area_cargo_density, 2),
                "top_cargo_ids": [e.cargo_id for e in hub_result.evaluated[:5]],
            })
        return results

    # ================================================================
    # 偏好关键字过滤
    # ================================================================

    def _apply_preference_keyword_filter(
        self,
        cargo_result: CargoEvaluationResult,
        preferences: list[dict[str, Any]],
    ) -> CargoEvaluationResult:
        """基于偏好文本中的关键字，标记明确冲突的货源。

        注意：这不做确定性判断，仅提高风险评分以提醒 LLM。
        实际偏好判断由 LLM 基于原文完成。
        """
        # 从偏好原文中提取禁用关键词
        forbidden_categories: set[str] = set()
        forbidden_regions: set[str] = set()

        for pref in preferences:
            text = str(pref.get("content", ""))

            # 品类禁用模式: "XX类" "XX货源" "凡是XX"
            import re
            cat_patterns = re.findall(r'(?:凡是|禁接|不接|一律不|推掉|干不了)\s*[\w一-鿿]+', text)
            for p in cat_patterns:
                # 提取品类名
                cat = p.replace("凡是", "").replace("禁接", "").replace("不接", "").replace("一律不", "").replace("推掉", "").replace("干不了", "").strip()
                if cat:
                    forbidden_categories.add(cat)

            # 地区禁用模式: "XX的货" "不往XX" "不进XX" "XX的"
            region_patterns = re.findall(r'(?:惠州|深圳|广州|东莞|佛山|珠海|汕头|中山|江门|肇庆|湛江|茂名|阳江|清远|韶关|河源|梅州|潮州|揭阳|汕尾|云浮)', text)
            for r in region_patterns:
                # 判断是否为禁止语境
                if any(neg in text for neg in ["不往", "不进", "不接", "不跑", "禁止", "别给我"]):
                    forbidden_regions.add(r)

        # 对每条货源提升风险评分
        for ev in cargo_result.evaluated:
            cargo_name = ev.cargo_name
            cargo = ev.raw
            start_city = str(cargo.get("start", {}).get("city", ""))
            end_city = str(cargo.get("end", {}).get("city", ""))

            # 品类匹配
            for cat in forbidden_categories:
                if cat in cargo_name:
                    ev.preference_risk_score = min(1.0, ev.preference_risk_score + 0.5)
                    ev.preference_risk_reasons.append(f"⚠️ 偏好禁止品类「{cat}」")

            # 地区匹配
            for region in forbidden_regions:
                if region in start_city or region in end_city:
                    ev.preference_risk_score = min(1.0, ev.preference_risk_score + 0.5)
                    ev.preference_risk_reasons.append(f"⚠️ 偏好禁止区域「{region}」")

            # 重新计算综合评分（风险提高后评分降低）
            if ev.preference_risk_score > 0.3:
                ev.composite_score = max(0, ev.composite_score * (1.0 - ev.preference_risk_score))

        # 重新排序
        cargo_result.evaluated.sort(key=lambda e: e.composite_score, reverse=True)
        return cargo_result

    # ================================================================
    # LLM 调用与解析
    # ================================================================

    def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        cargo_result: CargoEvaluationResult,
        scout_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """调用 LLM 并在失败时重试。"""
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                model_resp = self._api.model_chat_completion({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                })
                return self._parse_action(model_resp, cargo_result, scout_results)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                self._logger.warning(
                    "LLM attempt %s/%s failed: %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
                if attempt < self._max_retries:
                    user_prompt = "上一步解析失败，请严格按JSON格式输出。\n" + user_prompt[-2000:]

        self._logger.error("All LLM attempts failed: %s", last_error)
        return self._fallback_action(cargo_result, scout_results)

    def _parse_action(
        self,
        model_resp: dict[str, Any],
        cargo_result: CargoEvaluationResult,
        scout_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """解析 LLM 返回的 JSON 动作为标准化格式。"""
        choices = model_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型返回缺少 choices")

        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回 content 为空")

        # 清洗 markdown 包裹
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        action = json.loads(content)
        if not isinstance(action, dict):
            raise ValueError("动作不是 JSON 对象")

        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params 必须是对象")
        if action_name not in {"take_order", "reposition", "wait"}:
            raise ValueError(f"未知 action: {action_name}")

        # take_order
        if action_name == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                raise ValueError("take_order 缺少 cargo_id")
            # 合并所有已知货源 ID（主查询 + 侦察）
            valid_ids = {ev.cargo_id for ev in cargo_result.evaluated}
            for sr in scout_results:
                valid_ids.update(sr.get("top_cargo_ids", []))
            if cargo_id not in valid_ids:
                self._logger.warning("cargo_id=%s not in known list, attempting anyway", cargo_id)
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}

        # reposition
        if action_name == "reposition":
            latitude = float(params["latitude"])
            longitude = float(params["longitude"])
            if not (18.0 <= latitude <= 28.0):
                raise ValueError(f"latitude {latitude} 超出范围")
            if not (108.0 <= longitude <= 118.0):
                raise ValueError(f"longitude {longitude} 超出范围")
            return {"action": "reposition", "params": {"latitude": latitude, "longitude": longitude}}

        # wait
        duration = int(params.get("duration_minutes", self._default_wait_minutes))
        if duration <= 0:
            raise ValueError("wait.duration_minutes 必须为正整数")
        if duration > 720:
            self._logger.warning("wait %s min > 12h, capped to 480", duration)
            duration = 480
        return {"action": "wait", "params": {"duration_minutes": duration}}

    # ================================================================
    # Fallback 策略
    # ================================================================

    def _fallback_action(
        self,
        cargo_result: CargoEvaluationResult,
        scout_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """LLM 全部失败时的规则兜底。"""
        evaluated = cargo_result.evaluated

        # 1. 有高 PPM 低风险货源 → 接单
        good = [e for e in evaluated
                if e.profit_per_minute > 1.5 and e.preference_risk_score < 0.3]
        if good:
            best = good[0]
            self._logger.info("fallback: take good cargo %s", best.cargo_id)
            return {"action": "take_order", "params": {"cargo_id": best.cargo_id}}

        # 2. 有尚可货源 → 接最好的
        ok = [e for e in evaluated
              if e.profit_per_minute > 0.5 and e.preference_risk_score < 0.5]
        if ok:
            self._logger.info("fallback: take ok cargo %s", ok[0].cargo_id)
            return {"action": "take_order", "params": {"cargo_id": ok[0].cargo_id}}

        # 3. 侦察结果显示某枢纽货源更好 → 空驶过去
        if scout_results:
            best_scout = max(scout_results, key=lambda s: s["top5_ppm"])
            if best_scout["top5_ppm"] > 1.0:
                self._logger.info(
                    "fallback: reposition to %s (PPM=%.2f)",
                    best_scout["hub_name"], best_scout["top5_ppm"],
                )
                return {
                    "action": "reposition",
                    "params": {
                        "latitude": best_scout["latitude"],
                        "longitude": best_scout["longitude"],
                    },
                }

        # 4. 都没货 → 休息
        self._logger.info("fallback: wait %s min", self._default_wait_minutes)
        return {"action": "wait", "params": {"duration_minutes": self._default_wait_minutes}}

    # ================================================================
    # 辅助方法
    # ================================================================

    def _query_cargo_safe(
        self, driver_id: str, lat: float, lng: float, k: int,
    ) -> list[dict[str, Any]]:
        """安全查询货源（捕获异常）。"""
        try:
            resp = self._api.query_cargo(
                driver_id=driver_id, latitude=lat, longitude=lng, k=k,
            )
            return resp.get("items", [])
        except Exception as exc:
            self._logger.warning("query_cargo failed at (%.4f,%.4f): %s", lat, lng, exc)
            return []

    def _get_or_create_tracker(self, driver_id: str) -> PreferenceTracker:
        if driver_id not in self._trackers:
            self._trackers[driver_id] = PreferenceTracker(driver_id)
        return self._trackers[driver_id]

    @staticmethod
    def _format_preferences_for_prompt(preferences: list[dict[str, Any]]) -> str:
        if not preferences:
            return ""
        lines = []
        for i, pref in enumerate(preferences, 1):
            content = str(pref.get("content", ""))
            penalty = float(pref.get("penalty_amount", 0) or 0)
            cap = pref.get("penalty_cap")
            cap_str = f"上限¥{cap:,.0f}" if cap is not None else "无上限"
            lines.append(f"{i}. {content}  [扣¥{penalty:,.0f}/次, {cap_str}]")
        return "\n".join(lines)
