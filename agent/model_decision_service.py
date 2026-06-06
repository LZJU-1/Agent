"""增强版模型决策服务。

在原有 Demo 基础上引入：
  - 货源经济评估（PPM、机会成本、偏好风险）
  - 偏好状态追踪（跨步骤合规监控）
  - 经济学决策框架注入 prompt
  - 货源预筛选与排序
  - 异常处理与重试机制
  - Token 预算感知

依赖 `simkit.ports.SimulationApiPort`，由评测进程注入。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort

from .cargo_evaluator import CargoEvaluator, CargoEvaluationResult
from .context_builder import build_decision_prompt, build_system_prompt
from .preference_tracker import PreferenceTracker


class ModelDecisionService:
    """增强版决策服务：经济分析 + 偏好追踪 + 智能 prompt。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")

        # 偏好追踪器（按 driver_id 隔离）
        self._trackers: dict[str, PreferenceTracker] = {}

        # 配置
        self._max_cargo_display = 20      # 送入 LLM 的最大货源数
        self._query_cargo_k = 100         # 每次查询货源条数
        self._max_retries = 2             # LLM 调用最大重试次数
        self._default_wait_minutes = 240  # 默认休息 4 小时
        self._reposition_speed = 60.0     # km/h

    # ---------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------

    def decide(self, driver_id: str) -> dict[str, Any]:
        """单步决策主入口。

        流程：
        1. 获取司机状态与偏好
        2. 查询候选货源
        3. 更新偏好追踪器
        4. 经济评估货源
        5. 构建 prompt 并调用 LLM
        6. 解析动作（失败时 fallback）
        """
        # 1. 获取状态
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        sim_minutes = int(status.get("simulation_progress_minutes", 0))
        cost_per_km = float(status.get("cost_per_km", 1.5))
        truck_length = str(status.get("truck_length", "4.2米"))
        preferences = status.get("preferences", []) or []

        self._logger.info(
            "decision start: driver=%s time=%smin pos=(%.4f,%.4f) day=%s",
            driver_id,
            sim_minutes,
            lat,
            lng,
            sim_minutes // 1440 + 1,
        )

        # 2. 查询候选货源
        cargo_resp = self._api.query_cargo(
            driver_id=driver_id, latitude=lat, longitude=lng, k=self._query_cargo_k
        )
        cargo_items = cargo_resp.get("items", [])
        self._logger.info("query_cargo returned %s items", len(cargo_items))

        # 3. 更新偏好追踪器
        tracker = self._get_or_create_tracker(driver_id)
        tracker.set_preferences(preferences)

        # 获取决策历史
        hist_resp = self._api.query_decision_history(driver_id, -1)
        hist_records = hist_resp.get("records", []) if hist_resp else []
        tracker.update(
            simulation_minutes=sim_minutes,
            current_lat=lat,
            current_lng=lng,
            decision_history=hist_records,
        )
        snapshot = tracker.snapshot(cost_per_km=cost_per_km)

        # 4. 经济评估货源
        evaluator = CargoEvaluator(
            driver_lat=lat,
            driver_lng=lng,
            cost_per_km=cost_per_km,
            truck_length=truck_length,
        )
        cargo_result = evaluator.evaluate(cargo_items)

        self._logger.info(
            "cargo evaluation: %s candidates, %s filtered, top PPM=¥%.2f/min",
            len(cargo_result.evaluated),
            cargo_result.filtered_out_count,
            cargo_result.top5_profit_per_minute,
        )

        # 5. 构建 prompt
        system_prompt = build_system_prompt(
            driver_preferences_text=self._format_preferences_for_prompt(preferences)
        )
        user_prompt = build_decision_prompt(
            snapshot=snapshot,
            cargo_result=cargo_result,
            max_cargo_display=self._max_cargo_display,
        )

        # 6. 调用 LLM（带重试）
        action = self._call_llm_with_retry(system_prompt, user_prompt, cargo_result)

        self._logger.info(
            "decision output: driver=%s action=%s params=%s",
            driver_id,
            action.get("action"),
            action.get("params"),
        )
        return action

    # ---------------------------------------------------------------
    # LLM 调用与解析
    # ---------------------------------------------------------------

    def _call_llm_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        cargo_result: CargoEvaluationResult,
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
                return self._parse_action(model_resp, cargo_result)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                self._logger.warning(
                    "LLM call attempt %s/%s failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                if attempt < self._max_retries:
                    # 简化 prompt 重试
                    user_prompt = (
                        "上一步解析失败，请严格按格式输出JSON。\n" + user_prompt[-2000:]
                    )

        # 全部失败 → fallback
        self._logger.error("All LLM attempts failed, using fallback. Last error: %s", last_error)
        return self._fallback_action(cargo_result)

    def _parse_action(
        self, model_resp: dict[str, Any], cargo_result: CargoEvaluationResult
    ) -> dict[str, Any]:
        """解析 LLM 返回的 JSON 动作为标准化格式。"""
        choices = model_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型返回缺少 choices")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回 content 为空")

        # 清洗可能的 markdown 包裹
        content = content.strip()
        if content.startswith("```"):
            # 移除 ```json ... ``` 包裹
            lines = content.split("\n")
            content = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        action = json.loads(content)
        if not isinstance(action, dict):
            raise ValueError("模型返回动作不是 JSON 对象")

        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params")

        if action_name not in {"take_order", "reposition", "wait"}:
            raise ValueError(f"未知 action: {action_name}")
        if not isinstance(params, dict):
            raise ValueError("params 必须是对象")

        # ---- take_order ----
        if action_name == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                raise ValueError("take_order 缺少 cargo_id")
            # 验证 cargo_id 在候选列表中
            valid_ids = {ev.cargo_id for ev in cargo_result.evaluated}
            if cargo_id not in valid_ids:
                self._logger.warning(
                    "cargo_id=%s not in evaluated list (may have been filtered), attempting anyway",
                    cargo_id,
                )
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}

        # ---- reposition ----
        if action_name == "reposition":
            latitude = float(params["latitude"])
            longitude = float(params["longitude"])
            # 合理性校验
            if not (18.0 <= latitude <= 28.0):
                raise ValueError(f"latitude {latitude} 超出广东省范围")
            if not (108.0 <= longitude <= 118.0):
                raise ValueError(f"longitude {longitude} 超出广东省范围")
            return {"action": "reposition", "params": {"latitude": latitude, "longitude": longitude}}

        # ---- wait ----
        duration_minutes = int(params.get("duration_minutes", self._default_wait_minutes))
        if duration_minutes <= 0:
            raise ValueError("wait.duration_minutes 必须为正整数")
        # 建议休息时间合理范围
        if duration_minutes > 1440:
            self._logger.warning("wait duration=%s min (>24h), capping to 720", duration_minutes)
            duration_minutes = 720
        return {"action": "wait", "params": {"duration_minutes": duration_minutes}}

    # ---------------------------------------------------------------
    # Fallback 策略
    # ---------------------------------------------------------------

    def _fallback_action(self, cargo_result: CargoEvaluationResult) -> dict[str, Any]:
        """当 LLM 调用全部失败时的兜底策略。

        不使用 LLM，基于简单规则做决策：
        1. 如果有高 PPM 货源（>¥3/min），接最好的那个
        2. 如果货源质量一般，接最好的
        3. 完全没有货源则休息 4 小时
        """
        evaluated = cargo_result.evaluated

        if evaluated:
            # 取综合评分最高的低风险货源
            best = evaluated[0]
            # 如果最好货源的 PPM > 1.0，接单
            if best.profit_per_minute > 1.0:
                self._logger.info("fallback: taking best cargo %s (PPM=%.2f)", best.cargo_id, best.profit_per_minute)
                return {"action": "take_order", "params": {"cargo_id": best.cargo_id}}

            # PPM 太低，找低风险且 PPM > 0 的
            safe = [e for e in evaluated if e.preference_risk_score < 0.2 and e.profit_per_minute > 0]
            if safe:
                best_safe = safe[0]
                self._logger.info("fallback: taking safe cargo %s", best_safe.cargo_id)
                return {"action": "take_order", "params": {"cargo_id": best_safe.cargo_id}}

        # 没货或货太差 → 休息
        self._logger.info("fallback: no good cargos, waiting %s min", self._default_wait_minutes)
        return {"action": "wait", "params": {"duration_minutes": self._default_wait_minutes}}

    # ---------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------

    def _get_or_create_tracker(self, driver_id: str) -> PreferenceTracker:
        """获取或创建司机的偏好追踪器。"""
        if driver_id not in self._trackers:
            self._trackers[driver_id] = PreferenceTracker(driver_id)
        return self._trackers[driver_id]

    @staticmethod
    def _format_preferences_for_prompt(preferences: list[dict[str, Any]]) -> str:
        """将偏好列表格式化为易读文本。"""
        if not preferences:
            return ""

        lines = []
        for i, pref in enumerate(preferences, 1):
            content = str(pref.get("content", ""))
            penalty = float(pref.get("penalty_amount", 0) or 0)
            cap = pref.get("penalty_cap")
            cap_str = f"最高累计 ¥{cap:,.0f}" if cap is not None else "无罚分上限"
            lines.append(f"{i}. {content}")
            lines.append(f"   → 违规扣 ¥{penalty:,.0f}/次，{cap_str}")
        return "\n".join(lines)
