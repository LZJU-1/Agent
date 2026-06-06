"""Agent 算法大赛 - 卡车司机连续找货决策智能体。

模块结构：
  - model_decision_service: 主决策服务入口（ModelDecisionService）
  - cargo_evaluator: 货源经济评估（PPM、机会成本、偏好风险）
  - preference_tracker: 偏好状态追踪（跨步骤合规监控）
  - context_builder: Prompt 构建（经济学框架 + 状态 + 货源 + 策略）

使用方式：
  from agent.model_decision_service import ModelDecisionService
  service = ModelDecisionService(api_port)
  action = service.decide(driver_id)
"""
