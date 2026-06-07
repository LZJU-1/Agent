"""偏好管理器 — 7 条铁律（代码硬约束）+ 1 条软约束（LLM 判断）。

铁律 (代码保证，绝不违反):
  1. 每日休息 → 宵禁兜底
  2. 禁运品类 → 过滤货源
  3. 禁止区域 → 过滤货源
  4. 特殊日期不进某地 → 当天过滤
  5. 月度全休 → 主动安排
  6. 特殊日期私事 → 当天全休
  7. 要求区域 → 追踪进度 + prompt 提醒

软约束 (LLM 自行判断):
  - 空驶 ≤55km → 罚 ¥120/次，利润够大可以超
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ================================================================
@dataclass
class DailyRest:
    text: str; penalty_per_day: float; penalty_cap: float | None
    rest_type: str = ""          # "continuous" | "window"
    required_hours: int = 8      # 连续休息需要的小时数
    window_start: int = 0        # 定时窗口开始 (小时)
    window_end: int = 6          # 定时窗口结束 (小时)

@dataclass
class ForbiddenCargo:
    text: str; penalty_per_order: float; penalty_cap: float | None
    categories: list[str] = field(default_factory=list)

@dataclass
class ForbiddenRegion:
    text: str; penalty_per_time: float; penalty_cap: float | None
    regions: list[str] = field(default_factory=list)

@dataclass
class RequiredRegion:
    text: str; penalty_one_time: float; penalty_cap: float | None
    city: str = ""; min_days: int = 0

@dataclass
class MonthlyRest:
    text: str; penalty_one_time: float; penalty_cap: float | None
    days_needed: int = 2

@dataclass
class SpecialDate:
    text: str; penalty_one_time: float; penalty_cap: float | None
    dates: list[int] = field(default_factory=list)
    date_type: str = ""          # "forbidden_region" | "full_rest" | "goto_place"
    regions: list[str] = field(default_factory=list)


# ================================================================
class PreferenceManager:

    def __init__(self):
        self.daily_rests: list[DailyRest] = []
        self.forbidden_cargos: list[ForbiddenCargo] = []
        self.forbidden_regions: list[ForbiddenRegion] = []
        self.required_regions: list[RequiredRegion] = []
        self.monthly_rests: list[MonthlyRest] = []
        self.special_dates: list[SpecialDate] = []
        self._raw_prefs: list[dict] = []  # 保留原始偏好文本用于提取未分类规则

        # 追踪
        self._day = 1
        self._full_rest_days = 0
        self._visited_regions: list[str] = []
        self._today_active = False
        self._today_rest_intervals: list[tuple[int, int]] = []

    # ================================================================
    # 解析
    # ================================================================

    def parse(self, preferences: list[dict]) -> None:
        self.daily_rests.clear()
        self.forbidden_cargos.clear()
        self.forbidden_regions.clear()
        self.required_regions.clear()
        self.monthly_rests.clear()
        self.special_dates.clear()
        self._raw_prefs = list(preferences)  # 保存原始偏好

        for pref in preferences:
            text = str(pref.get("content", ""))
            penalty = float(pref.get("penalty_amount", 0) or 0)
            cap = float(pref.get("penalty_cap", 0)) if pref.get("penalty_cap") is not None else None

            # 分类判断（顺序重要！specific → general）
            if self._match_daily_rest(text):
                self.daily_rests.append(self._parse_daily_rest(text, penalty, cap))
            elif self._match_special_date(text):
                self.special_dates.append(self._parse_special_date(text, penalty, cap))
            elif self._match_monthly_rest(text):
                self.monthly_rests.append(self._parse_monthly_rest(text, penalty, cap))
            elif self._match_forbidden_cargo(text):
                self.forbidden_cargos.append(self._parse_forbidden_cargo(text, penalty, cap))
            elif self._match_required_region(text):
                self.required_regions.append(self._parse_required_region(text, penalty, cap))
            elif self._match_forbidden_region(text):
                self.forbidden_regions.append(self._parse_forbidden_region(text, penalty, cap))

    # ================================================================
    # 追踪更新
    # ================================================================

    def update_tracking(self, day: int, full_rest_days: int, visited_regions: list[str],
                        today_active: bool, today_rest: list[tuple[int, int]]) -> None:
        self._day = day
        self._full_rest_days = full_rest_days
        self._visited_regions = list(visited_regions)
        self._today_active = today_active
        self._today_rest_intervals = list(today_rest)

    def record_cargo(self, cargo_name: str, start_city: str, end_city: str) -> None:
        if start_city:
            self._visited_regions.append(start_city)
        if end_city:
            self._visited_regions.append(end_city)

    # ================================================================
    # 铁律 1+5+6: 需要休息吗？
    # ================================================================

    def should_rest_now(self, hour: int, minute: int) -> dict | None:
        """返回 wait action 或 None。

        检查顺序：月度全休 → 特殊日期全休 → 每日休息。
        月度全休和特殊日期全休优先级最高，必须全天休息。
        """

        # --- 铁律 5: 月度全休 (最高优先级 — 必须在每日休息前检查) ---
        for r in self.monthly_rests:
            if self._full_rest_days < r.days_needed and not self._today_active:
                remaining = 31 - self._day + 1
                need_more = r.days_needed - self._full_rest_days

                # 紧急：剩余天数刚好够 → 今天必须休
                if remaining <= need_more:
                    rem_today = 24 * 60 - (hour * 60 + minute)
                    if rem_today > 60:
                        return {"action": "wait", "params": {"duration_minutes": max(240, rem_today)}}

                # 主动均匀分布：在固定日期全休
                interval = max(1, 31 // max(r.days_needed, 1))
                offset = interval // 2
                for i in range(r.days_needed):
                    scheduled_day = offset + i * interval + 1
                    if scheduled_day > 31:
                        scheduled_day = 31
                    if self._day == scheduled_day:
                        rem_today = 24 * 60 - (hour * 60 + minute)
                        if rem_today > 120:
                            return {"action": "wait", "params": {"duration_minutes": rem_today}}

        # --- 铁律 6: 特殊日期全休 ---
        for r in self.special_dates:
            if self._day in r.dates:
                if r.date_type == "full_rest":
                    rem_today = 24 * 60 - (hour * 60 + minute)
                    if rem_today > 60:
                        return {"action": "wait", "params": {"duration_minutes": rem_today}}

        # --- 铁律 1: 每日休息 ---
        for r in self.daily_rests:
            if r.rest_type == "continuous":
                today_min = hour * 60 + minute
                remaining = 24 * 60 - today_min
                required = r.required_hours * 60

                # 🔑 关键修复：检查今天已经休息了多少
                already_rested = self._longest_rest()
                still_need = max(0, required - already_rested)

                # 已经休息够了 → 不强制休息
                if already_rested >= required:
                    continue

                # 凌晨 → 如果还没休息够，休息到6点
                if hour < 6:
                    dur = (6 - hour) * 60 - minute
                    # 确保休息时间至少覆盖剩余需求
                    dur = max(dur, still_need)
                    if dur > 0:
                        return {"action": "wait", "params": {"duration_minutes": min(dur, 720)}}

                # 当天剩余时间不够完成所需休息 + 60min 缓冲 → 立即休息
                if remaining <= still_need + 60:
                    return {"action": "wait", "params": {"duration_minutes": max(still_need, remaining)}}

            elif r.rest_type == "window":
                ws, we = r.window_start, r.window_end
                in_win = (ws < we and ws <= hour < we) or (ws > we and (hour >= ws or hour < we))
                if in_win:
                    if we > ws: dur = (we - hour) * 60 - minute
                    else: dur = ((we + 24) - hour) * 60 - minute if hour >= ws else (we - hour) * 60 - minute
                    if dur > 0:
                        return {"action": "wait", "params": {"duration_minutes": dur}}
                # 窗口前 2h: 确保在窗口开始前进入休息
                til = (ws - hour) * 60 - minute
                if til <= 0: til += 24 * 60
                if 0 < til <= 120:
                    wl = (we - ws) * 60 if we > ws else (we + 24 - ws) * 60
                    return {"action": "wait", "params": {"duration_minutes": til + wl}}

        return None

    # ================================================================
    # 铁律 2+3+4: 过滤货源
    # ================================================================

    def filter_cargos(self, cargos: list[dict], today_minutes: int) -> list[dict]:
        """返回安全货源列表。

        过滤规则（按顺序）：
        1. 禁运品类 → 移除
        2. 禁入区域 → 移除
        3. 特殊日期禁入区域 → 移除
        4. 超时（无法在休息窗口前完成）→ 移除
        5. 空驶距离超限（软约束）→ 先尝试过滤，若全过滤则放松
        """
        safe = list(cargos)

        # 铁律 2: 禁运品类
        for r in self.forbidden_cargos:
            safe = [c for c in safe
                    if not any(cat in c.get("cargo_name", "") for cat in r.categories)]

        # 铁律 3: 禁止区域
        for r in self.forbidden_regions:
            safe = [c for c in safe
                    if not any(reg in c.get("start_city","") or reg in c.get("end_city","")
                              for reg in r.regions)]

        # 铁律 4: 特殊日期不进某地
        for r in self.special_dates:
            if self._day in r.dates and r.date_type == "forbidden_region":
                safe = [c for c in safe
                        if not any(reg in c.get("start_city","") or reg in c.get("end_city","")
                                  for reg in r.regions)]

        # 铁律 1/5/6: 时间截止（必须在休息窗口前完成，+10min for query_scan）
        deadline = self._cargo_deadline()
        safe = [c for c in safe
                if today_minutes + c.get("total_time_min", 0) + 10 <= deadline]

        # 空驶距离限制（软约束：先过滤，若全过滤则保留超限货源）
        deadhead_limit = self._get_deadhead_limit()
        if deadhead_limit > 0:
            within_limit = [c for c in safe
                           if c.get("pickup_distance_km", 0) <= deadhead_limit]
            if within_limit:
                safe = within_limit
            # else: 保留原 safe（超限也比没货接好）

        return safe

    # ================================================================
    # Prompt 上下文 (给 LLM 看)
    # ================================================================

    def get_prompt_context(self) -> str:
        lines = []

        # 每日休息
        for r in self.daily_rests:
            if r.rest_type == "continuous":
                longest = self._longest_rest()
                h, m = divmod(longest, 60)
                if longest >= r.required_hours * 60:
                    lines.append(f"✅ 今日已连续休息 {h}h{m}m (需要{r.required_hours}h)")
                else:
                    need = r.required_hours * 60 - longest
                    nh, nm = divmod(need, 60)
                    lines.append(f"⚠️ 今日还需连续休息 {nh}h{nm}m (已休{h}h{m}m, 需要{r.required_hours}h)")
            elif r.rest_type == "window":
                lines.append(f"⏰ 休息窗口 {r.window_start:02d}:00-{r.window_end:02d}:00 (系统保证)")

        # 禁运品类
        for r in self.forbidden_cargos:
            lines.append(f"🚫 禁运: {'、'.join(r.categories)} (系统已过滤)")

        # 禁止区域
        for r in self.forbidden_regions:
            lines.append(f"🚫 禁去: {'、'.join(r.regions)} (系统已过滤)")

        # 铁律 7: 要求区域
        for r in self.required_regions:
            done = sum(1 for v in self._visited_regions if r.city in v)
            if done < r.min_days:
                lines.append(f"🔴 需 {r.min_days} 天到 {r.city}，已完成 {done} 天！请主动找 {r.city} 的货源！")
            else:
                lines.append(f"✅ {r.city}: {done}/{r.min_days} 天")

        # 月度全休
        for r in self.monthly_rests:
            if self._full_rest_days < r.days_needed:
                lines.append(f"🔴 月度全休需要 {r.days_needed} 天，完成 {self._full_rest_days} 天 (剩{31-self._day+1}天)")
            else:
                lines.append(f"✅ 月度全休: {self._full_rest_days}/{r.days_needed} 天")

        # 特殊日期（含提前 3 天预警）
        for r in self.special_dates:
            for d in r.dates:
                days_until = d - self._day
                if days_until == 0:
                    if r.date_type == "forbidden_region":
                        lines.append(f"🔴 今天 3月{d}日！禁止进入/接单涉及: {'、'.join(r.regions)}")
                    elif r.date_type == "goto_place":
                        lines.append(f"🔴 今天 3月{d}日！必须前往: {'、'.join(r.regions)} 并在当地停留！请找目的地为这些地区的货源，或 reposition 过去后 wait。")
                    elif r.date_type == "route":
                        lines.append(f"🔴 今天 3月{d}日！必须按路线前往: {' → '.join(r.regions)}！请规划路线依次到达，在终点停留。")
                    else:
                        lines.append(f"🔴 今天 3月{d}日！全天私事，必须全休！{r.text[:80]}")
                elif days_until == 1:
                    if r.date_type == "goto_place":
                        lines.append(f"⚠️ 明天 3月{d}日！需要前往: {'、'.join(r.regions)}，今天提前靠近！")
                    elif r.date_type == "route":
                        lines.append(f"⚠️ 明天 3月{d}日！路线: {' → '.join(r.regions)}，今天务必靠近起点 {'、'.join(r.regions[:1])}！")
                    elif r.date_type == "forbidden_region":
                        lines.append(f"⚠️ 明天 3月{d}日！禁止进入: {'、'.join(r.regions)}，今天提前离开！")
                    else:
                        lines.append(f"⚠️ 明天 3月{d}日！全天私事，今天提前收工休息。")
                elif 2 <= days_until <= 3:
                    if r.date_type in ("goto_place", "route"):
                        lines.append(f"📅 {days_until}天后(3月{d}日)需要前往: {' → '.join(r.regions)}，提前规划路线靠近！")

        return "\n".join(lines) if lines else "(无特殊偏好)"

    # ================================================================
    # 中文数字转换
    # ================================================================

    _CN_NUM = {"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,
               "十一":11,"十二":12,"十三":13,"十四":14,"十五":15,"十六":16,"十七":17,"十八":18,
               "十九":19,"二十":20,"二十一":21,"二十二":22,"二十三":23,"二十四":24,"二十五":25,
               "二十六":26,"二十七":27,"二十八":28,"二十九":29,"三十":30,"三十一":31}

    @classmethod
    def _extract_dates(cls, text: str) -> list[int]:
        """提取日期: '三月四号'→[4], '三月十二号'→[12], '3月5日'→[5], '三月四号五号'→[4,5]"""
        dates = []
        # ASCII 数字: "3月5日"
        for m in re.finditer(r'(?:三月|3月)\s*(\d+)\s*[号日]', text):
            dates.append(int(m.group(1)))
        # 中文数字: "三月四号", "三月十二号"
        for m in re.finditer(r'三月\s*([一二三四五六七八九十廿卅]+|[十二][一二三四五六七八九]|[二]?十[一二三四五六七八九]?|三十一?)\s*[号日]', text):
            cn = m.group(1)
            if cn in cls._CN_NUM:
                dates.append(cls._CN_NUM[cn])
        # 连续日期简写: "三四号"→[3,4], "四号五号"→[4,5]
        for m in re.finditer(r'([一二三四五])\s*号\s*([一二三四五])\s*号', text):
            a, b = cls._CN_NUM.get(m.group(1)), cls._CN_NUM.get(m.group(2))
            if a and b: dates.extend([a, b])
        # 单字简写: "三四号"→[3,4]
        for m in re.finditer(r'(?<!\d)([一二三四五])([一二三四五])号(?!\d)', text):
            a, b = cls._CN_NUM.get(m.group(1)), cls._CN_NUM.get(m.group(2))
            if a and b: dates.extend([a, b])
        return sorted(set(dates))

    # ================================================================
    # 分类匹配
    # ================================================================

    @staticmethod
    def _match_daily_rest(text: str) -> bool:
        # 连续型: "每天休息X小时"
        if re.search(r'每天|每日', text) and re.search(r'休息|睡觉|熄火|停车', text):
            return True
        # 定时窗口型: "零点到早上六点睡觉" (没有"每天"但有时段+睡眠)
        if re.search(r'(?:零点|\d+点|凌晨).*(?:到|至|～).*(?:\d+点|早上|凌晨)', text) and \
           re.search(r'睡觉|休息|停着|熄火', text):
            return True
        return False

    @staticmethod
    def _match_special_date(text: str) -> bool:
        return bool(PreferenceManager._extract_dates(text))

    @staticmethod
    def _match_monthly_rest(text: str) -> bool:
        # "三月...三个整天" / "这月...两个整天"
        has_month = bool(re.search(r'整月|这个月|这月|三月|本月', text))
        has_rest = bool(re.search(r'整天|完全歇|不出车|停驶|不排活|完全歇着|歇着', text))
        return has_month and has_rest

    @staticmethod
    def _match_forbidden_cargo(text: str) -> bool:
        has_ban = bool(re.search(r'不接|不拉|推掉|干不了|不能接|一律不|凡是.*不|赔不起', text))
        has_goods = bool(re.search(r'货源|品类|类|蔬菜|机械|设备|水产|玉米|水果|化工|建材|金属|家具|食品', text))
        has_city = bool(re.search(r'惠州|深圳|增城|广州|东莞|佛山', text))
        return has_ban and has_goods and not has_city

    @staticmethod
    def _match_required_region(text: str) -> bool:
        return any(kw in text for kw in ["起码", "至少", "不少于", "接够", "得接够"])

    @staticmethod
    def _match_forbidden_region(text: str) -> bool:
        # "不进/不往/不跑 XX" 或 "XX的货我一律不接" (当禁止的是地区)
        if any(kw in text for kw in ["不进", "不往", "别给我派", "不跑"]):
            return bool(re.search(r'惠州|深圳|广州|东莞|佛山|珠海|汕头|增城', text))
        # "XX的货，我一律不接" (地区禁止)
        if re.search(r'的货.*一律不接|一律不接.*的货', text):
            return bool(re.search(r'惠州|深圳|广州|东莞|佛山|珠海|汕头|增城', text))
        return False

    # ================================================================
    # 解析
    # ================================================================

    @staticmethod
    def _parse_daily_rest(text: str, penalty: float, cap: float | None) -> DailyRest:
        r = DailyRest(text=text, penalty_per_day=penalty, penalty_cap=cap)
        # 定时窗口: "零点到早上六点" / "23点到4点"
        if re.search(r'(?:零点|\d+点|凌晨).*(?:到|至|～).*(?:\d+点|早上|凌晨)', text):
            r.rest_type = "window"
            nums = re.findall(r'(\d+)\s*点', text)
            if len(nums) >= 2:
                r.window_start, r.window_end = int(nums[0]), int(nums[1])
            elif '零点' in text or '0点' in text:
                r.window_start, r.window_end = 0, 6
        else:
            r.rest_type = "continuous"
            m = re.search(r'(\d+)\s*小时', text)
            if m: r.required_hours = int(m.group(1))
        return r

    @staticmethod
    def _parse_forbidden_cargo(text: str, penalty: float, cap: float | None) -> ForbiddenCargo:
        r = ForbiddenCargo(text=text, penalty_per_order=penalty, penalty_cap=cap)
        # 提取品类名: "机械设备" "蔬菜" 等
        cats = re.findall(r'(?:凡是|禁接|不接|不拉|一律不|推掉|干不了|不能接)\s*([\w一-鿿]{2,4})', text)
        # 清洗：去掉 "货源" "货" "类" 等后缀
        cleaned = []
        for c in cats:
            c = c.strip()
            for suffix in ['货源', '货', '类', '品']:
                if c.endswith(suffix) and len(c) > len(suffix):
                    c = c[:-len(suffix)]
            if len(c) >= 2:
                cleaned.append(c)
        r.categories = cleaned
        if not r.categories:
            for kw in ['机械设备', '蔬菜', '鲜活水产品', '玉米', '水果', '化工', '建材', '金属', '家具', '食品']:
                if kw in text:
                    r.categories.append(kw)
        return r

    @staticmethod
    def _parse_forbidden_region(text: str, penalty: float, cap: float | None) -> ForbiddenRegion:
        r = ForbiddenRegion(text=text, penalty_per_time=penalty, penalty_cap=cap)
        cities = re.findall(r'惠州|深圳|广州|东莞|佛山|珠海|汕头|中山|江门|增城|番禺|宝安|龙岗|四会', text)
        r.regions = list(set(cities))
        return r

    @classmethod
    def _parse_required_region(cls, text: str, penalty: float, cap: float | None) -> RequiredRegion:
        r = RequiredRegion(text=text, penalty_one_time=penalty, penalty_cap=cap)
        # 提取城市名，区级优先于市级 (增城 > 广州)
        districts = re.findall(r'增城|宝安|龙岗|番禺|顺德|南海|四会', text)
        cities = re.findall(r'惠州|深圳|广州|东莞|佛山|珠海|汕头|中山|江门', text)
        r.city = districts[0] if districts else (cities[0] if cities else "")

        m = re.search(r'(\d+)\s*(?:个|天)', text)
        if m:
            r.min_days = int(m.group(1))
        else:
            cn_m = re.search(r'([一二两三四五六七八九十]+)\s*[个天]', text)
            if cn_m:
                r.min_days = cls._CN_NUM.get(cn_m.group(1), 0)
        return r

    @classmethod
    def _parse_monthly_rest(cls, text: str, penalty: float, cap: float | None) -> MonthlyRest:
        r = MonthlyRest(text=text, penalty_one_time=penalty, penalty_cap=cap)
        # "三个整天", "两个整天" — 用正则精确匹配中文数字+量词
        m = re.search(r'(\d+)\s*(?:天|个)', text)
        if m:
            r.days_needed = int(m.group(1))
        else:
            cn_m = re.search(r'([一二两三四五六七八九十]+)\s*[个天]\s*(?:整[天日]|完全|停驶|不出|不排)', text)
            if cn_m:
                r.days_needed = cls._CN_NUM.get(cn_m.group(1), 2)
        return r

    @classmethod
    def _parse_special_date(cls, text: str, penalty: float, cap: float | None) -> SpecialDate:
        r = SpecialDate(text=text, penalty_one_time=penalty, penalty_cap=cap)
        r.dates = cls._extract_dates(text)

        # 判断日期类型（顺序重要！）
        if any(kw in text for kw in ["不进", "不往", "别给我派", "查车"]):
            r.date_type = "forbidden_region"
            r.regions = re.findall(r'惠州|深圳|广州|东莞|佛山|珠海|汕头|中山|江门|增城', text)
        elif any(kw in text for kw in ["盘库", "对清楚", "清库存"]):
            # 需要去某地停留办事 — 不是全休！
            r.date_type = "goto_place"
            r.regions = list(set(re.findall(r'增城|惠州|深圳|广州|东莞|佛山|珠海|汕头|中山|江门|四会|从化|花都', text)))
        elif any(kw in text for kw in ["做寿", "赴宴", "捎上"]):
            # 寿宴/赴宴 — 判断是多地点路线还是单地点
            cities = list(set(re.findall(r'增城|惠州|深圳|广州|东莞|佛山|珠海|汕头|中山|江门|四会|从化|花都', text)))
            if len(cities) >= 2:
                r.date_type = "route"
            else:
                r.date_type = "goto_place"
            r.regions = cities
        elif any(kw in text for kw in ["得到", "去到", "去一趟", "停一趟", "过去", "跑一趟"]):
            r.date_type = "goto_place"
            r.regions = list(set(re.findall(r'增城|惠州|深圳|广州|东莞|佛山|珠海|汕头|中山|江门|四会|从化|花都', text)))
        else:
            r.date_type = "full_rest"
        return r

    # ================================================================
    # 铁律 8: 禁止 reposition 到禁区
    # ================================================================

    def is_reposition_forbidden(
        self, target_lat: float, target_lng: float
    ) -> tuple[bool, str]:
        """检查是否可以空驶到目标位置。

        返回 (is_forbidden, reason)。
        检查：禁入区域 + 特殊日期禁入区域。
        """
        # 用简单的区域名匹配（基于经纬度范围粗略判断）
        # 广东省主要城市经纬度范围
        _CITY_BOUNDS = {
            "深圳": (22.42, 22.89, 113.74, 114.66),
            "广州": (22.55, 23.66, 112.95, 114.05),
            "东莞": (22.72, 23.25, 113.55, 114.22),
            "惠州": (22.40, 23.95, 113.85, 115.08),
            "佛山": (22.58, 23.35, 112.55, 113.45),
            "珠海": (21.88, 22.45, 113.05, 113.85),
            "增城": (23.05, 23.45, 113.50, 113.95),
            "四会": (23.20, 23.55, 112.40, 112.95),
            "汕头": (23.05, 23.65, 116.30, 117.10),
        }

        def _in_city(lat: float, lng: float, city: str) -> bool:
            b = _CITY_BOUNDS.get(city)
            if not b:
                return False
            return b[0] <= lat <= b[1] and b[2] <= lng <= b[3]

        # 检查永久禁入区域
        for r in self.forbidden_regions:
            for region in r.regions:
                if _in_city(target_lat, target_lng, region):
                    return True, f"禁止进入 {region}: {r.text[:60]}"

        # 检查特殊日期禁入区域
        for r in self.special_dates:
            if self._day in r.dates and r.date_type == "forbidden_region":
                for region in r.regions:
                    if _in_city(target_lat, target_lng, region):
                        return True, f"3月{self._day}日禁止进入 {region}: {r.text[:60]}"

        return False, ""

    # ================================================================
    # 内部
    # ================================================================

    def _get_deadhead_limit(self) -> float:
        """从偏好中提取空驶距离上限（km），无限制返回 0。"""
        # 搜索所有已解析规则 + 原始偏好文本
        all_texts = []
        for r in self.daily_rests:
            all_texts.append(r.text)
        for r in self.forbidden_cargos:
            all_texts.append(r.text)
        for r in self.forbidden_regions:
            all_texts.append(r.text)
        for r in self.required_regions:
            all_texts.append(r.text)
        for r in self.monthly_rests:
            all_texts.append(r.text)
        for r in self.special_dates:
            all_texts.append(r.text)
        for p in self._raw_prefs:
            all_texts.append(str(p.get("content", "")))

        for text in all_texts:
            # 匹配 "空驶超过X公里" — X 可以是 ASCII 数字或中文数字
            m = re.search(r'空驶\s*(?:超过|大于|>)\s*([\d]+|[一二三四五六七八九十百]+)\s*(?:公里|km)', text)
            if m:
                return self._parse_number(m.group(1))
            # 匹配 "不超过X公里"
            m = re.search(r'(?:不超过|不大于|≤)\s*([\d]+|[一二三四五六七八九十百]+)\s*(?:公里|km)', text)
            if m:
                return self._parse_number(m.group(1))
        return 0

    @staticmethod
    def _parse_number(s: str) -> float:
        """解析数字字符串，支持中文数字和 ASCII 数字。"""
        if s.isdigit():
            return float(s)
        # 中文数字
        cn_map = {"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,
                  "百":100}
        result = 0
        unit = 1
        for ch in reversed(s):
            if ch in cn_map:
                val = cn_map[ch]
                if val >= 10:
                    result += max(1, result) * (val - 1) if result == 0 else 0
                    unit = val
                else:
                    result += val * (unit if unit >= 10 else 1)
        # 简化的中文数字解析
        # "五十五" → 55
        if "十" in s:
            parts = s.split("十")
            tens = cn_map.get(parts[0], 1) if parts[0] else 1
            ones = cn_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
            return float(tens * 10 + ones)
        return float(result) if result > 0 else 0.0

    def _cargo_deadline(self) -> int:
        """动态货源截止时间：基于今天已休息的时长计算最晚可接单完成时间。"""
        for r in self.daily_rests:
            if r.rest_type == "continuous":
                already = self._longest_rest()
                required = r.required_hours * 60
                still_need = max(0, required - already)
                if still_need <= 0:
                    return 24 * 60  # 已经休息够了，全天都可以接单
                # 需要在当天完成 still_need 的休息 + 1h 缓冲
                return 24 * 60 - still_need - 60
            elif r.rest_type == "window":
                ws = r.window_start
                return (ws - 1) * 60 if ws >= 1 else 23 * 60
        return 24 * 60  # 无休息偏好，全天可接单

    def _longest_rest(self) -> int:
        if not self._today_rest_intervals:
            return 0
        sorted_int = sorted(self._today_rest_intervals)
        merged = [sorted_int[0]]
        for s, e in sorted_int[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return max((e - s) for s, e in merged)
