from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.time_display import format_beijing_time, is_time_column

VALUE_LABELS = {
    None: "未知",
    True: "是",
    False: "否",
    "unknown": "未知",
    "UNKNOWN": "未知",
    "OK": "正常",
    "PASS": "通过",
    "FAIL": "失败",
    "RUNNING": "运行中",
    "READY": "就绪",
    "BLOCKED": "阻塞",
    "N/A": "不适用",
    "WARNING": "警告",
    "CRITICAL": "严重",
    "NOT_CONFIGURED": "未配置",
    "ALLOW": "允许",
    "SELL_ONLY": "仅卖出",
    "ABORT": "中止",
    "DEAD": "失效",
    "QUARANTINE": "隔离",
    "PAPER_READY": "纸面就绪",
    "LIVE_READY": "实盘就绪",
    "LIVE_SMALL_READY": "小仓实盘就绪",
    "KEEP_SHADOW": "继续影子观察",
    "RESEARCH_ONLY": "仅研究",
    "KILL": "淘汰",
    "fresh": "新鲜",
    "delayed": "延迟",
    "stale": "过期",
    "missing": "缺失",
}

COLUMN_LABELS = {
    "abnormal_symbols": "异常标的",
    "actual_bars": "实际 K 线数",
    "actual_rows": "真实成本行数",
    "mixed_rows": "混合真实+代理成本行数",
    "allowed_modes": "允许模式",
    "alpha_id": "因子编号",
    "alpha6_score": "alpha6 分数",
    "alpha6_side": "alpha6 方向",
    "as_of_date": "截至日期",
    "ask": "卖一",
    "avg_mae_bps": "平均 MAE(bps)",
    "avg_mfe_bps": "平均 MFE(bps)",
    "avg_net_bps": "平均净收益(bps)",
    "avg_volume": "平均成交量",
    "block_reason": "阻塞原因",
    "board_schema_version": "面板 schema 版本",
    "bucket_index": "分桶索引",
    "bundle_ts": "数据包时间",
    "channel": "频道",
    "close": "收盘价",
    "collector": "采集器",
    "command": "命令",
    "complete_sample_count": "完整样本数",
    "cost_day": "成本日期",
    "cost_bps": "成本(bps)",
    "cost_model_version": "成本模型版本",
    "cost_sensitivity": "成本敏感度",
    "cost_source": "成本来源",
    "cost_source_coverage": "成本来源覆盖率",
    "cost_source_mix": "成本来源组合",
    "count": "数量",
    "created_at": "创建时间",
    "current_position": "当前持仓",
    "current_weight": "当前权重",
    "candidate_event_rows": "候选事件行数",
    "candidate_id": "候选 ID",
    "candidate_name": "候选名称",
    "dataset": "数据集",
    "date": "日期",
    "day": "日期",
    "decision": "决策",
    "decision_reasons": "决策原因",
    "decision_ts": "决策时间",
    "downside_p25_bps": "下行 P25(bps)",
    "duplicate_bar_count": "重复 K 线数",
    "edge_cost_ratio": "边际/成本比",
    "eligible_before_filters": "过滤前可交易",
    "end_ts": "结束时间",
    "error_count": "错误次数",
    "expected_edge_bps": "预期边际(bps)",
    "expected_bars": "预期 K 线数",
    "exists": "是否存在",
    "f1": "f1",
    "f1_mom_5d": "f1 5日动量",
    "f2": "f2",
    "f2_mom_20d": "f2 20日动量",
    "f3": "f3",
    "f3_vol_adj_ret": "f3 波动调整收益",
    "f4": "f4",
    "f4_volume_expansion": "f4 放量",
    "f5": "f5",
    "f5_rsi_trend_confirm": "f5 RSI 趋势确认",
    "fallback_level": "回退级别",
    "feature_completeness": "特征完整率",
    "final_decision": "最终决策",
    "final_score": "最终分数",
    "freshness_seconds": "新鲜度(秒)",
    "freshness_status": "新鲜度状态",
    "gate_version": "门控版本",
    "global_default_rows": "全局默认成本行数",
    "hard_fallback_count": "硬回退行数",
    "hard_fallback_ratio": "硬回退比例",
    "gross_bps": "毛收益(bps)",
    "high": "最高价",
    "horizon_hours": "标签周期(小时)",
    "ingest_ts": "入湖时间",
    "key": "键",
    "lag": "延迟状态",
    "label_completeness": "标签完整率",
    "label_status": "标签状态",
    "label_ts": "标签时间",
    "latest_bundle_sha256": "最新数据包 SHA256",
    "latest_bundle_ts": "最新数据包时间",
    "latest_success_ts": "最近成功时间",
    "latest_trade_ts": "最近成交时间",
    "latest_timestamp": "最新数据时间",
    "latest_ts": "最新时间",
    "low": "最低价",
    "market_type": "市场类型",
    "max_drawdown_proxy": "最大回撤代理",
    "mae_bps": "MAE(bps)",
    "mean_abs_return": "平均绝对收益",
    "mean_reversion_score": "均值回归分数",
    "median_net_bps": "净收益中位数(bps)",
    "metrics": "指标",
    "mfe_bps": "MFE(bps)",
    "ml_score": "ML 分数",
    "missing_bars": "缺失 K 线数",
    "modified_at": "修改时间",
    "name": "名称",
    "net_bps_after_cost": "扣成本净收益(bps)",
    "next_action": "下一步动作",
    "oos_max_drawdown": "样本外最大回撤",
    "oos_sharpe": "样本外夏普",
    "open": "开盘价",
    "parquet_file_count": "Parquet 文件数",
    "passed": "是否通过",
    "paper_days": "纸面观察天数",
    "path": "路径",
    "p25_net_bps": "净收益 P25(bps)",
    "permission": "权限",
    "protect_level": "保护等级",
    "proxy_rows": "代理成本行数",
    "soft_fallback_count": "软回退行数",
    "soft_fallback_ratio": "软回退比例",
    "purpose": "用途",
    "question": "问题",
    "rate_limit_warnings": "限频警告数",
    "rank": "排名",
    "reasons": "原因",
    "received_at": "接收时间",
    "reconnect_count": "重连次数",
    "regime": "状态",
    "regime_breakdown": "状态拆分",
    "regime_state": "状态",
    "required_edge_bps": "要求边际(bps)",
    "risk_level": "风险等级",
    "rows": "行数",
    "run_id": "运行 ID",
    "sample_count": "样本数",
    "size_bytes": "大小（字节）",
    "size_sum": "成交量合计",
    "source": "来源",
    "spread_bps": "价差 bps",
    "stability_by_day": "日稳定性",
    "start_ts": "开始时间",
    "status": "状态",
    "strategy": "策略",
    "strategy_candidate": "策略候选",
    "symbol_breakdown": "标的拆分",
    "success_count": "成功次数",
    "symbol": "标的",
    "target_weight_after_risk": "风控后目标权重",
    "target_weight_raw": "原始目标权重",
    "timeframe": "周期",
    "timestamp_column": "时间列",
    "trade_count": "成交笔数",
    "ts": "时间",
    "ts_utc": "UTC 时间",
    "value": "值",
    "venue": "交易所",
    "version": "版本",
    "violation": "违规项",
    "volatility_regime": "波动状态",
    "volume": "成交量",
    "warning": "告警",
    "win": "胜负",
    "win_rate": "胜率",
}

VALUE_LOCALIZED_COLUMNS = {
    "decision",
    "eligible_before_filters",
    "exists",
    "fallback_ratio_status",
    "final_decision",
    "freshness_status",
    "label_status",
    "passed",
    "permission",
    "status",
    "win",
}


# Additional display translations for newer strategy-advisory and entry-quality surfaces.
VALUE_LABELS.update(
    {
        "paper": "纸面观察",
        "shadow": "影子观察",
        "research": "研究观察",
        "none": "不推荐",
        "audit": "审计",
        "advisory": "建议",
        "shadow_only": "仅影子观察",
        "not_live_validated": "尚未验证可实盘",
        "entry_quality_advisory_only": "仅入场质量建议",
        "not_paper_candidate": "不是纸面候选",
        "entry_quality_research": "入场质量研究",
        "public_spread_proxy": "公共盘口价差代理",
        "mixed_actual_proxy": "真实费用 + 价差代理",
        "actual_fills": "真实成交成本",
        "global_default": "全局默认成本",
        "quant_lab": "中台",
        "degraded": "降级",
        "actual_or_mixed": "真实或混合成本",
        "v5.entry_quality_missed_low_audit": "V5 错失低点审计",
        "v5.late_entry_chase_guard_shadow": "V5 追高保护影子观察",
        "v5.pullback_reversal_shadow_btc": "BTC 回调反转影子观察",
        "v5.pullback_reversal_shadow_sol": "SOL 回调反转影子观察",
        "v5.pullback_reversal_shadow_eth": "ETH 回调反转影子观察",
        "v5.pullback_reversal_shadow_bnb": "BNB 回调反转影子观察",
        "v5.sol_protect_alpha6_low_exception": "SOL 保护态 Alpha6 低分例外",
        "v5.f4_volume_expansion_entry": "F4 放量入场",
        "v5.alt_impulse_shadow": "Alt 脉冲影子观察",
        "v5.multi_position_k1": "多仓位 K1",
        "v5.multi_position_k2": "多仓位 K2",
        "v5.multi_position_k3": "多仓位 K3",
        "v5.f3_dominant_entry": "F3 主导入场",
        "PAPER_READY": "纸面就绪",
        "KEEP_SHADOW": "继续影子观察",
        "REGIME_SHADOW": "分状态影子观察",
        "RESEARCH_ONLY": "仅研究",
        "RESEARCH": "研究中",
        "DISCOVERED": "已发现",
        "SHADOW_ONLY": "仅影子观察",
        "READY_FOR_PAPER": "纸面观察就绪",
        "KILL": "淘汰",
        "research_only": "仅研究",
        "paper_shadow": "纸面/影子",
        "live_command": "实盘命令",
        "quant_lab_live_command_not_allowed": "中台不允许实盘命令",
        "v5_local_live_not_controlled_by_quant_lab": "V5 本地实盘不由中台控制",
        "quant_lab_advisory_permission_not_allow": "中台建议未放开实盘",
    }
)

COLUMN_LABELS.update(
    {
        "as_of_ts": "截至时间",
        "strategy_id": "策略编号",
        "v5_symbol": "V5 标的",
        "recommended_mode": "推荐模式",
        "advisory_intent": "建议意图",
        "cost_quality": "成本质量",
        "live_block_reasons": "实盘阻断原因",
        "max_paper_notional_usdt": "纸面名义上限(USDT)",
        "max_live_notional_usdt": "实盘名义上限(USDT)",
        "daily_would_enter_count": "今日入场数",
        "daily_v5_entry_count": "今日 V5 实际 paper entry",
        "daily_synthetic_would_enter_count": "今日中台 synthetic would_enter",
        "cumulative_v5_entry_count": "累计 V5 paper entry",
        "cumulative_synthetic_would_enter_count": "累计中台 synthetic would_enter",
        "cumulative_would_enter_count": "累计入场数",
        "would_enter_count": "累计入场数(兼容字段)",
        "paper_source": "纸面来源",
        "paper_count_scope": "纸面计数口径",
        "daily_paper_pnl_observed_count": "今日收益标签数",
        "cumulative_paper_pnl_observed_count": "累计收益标签数",
        "paper_pnl_observed_count": "纸面收益观测数",
        "count_scope": "计数口径",
        "entry_day_count": "入场天数",
        "slippage_coverage": "滑点覆盖率",
        "readiness_status": "就绪状态",
        "advisory_reasons": "建议原因",
        "ready_for_live": "实盘就绪",
        "start_date": "开始日期",
        "end_date": "结束日期",
        "window_mode": "窗口模式",
        "cost_mode": "成本模式",
        "threshold_bps": "阈值(bps)",
        "would_block_count": "将拦截次数",
        "would_block_loss_count": "将拦截亏损数",
        "would_block_profit_count": "将误杀盈利数",
        "false_positive_rate": "误杀率",
        "avg_net_bps_blocked": "被拦截平均净收益(bps)",
        "avg_net_bps_not_blocked": "未拦截平均净收益(bps)",
        "check_name": "检查项",
        "violation_count": "违规数",
        "detail": "详情",
    }
)

VALUE_LOCALIZED_COLUMNS.update(
    {
        "advisory",
        "advisory_reasons",
        "advisory_intent",
        "check_name",
        "cost_quality",
        "cost_source",
        "cost_source_mix",
        "decision_reasons",
        "live_block_reasons",
        "mode",
        "recommended_mode",
        "readiness_status",
        "source_type",
        "strategy_candidate",
    }
)

VALUE_LABELS.update(
    {
        "CLOSE_RESEARCH": "关闭研究",
        "CLOSED": "已关闭",
        "CLOSE": "关闭",
        "REVIEW": "复查",
        "DOWNGRADED_FROM_PAPER": "已从纸面降级",
        "PAUSED": "暂停",
        "ACTIVE": "进行中",
        "SHADOW": "影子观察",
        "PAPER": "纸面观察",
        "BASELINE_ONLY": "仅作基线",
        "ACTIVE_DIAGNOSTIC": "诊断中",
        "TRACK_AS_RESEARCH_BASELINE": "作为研究基线跟踪",
        "CONTINUE_PAPER": "继续纸面观察",
        "CONTINUE_SHADOW": "继续影子观察",
        "CONTINUE_SHADOW_OR_REVIEW": "转影子观察/复查",
        "PAUSED_TO_WEEKLY": "降为周度复查",
        "WAIT_FOR_MORE_SAMPLES": "等待更多样本",
        "WAIT_FOR_UNIVERSE_OUTPUT": "等待扩展币池输出",
        "IMPROVE_COST_QUALITY": "改善成本质量",
        "REVIEW_WEEKLY": "周度复查",
        "PAPER_RESEARCH": "纸面研究",
        "KEEP_RESEARCH": "继续研究",
        "REGIME_SHADOW": "分状态影子观察",
        "insufficient_complete_samples": "完整样本不足",
        "insufficient_total_samples": "总样本不足",
        "non_positive_after_cost_edge": "扣成本后没有正边际",
        "win_rate_below_threshold": "胜率低于阈值",
        "paper_negative_streak": "纸面结果连续转弱",
        "downgraded_from_paper": "已从纸面降级",
        "research_portfolio_kill": "研究组合已淘汰",
        "research_paused": "研究已暂停",
        "baseline_only": "仅作研究基线",
        "research_portfolio_shadow": "研究组合限制为影子观察",
        "cost_source_not_trusted": "成本来源不够可信",
        "cost_source_not_actual_or_mixed": "成本还不是真实或混合真实成本",
        "no_paper_days": "纸面观察天数不足",
        "no_live_slippage_coverage": "实盘滑点覆盖不足",
        "shadow_only_collect_more_samples": "仅影子观察并继续收集样本",
        "insufficient_sample_count": "样本数量不足",
        "weak_24h_avg_net_bps": "24h 平均净收益偏弱",
        "expanded_universe_not_live_approved": "扩展币池未获准实盘",
        "futures_data_missing": "合约数据缺失",
        "funding_not_observable": "资金费率不可观测",
        "local_estimate": "本地估算成本",
        "cost_not_requested_no_order": "未下单未请求成本",
        "BLOCK": "阻断",
        "PAPER_ONLY": "仅 paper 可信",
        "CANARY": "小规模 canary 可信",
        "SCALE_READY": "可放大级可信",
        "NONE": "无",
        "GLOBAL_DEFAULT": "全局默认",
        "REGIME_FALLBACK": "状态回退",
        "SAMPLE_TOO_SMALL": "样本不足",
        "SLIPPAGE_UNKNOWN": "滑点未知",
        "PUBLIC_SPREAD_PROXY": "公共盘口价差代理",
        "FEE_MISSING": "费用缺失",
        "negative_after_cost_edge": "扣成本后边际为负",
        "complete_samples_sufficient_and_negative_after_cost_edge": "样本充足且扣成本后为负",
        "historical_pullback_reversal_v1_negative_expectancy_and_large_mae": (
            "回调反转 v1 历史负期望且 MAE 偏大"
        ),
        "research_closed_by_operator_after_negative_or_low_quality_evidence": (
            "负收益或低质量证据，人工关闭研究"
        ),
        "generic_momentum_baseline_not_global_strategy_gate": (
            "通用动量仅作研究基线，不再作为全局 gate"
        ),
        "eth_f3_negative_paper_streak_keep_shadow_no_live": "ETH f3 纸面连续转弱，退回影子观察",
        "sol_protect_negative_paper_streak_keep_shadow_no_live": (
            "SOL protect 纸面连续转弱，退回影子观察"
        ),
        "sol_f4_negative_paper_streak_keep_shadow_no_live": "SOL f4 纸面连续转弱，退回影子观察",
        "symbol_thresholds_are_research_only_no_hard_guard": "分币种阈值仍是研究，不启用硬拦截",
        "alt_impulse_is_regime_dependent_and_not_live_validated": (
            "Alt 脉冲依赖行情状态，尚未验证可实盘"
        ),
        "expanded_universe_shadow_collecting_replacement_candidates": (
            "扩展币池影子观察，继续收集替代候选"
        ),
        "no_recent_trigger": "近期无触发",
        "cost_source_not_paper_ready": "成本质量不足以纸面晋级",
        "insufficient_recent_samples": "近期样本不足",
        "validation_negative": "验证集为负",
        "recent_7d_negative": "近 7 天转弱",
        "futures_proxy_not_paper_ready": "合约代理结果不能纸面晋级",
        "ready_for_review": "需要复查",
        "quant_lab_would_have_missed_profit": "中台若硬拦会错过盈利",
        "quant_lab_correctly_avoided_loss": "中台正确避开亏损",
        "v5_missed_profit_opportunity": "V5 错过盈利机会",
        "v5_correctly_skipped_loss": "V5 正确跳过亏损",
        "pending": "待观察",
        "expanded_paper": "扩展币池纸面",
        "second_stage_alpha_factory": "二阶段 Alpha Factory",
        "alpha_factory": "Alpha Factory",
        "v5.core.momentum": "核心动量基线",
        "v5.expanded_relative_strength_top1_shadow": "扩展币池相对强度 Top1 影子",
        "v5.expanded_relative_strength_top3_shadow": "扩展币池相对强度 Top3 影子",
        "v5.expanded_relative_strength_rotation_shadow": "扩展币池相对强度轮动影子",
        "v5.futures_risk_off_hedge_proxy_shadow": "风险关闭合约对冲代理影子",
        "v5.futures_downtrend_short_proxy_shadow": "下跌趋势做空代理影子",
        "v5.risk_on_multi_buy_top1_shadow": "Risk-on 多币 Top1 影子",
        "v5.risk_on_multi_buy_top2_shadow": "Risk-on 多币 Top2 影子",
        "v5.risk_on_multi_buy_top3_shadow": "Risk-on 多币 Top3 影子",
        "v5.expanded_crypto_universe_shadow": "扩展币池影子研究",
        "v5.af.expanded_relative_strength_top1_shadow": "Alpha Factory 扩展相对强度 Top1",
        "v5.af.expanded_relative_strength_top3_shadow": "Alpha Factory 扩展相对强度 Top3",
        "v5.af.expanded_relative_strength": "Alpha Factory 扩展相对强度",
        "ETH_F3_DOMINANT_ENTRY_PAPER_V1": "ETH F3 主导纸面策略",
        "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1": "ETH F3 主导纸面策略",
        "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1": "SOL 保护态 Alpha6 低分例外纸面策略",
        "SOL_F4_VOLUME_EXPANSION_PAPER_V1": "SOL F4 放量纸面策略",
        "SOL_PROTECT_MOMENTUM_CONTINUATION_PAPER_V1": "SOL 保护态动量延续纸面策略",
        "ALT_IMPULSE_REGIME_SHADOW_V1": "Alt 脉冲分状态影子策略",
        "BTC_STRICT_PROBE_MONITOR_V1": "BTC 严格探针监控",
        "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW": "BTC 严格探针退出策略复查",
        "v5.missed_opportunity_audit": "错失机会审计",
        "research_baseline": "研究基线",
        "candidate_for_expanded_paper_universe": "扩展纸面币池候选",
        "quality_watchlist": "质量观察名单",
        "outcome_watchlist": "收益观察名单",
        "reject_list": "低优先/拒绝名单",
        "reject_low_priority_current_weak": "当前结果弱，低优先",
        "reject_low_liquidity": "流动性不足",
        "reject_high_spread": "价差过高",
        "reject_f3_noise": "F3 噪声偏高",
        "reject_negative_expectancy": "负期望",
        "keep_current": "保留当前币池",
    }
)

COLUMN_LABELS.update(
    {
        "action": "研究动作",
        "takeaway": "重点结论",
        "key_metrics": "关键指标",
        "severity": "影响级别",
        "next_action": "建议动作",
        "research_id": "研究编号",
        "module": "模块",
        "reason": "原因",
        "downgrade_reason": "降级原因",
        "paper_negative_streak": "纸面连续转弱天数",
        "latest_paper_trend": "最新纸面趋势",
        "last_review_date": "最近复查日",
        "next_review_date": "下次复查日",
        "priority": "优先级",
        "source_module": "来源模块",
        "template_family": "模板族",
        "promotion_state": "晋级状态",
        "alpha_factory_score": "Alpha Factory 分数",
        "universe_type": "币池类型",
        "cost_quality_score": "成本质量分",
        "paper_ready_block_reasons": "纸面阻断原因",
        "avg_net_bps_by_horizon": "分周期平均净收益",
        "win_rate_by_horizon": "分周期胜率",
        "downside_p25_by_horizon": "分周期 P25",
        "quality_score": "质量分",
        "bar_coverage": "K 线覆盖率",
        "spread_bps_p75": "P75 价差(bps)",
        "quote_volume_24h": "24h 成交额",
        "replacement_target_candidate": "替换目标候选",
        "candidate_state": "候选状态",
        "outcome_if_blocked": "如果拦截的结果",
        "current_regime": "当前行情",
        "actual_trade_opened": "V5 实际开仓",
        "quant_lab_recommended_mode": "中台建议模式",
        "future_4h_net_bps": "未来 4h 净收益(bps)",
        "future_8h_net_bps": "未来 8h 净收益(bps)",
        "future_24h_net_bps": "未来 24h 净收益(bps)",
        "selected_symbols": "选中标的",
        "would_buy_symbol": "影子买入标的",
        "portfolio_avg_net_bps": "组合平均净收益(bps)",
        "actual_v5_bought_symbols": "V5 实际买入标的",
        "missed_symbols": "错过标的",
        "vs_actual_v5_net_bps": "相对 V5 实际净收益(bps)",
        "regime_source": "行情来源",
        "recent_sample_sufficient": "近期样本充足",
        "actual_fill_count": "真实成交数",
        "api_cost_usage_rows": "API 成本请求数",
        "api_degraded_cost_count": "API 降级成本数",
        "api_global_default_count": "API 全局默认数",
        "api_regime_fallback_count": "API 状态回退数",
        "api_symbol_proxy_hit_count": "API 标的代理命中数",
        "cost_trust_level": "成本可信等级",
        "cost_trusted_for_live_canary": "canary 级可信",
        "cost_trusted_for_live_scale": "scale 级可信",
        "cost_probe_fill_count": "成本探针样本数",
        "eligible_for_live_cost_coverage": "可计入实盘成本覆盖",
        "fallback_reason": "回退原因",
        "fee_bps_p75": "P75 费用(bps)",
        "mixed_fill_count": "混合成本成交数",
        "notional_bucket": "名义金额分桶",
        "one_way_all_in_cost_bps": "单边 all-in 成本(bps)",
        "proxy_only_count": "纯代理成本数",
        "proxy_sample_count": "代理样本数",
        "private_fill_count": "只读私有成交数",
        "roundtrip_all_in_cost_bps": "往返 all-in 成本(bps)",
        "sample_origin_mix": "样本来源组合",
        "slippage_bps_p75": "P75 滑点(bps)",
        "strategy_live_fill_count": "策略实盘样本数",
        "symbols_with_actual_cost": "有真实成本标的",
        "symbols_with_mixed_cost": "有混合成本标的",
        "symbols_with_proxy_only": "仅代理成本标的",
        "total_cost_bps_p50": "P50 总成本(bps)",
        "total_cost_bps_p75": "P75 总成本(bps)",
        "total_cost_bps_p90": "P90 总成本(bps)",
        "warnings_json": "告警明细",
        "best_short_avg_net_bps": "最佳短周期净收益(bps)",
        "best_short_horizon_hours": "最佳短周期(小时)",
        "maturity_state": "成熟度状态",
        "positive_short_horizon_count": "为正短周期数",
        "watch_reason": "观察原因",
        "watchlist_type": "观察名单类型",
    }
)

VALUE_LOCALIZED_COLUMNS.update(
    {
        "action",
        "module",
        "promotion_state",
        "reason",
        "downgrade_reason",
        "latest_paper_trend",
        "paper_ready_block_reasons",
        "outcome_if_blocked",
        "current_regime",
        "cost_trust_level",
        "fallback_level",
        "fallback_reason",
        "recommendation",
        "regime_source",
        "research_id",
        "selected_symbols",
        "source_module",
        "source",
        "universe_type",
        "actual_v5_bought_symbols",
        "blocking_reasons",
        "missed_symbols",
        "maturity_state",
        "watch_reason",
        "watchlist_type",
    }
)


def streamlit_module(st_module: Any | None = None) -> Any:
    if st_module is not None:
        return st_module
    import streamlit as st

    return st


def show_warnings(st: Any, warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)


def show_frame(st: Any, df: pl.DataFrame, empty_message: str) -> None:
    if df.is_empty():
        st.info(empty_message)
        return
    localized = localize_frame(df)
    try:
        st.dataframe(localized, width="stretch", hide_index=True)
    except TypeError:
        try:
            st.dataframe(localized, use_container_width=True, hide_index=True)
        except TypeError:
            st.dataframe(localized)


def lake_caption(st: Any, lake_root: str | Path) -> None:
    st.caption(f"Lake 根目录：{Path(lake_root)}")


def display_value(value: Any) -> Any:
    if _is_display_time_value(value):
        return format_beijing_time(value)
    if isinstance(value, list):
        return "、".join(str(display_value(item)) for item in value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            label = str(display_value(str(key)))
            parts.append(f"{label}: {display_value(item)}")
        return "；".join(parts)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text in VALUE_LABELS:
            return VALUE_LABELS[text]
        jsonish = _display_jsonish_value(text)
        if jsonish is not None:
            return jsonish
    try:
        return VALUE_LABELS.get(value, value)
    except TypeError:
        return value


def display_unknown(value: Any) -> Any:
    if value is None or value == "unknown":
        return "未知"
    displayed = display_value(value)
    return str(displayed) if displayed is value else displayed


def localize_frame(df: pl.DataFrame) -> pl.DataFrame:
    df = _localize_frame_values(df)
    rename_map = {}
    used_names = set(df.columns)
    for column in df.columns:
        label = COLUMN_LABELS.get(column)
        if label is None or label in used_names:
            continue
        rename_map[column] = label
        used_names.add(label)
    if not rename_map:
        return df
    return df.rename(rename_map)


def _localize_frame_values(df: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    for column in df.columns:
        if is_time_column(column):
            expressions.append(
                pl.col(column).map_elements(_display_time_cell, return_dtype=pl.Utf8).alias(column)
            )
            continue
        if column == "value" and "key" in df.columns:
            expressions.append(
                pl.struct(["key", "value"])
                .map_elements(_display_key_value_cell, return_dtype=pl.Utf8)
                .alias(column)
            )
            continue
        if column not in VALUE_LOCALIZED_COLUMNS:
            continue
        expressions.append(
            pl.col(column)
            .map_elements(
                lambda value: str(display_value(value)),
                return_dtype=pl.Utf8,
            )
            .alias(column)
        )
    if not expressions:
        return df
    return df.with_columns(expressions)


def _display_time_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(display_value(value))


def _display_key_value_cell(row: dict[str, Any]) -> str:
    key = str(row.get("key") or "")
    value = row.get("value")
    if is_time_column(key):
        return _display_time_cell(value)
    return "" if value is None else str(display_value(value))


def _display_jsonish_value(text: str) -> str | None:
    if not (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("{") and text.endswith("}"))
    ):
        return None
    try:
        import json

        parsed = json.loads(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return "、".join(str(display_value(item)) for item in parsed)
    if isinstance(parsed, dict):
        parts = []
        for key, value in parsed.items():
            label = str(display_value(str(key)))
            parts.append(f"{label}: {display_value(value)}")
        return "；".join(parts)
    return str(display_value(parsed))


def _is_display_time_value(value: Any) -> bool:
    if hasattr(value, "tzinfo"):
        return True
    text = str(value) if value is not None else ""
    return bool(text and ("T" in text or text.endswith("Z")) and format_beijing_time(text) != text)
