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
    "PAPER_READY": "Paper 就绪",
    "LIVE_READY": "Live 就绪",
    "LIVE_SMALL_READY": "小仓 live 就绪",
    "KEEP_SHADOW": "继续 Shadow",
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
    "alpha_id": "Alpha ID",
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
    "bundle_ts": "bundle 时间",
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
    "fallback_level": "fallback 级别",
    "feature_completeness": "特征完整率",
    "final_decision": "最终决策",
    "final_score": "最终分数",
    "freshness_seconds": "新鲜度(秒)",
    "freshness_status": "新鲜度状态",
    "gate_version": "Gate 版本",
    "global_default_rows": "全局默认成本行数",
    "gross_bps": "毛收益(bps)",
    "high": "最高价",
    "horizon_hours": "标签周期(小时)",
    "ingest_ts": "入湖时间",
    "key": "键",
    "lag": "延迟状态",
    "label_completeness": "标签完整率",
    "label_status": "标签状态",
    "label_ts": "标签时间",
    "latest_bundle_sha256": "最新 bundle SHA256",
    "latest_bundle_ts": "最新 bundle 时间",
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
    "paper_days": "Paper 天数",
    "path": "路径",
    "p25_net_bps": "净收益 P25(bps)",
    "permission": "权限",
    "protect_level": "保护等级",
    "proxy_rows": "代理成本行数",
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
    st.dataframe(localize_frame(df))


def lake_caption(st: Any, lake_root: str | Path) -> None:
    st.caption(f"Lake 根目录：{Path(lake_root)}")


def display_value(value: Any) -> Any:
    if _is_display_time_value(value):
        return format_beijing_time(value)
    return VALUE_LABELS.get(value, value)


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
                pl.col(column)
                .map_elements(_display_time_cell, return_dtype=pl.Utf8)
                .alias(column)
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


def _is_display_time_value(value: Any) -> bool:
    if hasattr(value, "tzinfo"):
        return True
    text = str(value) if value is not None else ""
    return bool(text and ("T" in text or text.endswith("Z")) and format_beijing_time(text) != text)
