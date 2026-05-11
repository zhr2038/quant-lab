from pathlib import Path
from typing import Any

import polars as pl

VALUE_LABELS = {
    None: "未知",
    True: "是",
    False: "否",
    "unknown": "未知",
    "UNKNOWN": "未知",
    "OK": "正常",
    "WARNING": "警告",
    "CRITICAL": "严重",
    "NOT_CONFIGURED": "未配置",
    "ALLOW": "允许",
    "SELL_ONLY": "仅卖出",
    "ABORT": "中止",
    "fresh": "新鲜",
    "delayed": "延迟",
    "stale": "过期",
    "missing": "缺失",
}

COLUMN_LABELS = {
    "abnormal_symbols": "异常标的",
    "actual_bars": "实际 K 线数",
    "allowed_modes": "允许模式",
    "alpha_id": "Alpha ID",
    "ask": "卖一",
    "avg_volume": "平均成交量",
    "bucket_index": "分桶索引",
    "channel": "频道",
    "close": "收盘价",
    "collector": "采集器",
    "cost_day": "成本日期",
    "cost_model_version": "成本模型版本",
    "count": "数量",
    "created_at": "创建时间",
    "dataset": "数据集",
    "date": "日期",
    "day": "日期",
    "duplicate_bar_count": "重复 K 线数",
    "edge_cost_ratio": "边际/成本比",
    "error_count": "错误次数",
    "expected_bars": "预期 K 线数",
    "exists": "是否存在",
    "fallback_level": "fallback 级别",
    "gate_version": "Gate 版本",
    "high": "最高价",
    "ingest_ts": "入湖时间",
    "key": "键",
    "lag": "延迟状态",
    "latest_success_ts": "最近成功时间",
    "latest_trade_ts": "最近成交时间",
    "latest_ts": "最新时间",
    "low": "最低价",
    "market_type": "市场类型",
    "mean_abs_return": "平均绝对收益",
    "metrics": "指标",
    "missing_bars": "缺失 K 线数",
    "modified_at": "修改时间",
    "name": "名称",
    "next_action": "下一步动作",
    "oos_max_drawdown": "样本外最大回撤",
    "oos_sharpe": "样本外夏普",
    "open": "开盘价",
    "parquet_file_count": "Parquet 文件数",
    "passed": "是否通过",
    "path": "路径",
    "permission": "权限",
    "question": "问题",
    "rate_limit_warnings": "限频警告数",
    "reasons": "原因",
    "received_at": "接收时间",
    "reconnect_count": "重连次数",
    "regime": "状态",
    "rows": "行数",
    "sample_count": "样本数",
    "size_bytes": "大小（字节）",
    "size_sum": "成交量合计",
    "source": "来源",
    "spread_bps": "价差 bps",
    "status": "状态",
    "strategy": "策略",
    "success_count": "成功次数",
    "symbol": "标的",
    "timeframe": "周期",
    "trade_count": "成交笔数",
    "ts": "时间",
    "value": "值",
    "venue": "交易所",
    "version": "版本",
    "violation": "违规项",
    "volatility_regime": "波动状态",
    "volume": "成交量",
    "warning": "告警",
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
    return VALUE_LABELS.get(value, value)


def display_unknown(value: Any) -> Any:
    if value is None or value == "unknown":
        return "未知"
    displayed = display_value(value)
    return str(displayed) if displayed is value else displayed


def localize_frame(df: pl.DataFrame) -> pl.DataFrame:
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
