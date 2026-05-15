from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import (
    display_unknown,
    display_value,
    lake_caption,
    show_frame,
    show_warnings,
    streamlit_module,
)


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.v5_telemetry_summary(lake_root)
    latest = summary["latest"]

    st.title("V5 遥测")
    lake_caption(st, lake_root)
    st.metric("状态", display_value(latest.get("status", "UNKNOWN")))
    st.metric("最新 bundle 包时间", display_unknown(latest.get("latest_bundle_ts", "unknown")))
    st.metric(
        "最新 bundle 包 sha256",
        display_unknown(latest.get("latest_bundle_sha256", "unknown")),
    )
    st.metric("72h 运行次数", latest.get("run_count_72h", 0))
    st.metric("24h 决策审计数", latest.get("decision_audit_count_24h", 0))
    st.metric("24h 交易笔数", latest.get("trade_count_24h", 0))
    st.metric("72h 交易笔数", latest.get("trade_count_72h", 0))
    st.metric("72h 回合数", latest.get("roundtrip_count_72h", 0))
    st.metric("当前持仓数", latest.get("open_position_count", 0))
    st.metric("残余小额持仓数", latest.get("dust_residual_position_count", 0))
    st.metric("熔断开关", display_unknown(latest.get("kill_switch_enabled", "unknown")))
    st.metric("对账正常", display_unknown(latest.get("reconcile_ok", "unknown")))
    st.metric("账本正常", display_unknown(latest.get("ledger_ok", "unknown")))
    st.metric("自动风险等级", display_unknown(latest.get("auto_risk_level", "unknown")))
    st.metric("高风险问题数", latest.get("high_issue_count", 0))
    st.metric("中风险问题数", latest.get("medium_issue_count", 0))
    st.metric("未生效配置数", latest.get("config_not_consumed_count", 0))
    st.metric("高分阻塞数", latest.get("high_score_blocked_count", 0))
    st.metric("高分成熟阻塞数", latest.get("high_score_blocked_matured_count", 0))

    st.subheader("策略每日健康")
    show_frame(st, summary["health_rows"], "暂无 V5 strategy_health_daily 数据。")
    st.subheader("门控合规")
    show_frame(st, summary["gate_compliance_rows"], "暂无 V5 门控合规数据。")
    st.subheader("quant-lab 模式")
    show_frame(st, summary["quant_lab_mode_rows"], "暂无 V5 quant-lab 模式数据。")
    st.subheader("quant-lab 执行情况")
    show_frame(
        st,
        summary["quant_lab_enforcement_rows"],
        "暂无 V5 quant-lab 执行数据。",
    )
    show_warnings(st, summary["warnings"])
