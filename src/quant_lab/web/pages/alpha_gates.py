from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import (
    display_value,
    lake_caption,
    show_frame,
    show_warnings,
    streamlit_module,
)


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.alpha_gate_summary(lake_root)

    st.title("\u7b56\u7565\u673a\u4f1a")
    lake_caption(st, lake_root)
    st.caption(
        "\u53ea\u8bfb\u7814\u7a76\u9875\u9762\u3002\u91cd\u70b9\u5c55\u793a V5 "
        "\u53ef\u8bfb\u53d6\u7684\u7b56\u7565\u5019\u9009\u3001"
        "\u7814\u7a76\u7ec4\u5408\u88c1\u526a\u3001Alpha Factory "
        "\u548c\u6269\u5c55\u5e01\u6c60 shadow\uff1b"
        "\u4e0d\u63d0\u4f9b\u4efb\u4f55\u4ea4\u6613\u64cd\u4f5c\u3002"
    )

    st.subheader("今日先看：研究组合裁剪")
    st.caption(
        "这里是最终研究资源管理视图。“关闭研究”表示该方向不再进入重点日报或晋级队列，不是交易平仓。"
    )
    show_frame(
        st,
        summary["research_portfolio_status"],
        "暂无研究组合裁剪状态。建议运行 qlab build-research-portfolio-status。",
    )

    st.subheader("给 V5 的只读策略建议")
    st.caption(
        "这是 V5 可读取的最终 advisory。"
        "即使推荐 paper/shadow，max_live_notional_usdt 也必须保持 0。"
    )
    show_frame(
        st,
        summary["strategy_opportunity_advisory"],
        "\u6682\u65e0\u7b56\u7565\u673a\u4f1a\u5efa\u8bae\u6570\u636e\u3002",
    )

    st.subheader("逐笔 trade-level 判断")
    st.caption(
        "V5 先提出候选交易，中台只读分层为 HARD_BLOCK / PAPER / MICRO_CANARY_REVIEW。"
        "这里不下单，只用于逐笔审计和人工复核。"
    )
    show_frame(
        st,
        summary["trade_level_judgment"],
        "暂无 trade-level 逐笔判断。建议运行 qlab build-trade-level-judgment。",
    )
    show_frame(
        st,
        summary["quant_lab_false_block_audit"],
        "暂无中台误杀审计。",
    )

    st.subheader("Alpha Factory 与自动候选")
    st.caption(
        "自动生成新候选、做 train/validation/recent 分层，再进入 shadow/paper 队列。"
        "这里优先看晋级状态、近期稳定性和纸面阻断原因。"
    )
    show_frame(
        st,
        summary["alpha_factory_promotion_queue"],
        "暂无 Alpha Factory promotion queue。",
    )
    show_frame(
        st,
        summary["alpha_factory_result"],
        "暂无 Alpha Factory 结果。",
    )

    st.subheader("\u6269\u5c55\u5e01\u6c60\u91cd\u70b9")
    st.caption(
        "\u4ece OKX USDT \u73b0\u8d27\u4e2d\u7b5b\u9009\u9ad8\u6d41\u52a8\u6027\u3001"
        "\u4f4e\u70b9\u5dee\u3001\u975e\u7a33\u5b9a\u5e01\u3001\u975e\u9ad8\u98ce\u9669 meme "
        "\u7684\u5019\u9009\u3002\u5148\u770b watchlist\u3001quality \u548c promotion queue；"
        "\u5019\u9009\u4e8b\u4ef6\u548c label \u4e0d\u518d\u9ed8\u8ba4"
        "\u94fa\u6ee1\u9875\u9762\u3002"
    )
    show_frame(
        st,
        summary["expanded_universe_watchlist"],
        "暂无扩展币池 watchlist。",
    )
    show_frame(
        st,
        summary["expanded_universe_promotion_queue"],
        "暂无扩展币池 promotion queue。",
    )
    show_frame(
        st,
        summary["expanded_universe_quality"],
        "暂无扩展币池质量评分。",
    )

    st.subheader("错失机会与 Risk-on 多币影子")
    st.caption(
        "用于回答“中台如果过度保守会错过什么”以及“上涨状态下是否应该同时观察多个主池币”。"
        "这些结果只做审计和 shadow，不改变 V5 live。"
    )
    show_frame(
        st,
        summary["missed_opportunity_audit"],
        "暂无错失机会审计。",
    )
    show_frame(
        st,
        summary["risk_on_multi_buy_shadow"],
        "暂无 risk-on 多币影子结果。",
    )

    st.subheader("\U0001f4ca \u5019\u9009\u51b3\u7b56\u9762\u677f")
    for decision, count in summary["alpha_discovery_counts"].items():
        st.metric(display_value(decision), count)
    show_frame(
        st,
        summary["alpha_discovery_board"],
        "\u6682\u65e0\u5019\u9009\u51b3\u7b56\u9762\u677f\u6570\u636e\u3002",
    )

    st.subheader("\U0001f9ea \u7b56\u7565\u8bc1\u636e\u805a\u5408")
    for decision, count in summary["strategy_counts"].items():
        st.metric(display_value(decision), count)
    show_frame(
        st,
        summary["strategy_evidence"],
        "\u6682\u65e0\u7b56\u7565\u8bc1\u636e\u805a\u5408\u6570\u636e\u3002",
    )

    st.subheader("\u57fa\u7ebf\u56e0\u5b50\u95e8\u63a7\uff08\u4ec5\u4f5c\u57fa\u7ebf\uff09")
    for status, count in summary["counts"].items():
        st.metric(display_value(status), count)
    show_frame(st, summary["gates"], "\u6682\u65e0\u95e8\u63a7\u51b3\u7b56\u6570\u636e\u3002")

    st.subheader("\U0001f9fe V5 \u5019\u9009\u8d28\u91cf")
    show_frame(
        st,
        summary["candidate_quality"],
        "\u6682\u65e0 V5 \u5019\u9009\u8d28\u91cf\u6570\u636e\u3002",
    )
    show_frame(
        st,
        summary["candidate_outcomes"],
        "\u6682\u65e0 V5 \u5019\u9009\u540e\u9a8c\u6c47\u603b\u6570\u636e\u3002",
    )
    show_warnings(st, summary["warnings"])
