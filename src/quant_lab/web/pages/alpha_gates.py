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
        "\u6269\u5c55\u5e01\u6c60 shadow\u3001strategy evidence "
        "\u4e0e\u57fa\u7ebf\u95e8\u63a7\uff1b"
        "\u4e0d\u63d0\u4f9b\u4efb\u4f55\u4ea4\u6613\u64cd\u4f5c\u3002"
    )

    st.subheader("V5 \u53ef\u8bfb\u7b56\u7565\u673a\u4f1a")
    show_frame(
        st,
        summary["strategy_opportunity_advisory"],
        "\u6682\u65e0\u7b56\u7565\u673a\u4f1a\u5efa\u8bae\u6570\u636e\u3002",
    )

    st.subheader("研究组合裁剪")
    st.caption(
        "每天自动给出研究项的保留、降级、暂停和关闭建议；"
        "该表只用于研究资源管理，不改变 V5 live 或 risk permission。"
    )
    show_frame(
        st,
        summary["research_portfolio_status"],
        "暂无研究组合裁剪状态。建议运行 qlab build-research-portfolio-status。",
    )

    st.subheader("\u6269\u5c55\u5e01\u6c60 Shadow \u7814\u7a76")
    st.caption(
        "\u4ece OKX USDT \u73b0\u8d27\u4e2d\u7b5b\u9009\u9ad8\u6d41\u52a8\u6027\u3001"
        "\u4f4e\u70b9\u5dee\u3001\u975e\u7a33\u5b9a\u5e01\u3001\u975e\u9ad8\u98ce\u9669 meme "
        "\u7684\u5019\u9009\uff0c\u89c2\u5bdf\u662f\u5426\u6709\u6bd4 ETH/BNB "
        "\u66f4\u9002\u5408 V5 \u7684\u66ff\u4ee3\u6807\u7684\u3002\u8be5\u6a21\u5757\u53ea\u505a "
        "shadow/recommendation\uff0c\u4e0d\u4fee\u6539 V5 \u5b9e\u76d8\u914d\u7f6e\u3002"
    )
    show_frame(
        st,
        summary["expanded_crypto_universe_shadow"],
        "\u6682\u65e0\u6269\u5c55\u5e01\u6c60 shadow \u7ed3\u679c\u3002"
        "\u5efa\u8bae\u8fd0\u884c qlab build-expanded-universe-shadow\u3002",
    )
    show_frame(st, summary["symbol_quality_score"], "\u6682\u65e0 symbol quality score\u3002")
    show_frame(
        st,
        summary["expanded_crypto_recommendations"],
        "\u6682\u65e0\u6269\u5c55\u5e01\u6c60\u63a8\u8350\u6458\u8981\u3002",
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

    st.subheader("\u57fa\u7ebf\u56e0\u5b50\u95e8\u63a7")
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
