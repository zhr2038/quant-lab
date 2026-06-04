import { Waves } from "lucide-react";
import { bps, safeRows, stringValue } from "../lib/api";

export function MarketLiquidity({ market }: { market: Record<string, unknown> }) {
  const rows = safeRows(market.regimes);
  return (
    <section className="card market pad">
      <h2 className="section-title icon-title"><Waves size={23} />市场状态与流动性</h2>
      <p className="sub">波动状态 / spread bps / trade activity 由 market_regime_summary 汇总。</p>
      {rows.slice(0, 6).map((row, i) => {
        const spread = Number(row.spread_bps ?? 0);
        const tone = spread >= 6 ? "red" : spread >= 3 ? "yellow" : "";
        return (
          <div className={`ticker ${tone}`} key={`${row.symbol}-${i}`}>
            <b>{String(row.symbol ?? "—")}</b>
            <span className="state">{stringValue(row.volatility_regime ?? row.regime, "未知")}</span>
            <span className="wave-line" style={{ ["--phase" as string]: `${i * 8}px` }} />
            <span>{bps(row.spread_bps)}</span>
          </div>
        );
      })}
    </section>
  );
}
