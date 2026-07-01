import { Waves } from "lucide-react";
import { bps, safeRows, stringValue } from "../lib/api";

type MarketLiquidityProps = {
  market: Record<string, unknown>;
  matrix?: Record<string, unknown>;
  density?: "compact" | "full";
};

export function MarketLiquidity({ market, matrix, density = "compact" }: MarketLiquidityProps) {
  const rows = mergeMarketRows(safeRows(market.regimes), safeRows(matrix?.rows));
  const visibleRows = rows.slice(0, density === "full" ? 24 : 16);
  const spreadValues = rows
    .map((row) => Number(row.spread_bps))
    .filter((value) => Number.isFinite(value));
  const avgSpread = spreadValues.length
    ? spreadValues.reduce((total, value) => total + value, 0) / spreadValues.length
    : undefined;
  const maxSpreadRow = rows.reduce<Record<string, unknown> | undefined>((best, row) => {
    const spread = Number(row.spread_bps);
    if (!Number.isFinite(spread)) return best;
    if (!best) return row;
    const bestSpread = Number(best.spread_bps);
    return spread > bestSpread ? row : best;
  }, undefined);
  const missingSpreadCount = rows.length - spreadValues.length;
  const lowVolCount = rows.filter((row) => stringValue(row.volatility_regime ?? row.regime, "").includes("低")).length;
  const normalCount = rows.filter((row) => stringValue(row.volatility_regime ?? row.regime, "").includes("正常")).length;
  return (
    <section className={`card market pad market-${density}`}>
      <h2 className="section-title icon-title"><Waves size={23} />市场状态与流动性</h2>
      <p className="sub">波动状态 / spread bps / trade activity 由 market_regime_summary 汇总。</p>
      {!rows.length && (
        <div className="market-empty-state">
          <Waves size={34} />
          <b>等待市场 regime 数据</b>
          <span>market_regime_summary 暂无可展示行；数据恢复后会自动填充流动性队列。</span>
        </div>
      )}
      {!!rows.length && (
        <div className="market-body">
          <div className="market-list">
            {visibleRows.map((row, i) => {
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
          </div>
          <aside className="market-summary-panel">
            <div className="market-summary-title">Regime Summary</div>
            <div className="market-summary-grid">
              <span><b>{rows.length}</b><em>symbols</em></span>
              <span><b>{lowVolCount}</b><em>低波动</em></span>
              <span><b>{normalCount}</b><em>正常</em></span>
              <span><b>{bps(avgSpread)}</b><em>平均价差</em></span>
            </div>
            <div className="market-summary-line">
              <span>最高价差</span>
              <strong>
                {stringValue(maxSpreadRow?.symbol, "—")}
                {" · "}
                {bps(maxSpreadRow?.spread_bps)}
              </strong>
            </div>
            <div className="market-summary-line">
              <span>缺失价差</span>
              <strong>{missingSpreadCount}</strong>
            </div>
          </aside>
        </div>
      )}
    </section>
  );
}

function mergeMarketRows(
  regimeRows: Record<string, unknown>[],
  matrixRows: Record<string, unknown>[]
): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  const seen = new Set<string>();
  const addRow = (row: Record<string, unknown>) => {
    const symbol = stringValue(row.symbol, "").trim();
    if (!symbol.endsWith("-USDT")) return;
    if (!symbol || seen.has(symbol)) return;
    seen.add(symbol);
    rows.push(row);
  };
  regimeRows.forEach(addRow);
  matrixRows.forEach((row) => {
    const marketBar = (row.market_bar ?? {}) as Record<string, unknown>;
    const spread = (row.spread ?? {}) as Record<string, unknown>;
    addRow({
      symbol: row.symbol,
      volatility_regime: marketBar.regime,
      regime: marketBar.regime,
      spread_bps: spread.spread_bps
    });
  });
  return rows;
}
