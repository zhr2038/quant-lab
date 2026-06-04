import { DatabaseZap } from "lucide-react";
import { statusClass, statusText } from "../lib/api";

export function DataMatrix({
  matrix
}: {
  matrix: { columns: string[]; rows: Record<string, unknown>[] };
}) {
  const labels: Record<string, string> = {
    market_bar: "行情",
    ws: "WS",
    spread: "价差",
    trade: "成交",
    cost: "成本",
    evidence: "证据",
    advisory: "建议"
  };
  return (
    <section className="card heatmap pad">
      <h2 className="section-title icon-title"><DatabaseZap size={23} />数据 / 市场可信矩阵</h2>
      <p className="sub">按 symbol 汇总行情、WS、价差、成交、成本、证据、advisory；异常格会闪烁。</p>
      <div className="matrix">
        <div />
        {matrix.columns.map((column) => <div className="h" key={column}>{labels[column] ?? column}</div>)}
        {matrix.rows.slice(0, 8).flatMap((row) => {
          const symbol = String(row.symbol ?? "—").replace("-USDT", "");
          return [
            <div className="symbol" key={`${symbol}-label`}>{symbol}</div>,
            ...matrix.columns.map((column) => {
              const cell = row[column] as Record<string, unknown> | undefined;
              const status = statusClass(cell?.status);
              return (
                <span
                  className={`cell ${status}`}
                  key={`${symbol}-${column}`}
                  title={`${symbol} ${labels[column] ?? column}: ${statusText(cell?.status)} · ${JSON.stringify(cell ?? {})}`}
                />
              );
            })
          ];
        })}
      </div>
      <svg className="sparkline matrix-spark" viewBox="0 0 480 56" aria-hidden="true">
        <path d="M0 34 C40 4 72 16 116 44 S196 60 240 32 S314 10 360 28 S430 6 480 38" fill="none" stroke="#50a9ff" strokeWidth="2" />
        <text x="0" y="52" fill="#8aa4be" fontSize="11">Freshness Timeline</text>
      </svg>
    </section>
  );
}
