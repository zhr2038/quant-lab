import { DatabaseZap } from "lucide-react";
import { statusClass, statusText } from "../lib/api";

type MatrixStatusCounts = {
  ok: number;
  warning: number;
  critical: number;
  unknown: number;
};

export function DataMatrix({
  matrix,
  variant = "default"
}: {
  matrix: { columns: string[]; rows: Record<string, unknown>[] };
  variant?: "default" | "overview";
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
  const rowLimit = variant === "overview" ? 12 : 18;
  const rows = matrix.rows
    .filter((row) => String(row.symbol ?? "").endsWith("-USDT"))
    .slice(0, rowLimit);
  const matrixCells = rows.flatMap((row) => {
    const symbol = String(row.symbol ?? "—").replace("-USDT", "");
    return matrix.columns.map((column) => {
      const cell = row[column] as Record<string, unknown> | undefined;
      const status = statusClass(cell?.status);
      return {
        symbol,
        column,
        label: labels[column] ?? column,
        cell,
        status
      };
    });
  });
  const statusCounts = rows.reduce<MatrixStatusCounts>(
    (counts, row) => {
      matrix.columns.forEach((column) => {
        const cell = row[column] as Record<string, unknown> | undefined;
        const status = statusClass(cell?.status);
        if (status === "critical") counts.critical += 1;
        else if (status === "warning") counts.warning += 1;
        else if (status === "ok") counts.ok += 1;
        else counts.unknown += 1;
      });
      return counts;
    },
    { ok: 0, warning: 0, critical: 0, unknown: 0 }
  );
  const statusTotal = Math.max(
    1,
    statusCounts.ok + statusCounts.warning + statusCounts.critical + statusCounts.unknown
  );
  const issueCells = matrixCells
    .filter((item) => item.status === "critical" || item.status === "warning")
    .sort((a, b) => {
      const rank = { critical: 0, warning: 1, ok: 2, info: 3 };
      return rank[a.status] - rank[b.status];
    })
    .slice(0, 6);
  const summarizeCell = (cell: Record<string, unknown> | undefined) => {
    const keys = [
      "reason",
      "freshness_reason",
      "coverage_reason",
      "message",
      "state",
      "latest_close_ts",
      "latest_ts",
      "updated_at",
      "source"
    ];
    const detail = keys
      .map((key) => cell?.[key])
      .find((value) => value !== undefined && value !== null && String(value).trim());
    const text = String(detail ?? statusText(cell?.status));
    return text.length > 78 ? `${text.slice(0, 75)}...` : text;
  };
  const coverageRows = [
    { key: "ok", label: "OK", value: statusCounts.ok, tone: "ok" },
    { key: "warning", label: "注意", value: statusCounts.warning, tone: "warning" },
    { key: "critical", label: "异常", value: statusCounts.critical, tone: "critical" },
    { key: "unknown", label: "未知", value: statusCounts.unknown, tone: "info" }
  ];
  return (
    <section className={`card heatmap pad${variant === "overview" ? " overview-matrix-card" : ""}`}>
      <h2 className="section-title icon-title"><DatabaseZap size={23} />数据 / 市场可信矩阵</h2>
      <p className="sub">按 symbol 汇总行情、WS、价差、成交、成本、证据、advisory；异常格会闪烁。</p>
      <div className="matrix-grid-shell">
        <div className="matrix">
          <div />
          {matrix.columns.map((column) => <div className="h" key={column}>{labels[column] ?? column}</div>)}
          {rows.flatMap((row) => {
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
          {!rows.length && (
            <div className="matrix-empty-state">
              暂无 symbol 矩阵数据 · 等待 market_bar / advisory / evidence 刷新
            </div>
          )}
        </div>
        <aside className="matrix-side-panel">
          <div className="matrix-side-title">Freshness Timeline</div>
          <svg className="sparkline matrix-spark" viewBox="0 0 480 96" aria-hidden="true">
            <path d="M0 58 C42 16 84 24 126 62 S214 86 256 48 S336 10 386 34 S444 18 480 52" fill="none" stroke="#50a9ff" strokeWidth="3" />
            <path d="M0 72 C58 46 104 48 154 70 S256 88 316 56 S412 44 480 66" fill="none" stroke="rgba(45,232,166,.72)" strokeWidth="2" />
          </svg>
          <div className="matrix-side-stats">
            <span><b>{rows.length}</b><em>symbols</em></span>
            <span><b>{statusCounts.ok}</b><em>ok</em></span>
            <span><b>{statusCounts.warning}</b><em>warn</em></span>
            <span><b>{statusCounts.critical}</b><em>critical</em></span>
          </div>
        </aside>
      </div>
      <div className="matrix-bottom-panel">
        <div className="matrix-issue-panel">
          <div className="matrix-mini-title">需关注格子</div>
          <div className="matrix-issue-list">
            {issueCells.map((item) => (
              <div className={`matrix-issue-row ${item.status}`} key={`${item.symbol}-${item.column}-issue`}>
                <strong>{item.symbol}</strong>
                <span>{item.label}</span>
                <em>{summarizeCell(item.cell)}</em>
              </div>
            ))}
            {!issueCells.length && (
              <div className="matrix-issue-empty">当前矩阵没有 critical / warning 格子。</div>
            )}
          </div>
        </div>
        <div className="matrix-coverage-panel">
          <div className="matrix-mini-title">覆盖概览</div>
          <div className="matrix-coverage-list">
            {coverageRows.map((item) => (
              <div className="matrix-status-bar" key={item.key}>
                <span>{item.label}</span>
                <div className="matrix-status-track">
                  <i
                    className={`matrix-status-fill ${item.tone}`}
                    style={{ width: `${Math.max(3, Math.round((item.value / statusTotal) * 100))}%` }}
                  />
                </div>
                <b>{item.value}</b>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
