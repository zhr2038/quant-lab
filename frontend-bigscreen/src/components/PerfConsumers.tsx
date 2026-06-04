import ReactECharts from "echarts-for-react";
import { Gauge, ShieldCheck } from "lucide-react";
import { ms, safeRows } from "../lib/api";

export function PerfConsumers({
  perf,
  consumers
}: {
  perf: Record<string, unknown>;
  consumers: Record<string, unknown>;
}) {
  const events = safeRows(perf.recent_events);
  const values = events.map((row) => Number(row.elapsed_ms)).filter((value) => Number.isFinite(value));
  const fallback = [18, 26, 20, 32, 24, 46, 35, 22, 29, 25, 31, 21];
  const series = values.length ? values.slice(-24) : fallback;
  const option = {
    backgroundColor: "transparent",
    grid: { top: 8, left: 0, right: 0, bottom: 0 },
    xAxis: { type: "category", show: false, data: series.map((_, i) => i) },
    yAxis: { type: "value", show: false },
    series: [
      {
        type: "line",
        smooth: true,
        data: series,
        symbol: "none",
        lineStyle: { color: "#A77DFF", width: 2 },
        areaStyle: { color: "rgba(167,125,255,.18)" }
      }
    ]
  };
  const permissions = (consumers.permissions ?? {}) as Record<string, unknown>;
  return (
    <section className="card perf pad">
      <h2 className="section-title icon-title"><Gauge size={23} />Web 性能 / 策略消费者</h2>
      <p className="sub">API metrics、reader cache、risk_permission、fallback audit 合并。</p>
      <ReactECharts option={option} style={{ height: 105, marginTop: 20 }} />
      <div className="latency-text">{ms(perf.api_p50_ms)} / {ms(perf.api_p95_ms)}</div>
      <div className="perm-grid">
        <div className="perm"><span><ShieldCheck size={14} />V5</span><strong>{String(permissions.v5 ?? "UNKNOWN")}</strong></div>
        <div className="perm"><span>V7</span><strong>{String(permissions.v7 ?? "UNKNOWN")}</strong></div>
        <div className="perm"><span>fallback</span><strong>{String(consumers.fallback_rows ?? 0)} rows</strong></div>
        <div className="perm"><span>rglob</span><strong>{String(perf.rglob_fallback ?? 0)}</strong></div>
      </div>
    </section>
  );
}
