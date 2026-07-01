import { Gauge, ShieldCheck } from "lucide-react";
import { ms, permissionDisplay, safeRows, shortNumber, stringValue } from "../lib/api";
import { ReactECharts } from "./EChart";

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
  const v5Permission = permissionDisplay(permissions.v5);
  const permissionRows = safeRows(consumers.permission_rows);
  const latestPermission = permissionRows.find((row) => stringValue(row.permission_status, "")) ?? permissionRows[0] ?? {};
  const tradeLevelSummary = tradeLevelDisplay(
    latestPermission.trade_level_decision_summary,
    latestPermission.micro_canary_review_count
  );
  return (
    <section className="card perf pad">
      <h2 className="section-title icon-title"><Gauge size={23} />Web 性能 / 策略消费者</h2>
      <p className="sub">API metrics、reader cache、risk_permission、fallback audit 合并。</p>
      <ReactECharts option={option} style={{ height: 105, marginTop: 20 }} />
      <div className="latency-text">{ms(perf.api_p50_ms)} / {ms(perf.api_p95_ms)}</div>
      <div className="perm-grid">
        <div className="perm"><span><ShieldCheck size={14} />V5</span><strong>{v5Permission.value}</strong></div>
        <div className="perm" title={tradeLevelSummary.title}><span>逐笔</span><strong>{tradeLevelSummary.value}</strong></div>
        <div className="perm"><span>fallback</span><strong>{String(consumers.fallback_rows ?? 0)} rows</strong></div>
        <div className="perm"><span>rglob</span><strong>{String(perf.rglob_fallback ?? 0)}</strong></div>
      </div>
    </section>
  );
}

function tradeLevelDisplay(
  summaryValue: unknown,
  microCanaryReviewCount: unknown
): { value: string; title: string } {
  const reviewCount = Number(microCanaryReviewCount ?? 0);
  const raw = stringValue(summaryValue, "");
  const fallback = {
    value: reviewCount > 0 ? `review ${shortNumber(reviewCount)}` : "no rows",
    title: raw || "trade_level_decision_summary not observable"
  };
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const entries = Object.entries(parsed)
      .filter(([, value]) => Number.isFinite(Number(value)))
      .sort((left, right) => Number(right[1]) - Number(left[1]));
    const top = entries[0];
    if (!top) return fallback;
    const [decision, count] = top;
    const prefix = reviewCount > 0 ? `review ${shortNumber(reviewCount)}` : decision.replace(/_/g, " ");
    return {
      value: `${prefix} ${shortNumber(count)}`,
      title: raw
    };
  } catch {
    return fallback;
  }
}
