import ReactECharts from "echarts-for-react";
import { FlaskConical, GitBranch, Rocket, Sparkles } from "lucide-react";
import { bps, safeRows, shortNumber, stringValue } from "../lib/api";

export function StrategyFlow({ flow }: { flow: Record<string, unknown> }) {
  const counts = (flow.counts ?? {}) as Record<string, number>;
  const topCandidates = dedupeCandidates(safeRows(flow.top_live_candidates).length
    ? safeRows(flow.top_live_candidates)
    : safeRows(flow.top_candidates));
  const factorFactory = (flow.factor_factory ?? {}) as Record<string, unknown>;
  const factorReviewQueue = safeRows(factorFactory.paper_review_queue);
  const factorRows = factorReviewQueue.length
    ? factorReviewQueue
    : safeRows(factorFactory.paper_ready_candidates).length
      ? safeRows(factorFactory.paper_ready_candidates)
      : safeRows(factorFactory.top_candidates);
  const option = {
    backgroundColor: "transparent",
    animationDuration: 1200,
    tooltip: {},
    series: [
      {
        type: "sankey",
        top: 20,
        bottom: 20,
        left: 8,
        right: 8,
        emphasis: { focus: "adjacency" },
        nodeAlign: "justify",
        data: [
          { name: "Research", itemStyle: { color: "#50A9FF" } },
          { name: "Shadow", itemStyle: { color: "#A77DFF" } },
          { name: "Paper", itemStyle: { color: "#2DE8A6" } },
          { name: "Kill", itemStyle: { color: "#FF5D7D" } }
        ],
        links: [
          { source: "Research", target: "Shadow", value: Math.max(1, counts.shadow ?? 0) },
          { source: "Shadow", target: "Paper", value: Math.max(1, counts.paper ?? 0) },
          { source: "Research", target: "Kill", value: Math.max(1, counts.kill ?? 0) }
        ],
        lineStyle: { color: "gradient", curveness: 0.5 },
        label: { color: "#eaf6ff", fontSize: 12 }
      }
    ]
  };
  return (
    <section className="card pad strategy-card">
      <h2 className="section-title icon-title"><GitBranch size={23} />策略机会流</h2>
      <p className="sub">研究组合裁剪 → shadow → paper → advisory；主屏优先展示最接近 paper/live-review 的候选。</p>
      <div className="flow-metrics">
        <Metric label="Research" value={counts.research ?? 0} tone="blue" />
        <Metric label="Shadow" value={counts.shadow ?? 0} tone="purple" />
        <Metric label="Paper" value={counts.paper ?? 0} tone="green" />
        <Metric label="Kill" value={counts.kill ?? 0} tone="red" />
      </div>
      <div className="flow-line"><i /></div>
      <ReactECharts option={option} style={{ height: 112, marginTop: -8 }} />
      <div className="factor-factory-mini">
        <div className="candidate-title"><FlaskConical size={15} /> Factor Factory</div>
        <div className="factor-factory-stats">
          <span><b>{shortNumber(factorFactory.candidate_count)}</b><em>候选</em></span>
          <span><b>{shortNumber(factorFactory.paper_review_queue_count ?? factorFactory.paper_ready_count)}</b><em>Paper候选</em></span>
          <span><b>{shortNumber(factorFactory.strategy_bridge_candidate_count ?? safeRows(factorFactory.strategy_bridge_candidates).length)}</b><em>Bridge</em></span>
        </div>
        {factorRows.slice(0, 2).map((factor, i) => (
          <div className="factor-chip" key={`${factor.factor_id}-${i}`}>
            <Sparkles size={13} />
            <span>{stringValue(factor.factor_id ?? factor.factor_name, "factor")}</span>
            <em>{stringValue(factor.state ?? factor.candidate_state, "RESEARCH")}</em>
            <strong>{bps(factor.long_short_bps ?? factor.best_long_short_mean_bps)}</strong>
          </div>
        ))}
        {!factorRows.length && <div className="factor-empty">Factor Factory 暂无候选</div>}
      </div>
      <div className="candidate-list">
        <div className="candidate-title"><Rocket size={15} /> 策略候选（只读）</div>
        {topCandidates.slice(0, 4).map((candidate, i) => (
          <div className="chip" key={candidateKey(candidate, i)} title={candidateTitle(candidate)}>
            <span className="candidate-main">
              <b>{candidateIdentity(candidate)}</b>
              <small>{stringValue(candidate.strategy_candidate ?? candidate.takeaway, "candidate")}</small>
            </span>
            <em>{modeLabel(candidate)}</em>
            <strong>{bps(candidate.avg_net_bps)}</strong>
            <span className="candidate-detail">{candidateDetail(candidate)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function dedupeCandidates(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  const seen = new Set<string>();
  return rows.filter((candidate) => {
    const key = [
      stringValue(candidate.symbol, ""),
      stringValue(candidate.horizon_hours, ""),
      stringValue(candidate.recommended_mode ?? candidate.decision, "")
    ].join("|");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function Metric({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className={`stage ${tone}`}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function candidateKey(candidate: Record<string, unknown>, index: number): string {
  return [
    candidate.strategy_candidate,
    candidate.symbol,
    candidate.horizon_hours,
    candidate.source_module,
    candidate.promotion_state,
    index
  ].map((value) => stringValue(value, "na")).join("|");
}

function candidateIdentity(candidate: Record<string, unknown>): string {
  const symbol = stringValue(candidate.symbol, "MULTI");
  const horizon = candidate.horizon_hours ? `${shortNumber(candidate.horizon_hours)}h` : "horizon ?";
  return `${symbol} · ${horizon}`;
}

function candidateDetail(candidate: Record<string, unknown>): string {
  const source = stringValue(candidate.source_module ?? candidate.promotion_state, "advisory");
  const samples = candidate.complete_sample_count ? `n=${shortNumber(candidate.complete_sample_count)}` : "n=?";
  const p25 = `p25 ${bps(candidate.p25_net_bps)}`;
  return `${source} · ${samples} · ${p25}`;
}

function modeLabel(candidate: Record<string, unknown>): string {
  const mode = stringValue(candidate.recommended_mode, "").toLowerCase();
  const decision = stringValue(candidate.decision, "").toUpperCase();
  if (mode === "paper" || decision === "PAPER_READY") return "PAPER";
  if (mode === "shadow" || decision.includes("SHADOW")) return "SHADOW";
  if (mode === "research" || decision === "RESEARCH_ONLY") return "RESEARCH";
  if (mode === "none" || decision === "KILL") return "RESEARCH-ONLY";
  return stringValue(candidate.recommended_mode ?? candidate.decision, "RESEARCH");
}

function candidateTitle(candidate: Record<string, unknown>): string {
  return [
    candidateIdentity(candidate),
    stringValue(candidate.strategy_candidate ?? candidate.takeaway, "candidate"),
    `mode=${modeLabel(candidate)}`,
    `avg=${bps(candidate.avg_net_bps)}`,
    candidateDetail(candidate)
  ].join(" | ");
}
