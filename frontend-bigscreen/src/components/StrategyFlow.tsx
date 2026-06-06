import ReactECharts from "echarts-for-react";
import { FlaskConical, GitBranch, Rocket, Sparkles } from "lucide-react";
import { bps, safeRows, shortNumber, stringValue } from "../lib/api";

export function StrategyFlow({ flow }: { flow: Record<string, unknown> }) {
  const counts = (flow.counts ?? {}) as Record<string, number>;
  const topCandidates = safeRows(flow.top_live_candidates).length
    ? safeRows(flow.top_live_candidates)
    : safeRows(flow.top_candidates);
  const factorFactory = (flow.factor_factory ?? {}) as Record<string, unknown>;
  const factorRows = safeRows(factorFactory.paper_ready_candidates).length
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
      <ReactECharts option={option} style={{ height: 150, marginTop: -8 }} />
      <div className="factor-factory-mini">
        <div className="candidate-title"><FlaskConical size={15} /> Factor Factory</div>
        <div className="factor-factory-stats">
          <span><b>{shortNumber(factorFactory.candidate_count)}</b><em>候选</em></span>
          <span><b>{shortNumber(factorFactory.paper_ready_count)}</b><em>PAPER</em></span>
          <span><b>{shortNumber(factorFactory.high_correlation_pair_count)}</b><em>高相关</em></span>
        </div>
        {factorRows.slice(0, 2).map((factor, i) => (
          <div className="factor-chip" key={`${factor.factor_id}-${i}`}>
            <Sparkles size={13} />
            <span>{stringValue(factor.factor_id ?? factor.factor_name, "factor")}</span>
            <em>{stringValue(factor.candidate_state, "RESEARCH")}</em>
            <strong>{bps(factor.best_long_short_mean_bps)}</strong>
          </div>
        ))}
        {!factorRows.length && <div className="factor-empty">Factor Factory 暂无候选</div>}
      </div>
      <div className="candidate-list">
        <div className="candidate-title"><Rocket size={15} /> 最可能上线的研究候选</div>
        {topCandidates.slice(0, 4).map((candidate, i) => (
          <div className="chip" key={`${candidate.strategy_candidate}-${i}`}>
            <span>{stringValue(candidate.strategy_candidate ?? candidate.takeaway, "candidate")}</span>
            <em>{stringValue(candidate.recommended_mode ?? candidate.decision, "research")}</em>
            <strong>{bps(candidate.avg_net_bps)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function Metric({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className={`stage ${tone}`}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}
