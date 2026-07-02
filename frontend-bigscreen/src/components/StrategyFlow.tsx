import { FlaskConical, GitBranch, Rocket, Scale, Sparkles } from "lucide-react";
import { bps, safeRows, shortNumber, stringValue } from "../lib/api";

export function StrategyFlow({ flow }: { flow: Record<string, unknown> }) {
  const counts = (flow.counts ?? {}) as Record<string, number>;
  const topCandidates = dedupeCandidates([
    ...safeRows(flow.top_live_candidates),
    ...safeRows(flow.top_candidates)
  ]);
  const factorFactory = (flow.factor_factory ?? {}) as Record<string, unknown>;
  const opportunityCost = (flow.opportunity_cost ?? {}) as Record<string, unknown>;
  const opportunityBuckets = safeRows(opportunityCost.top_buckets);
  const factorReviewQueue = safeRows(factorFactory.paper_review_queue);
  const factorRows = dedupeFactorRows([
    ...safeRows(factorFactory.paper_ready_candidates),
    ...factorReviewQueue,
    ...safeRows(factorFactory.top_candidates),
    ...safeRows(factorFactory.strategy_bridge_candidates)
  ]);
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
      <div className="flow-rail" aria-label="策略机会流阶段">
        <span>Research</span>
        <i />
        <span>Shadow</span>
        <i />
        <span>Paper</span>
        <i />
        <span>Advisory</span>
      </div>
      <div className="strategy-research-grid">
        <div className="opportunity-cost-mini">
          <div className="candidate-title"><Scale size={15} /> 机会成本 / 拦截价值</div>
          <div className="opportunity-cost-stats">
            <span><b>{bps(opportunityCost.veto_net_value_bps)}</b><em>今日净值</em></span>
            <span><b>{bps(opportunityCost.missed_profit_bps)}</b><em>错过收益</em></span>
            <span><b>{bps(opportunityCost.loss_saved_bps)}</b><em>保护亏损</em></span>
            <span><b>{bps(opportunityCost.veto_net_value_bps_7d)}</b><em>7日净值</em></span>
            <span><b>{shortNumber(opportunityCost.false_block_count)}</b><em>误杀次数</em></span>
            <span><b>{shortNumber(opportunityCost.loss_saved_count)}</b><em>保护次数</em></span>
          </div>
          <div className="opportunity-cost-note" title={stringValue(opportunityCost.status, "NO_DATA")}>
            <span>{stringValue(opportunityCost.latest_day, "no-day")}</span>
            <em>{stringValue(opportunityCost.status, "NO_DATA")}</em>
            <strong>
              高置信误杀 {shortNumber(opportunityCost.high_confidence_false_block_count_7d)}
              {" / "}
              保护 {shortNumber(opportunityCost.high_confidence_loss_saved_count_7d)}
            </strong>
          </div>
          {opportunityBuckets.slice(0, 6).map((bucket, i) => (
            <div className="opportunity-bucket" key={`${bucket.bucket_key}-${i}`} title={bucketTitle(bucket)}>
              <span>{bucketIdentity(bucket)}</span>
              <em>{stringValue(bucket.recommended_trade_level_decision, "REVIEW")}</em>
              <strong>{bps(bucket.veto_net_value_bps)}</strong>
            </div>
          ))}
          {!opportunityBuckets.length && <div className="factor-empty">暂无拦截价值样本</div>}
        </div>
        <div className="factor-factory-mini">
          <div className="candidate-title"><FlaskConical size={15} /> Factor Factory</div>
          <div className="factor-factory-stats">
            <span><b>{shortNumber(factorFactory.candidate_count)}</b><em>候选</em></span>
            <span><b>{shortNumber(factorFactory.paper_review_queue_count ?? factorFactory.paper_ready_count)}</b><em>Paper候选</em></span>
            <span><b>{shortNumber(factorFactory.strategy_bridge_candidate_count ?? safeRows(factorFactory.strategy_bridge_candidates).length)}</b><em>Bridge</em></span>
          </div>
          <div className="factor-chip-grid">
            {factorRows.slice(0, 24).map((factor, i) => (
              <div className="factor-chip" key={`${factor.factor_id}-${i}`}>
                <Sparkles size={13} />
                <span>{stringValue(factor.factor_id ?? factor.factor_name, "factor")}</span>
                <em>{stringValue(factor.state ?? factor.candidate_state, "RESEARCH")}</em>
                <strong>{bps(factor.long_short_bps ?? factor.best_long_short_mean_bps)}</strong>
              </div>
            ))}
          </div>
          {!factorRows.length && <div className="factor-empty">Factor Factory 暂无候选</div>}
        </div>
        <div className="candidate-list">
          <div className="candidate-title"><Rocket size={15} /> 策略候选（只读）</div>
          <div className="candidate-list-body">
            {topCandidates.slice(0, 18).map((candidate, i) => (
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
        </div>
      </div>
    </section>
  );
}

function bucketIdentity(bucket: Record<string, unknown>): string {
  const symbol = stringValue(bucket.symbol, "UNKNOWN");
  const rank = stringValue(bucket.rank_bucket, "rank?");
  const alpha = stringValue(bucket.alpha6_bucket, "alpha?");
  return `${symbol} · ${rank} · ${alpha}`;
}

function bucketTitle(bucket: Record<string, unknown>): string {
  return [
    stringValue(bucket.bucket_key, "bucket"),
    `false_block=${shortNumber(bucket.false_block_count)}`,
    `loss_saved=${shortNumber(bucket.loss_saved_count)}`,
    `veto_net=${bps(bucket.veto_net_value_bps)}`
  ].join(" | ");
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

function dedupeFactorRows(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  const seen = new Set<string>();
  return rows.filter((factor) => {
    const key = [
      stringValue(factor.factor_id ?? factor.factor_name, ""),
      stringValue(factor.best_horizon_bars ?? factor.horizon ?? factor.horizon_bars, ""),
      stringValue(factor.candidate_state ?? factor.state ?? factor.recommended_action, "")
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
