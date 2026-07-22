import { FlaskConical, GitBranch, Rocket, Scale } from "lucide-react";
import { bps, pct, safeRows, shortNumber, stringValue } from "../lib/api";

export function StrategyFlow({ flow }: { flow: Record<string, unknown> }) {
  const counts = (flow.counts ?? {}) as Record<string, number>;
  const topCandidates = dedupeCandidates([
    ...safeRows(flow.top_live_candidates),
    ...safeRows(flow.top_candidates)
  ]);
  const factorFactory = (flow.factor_factory ?? {}) as Record<string, unknown>;
  const opportunityCost = (flow.opportunity_cost ?? {}) as Record<string, unknown>;
  const paperLifecycle = (flow.paper_lifecycle ?? {}) as Record<string, unknown>;
  const lifecycleCounts = (paperLifecycle.counts ?? {}) as Record<string, number>;
  const hypothesisRows = safeRows(factorFactory.hypotheses);
  const trialRows = safeRows(factorFactory.trials);
  const attributionRows = safeRows(factorFactory.attribution).filter(
    (row) => stringValue(row.trial_id, "") && stringValue(row.factor_id, "")
  );
  const portfolioRows = safeRows(factorFactory.portfolio_validation).filter(
    (row) => stringValue(row.trial_id, "") && stringValue(row.factor_id, "")
  );
  const nasTask = (factorFactory.nas_task ?? {}) as Record<string, unknown>;
  const nasTaskDetail = (nasTask.task ?? {}) as Record<string, unknown>;
  const generation = (factorFactory.generation ?? {}) as Record<string, unknown>;
  const nasTaskId = stringValue(nasTaskDetail.task_id, "");
  const generationId = stringValue(generation.generation_id, "");
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
      <div className="paper-lifecycle" aria-label="Paper 策略生命周期">
        <LifecycleStage label="提案就绪" value={lifecycleCounts.PAPER_PROPOSAL_READY} />
        <LifecycleStage label="等待 ACK" value={lifecycleCounts.PAPER_ACK_PENDING} />
        <LifecycleStage label="Tracker" value={lifecycleCounts.PAPER_TRACKER_ACTIVE} />
        <LifecycleStage label="证据不足" value={lifecycleCounts.PAPER_EVIDENCE_INSUFFICIENT} />
        <LifecycleStage label="晋级就绪" value={lifecycleCounts.PAPER_PROMOTION_READY} />
        <span className="paper-lifecycle-mode">
          <b>{stringValue(paperLifecycle.runtime_mode, "PAPER_ONLY")}</b>
          <em>{stringValue(paperLifecycle.live_order_effect, "none")}</em>
        </span>
      </div>
      <div className="strategy-research-grid">
        <div className="opportunity-cost-mini">
          <div className="candidate-title"><Scale size={15} /> 机会成本 / 拦截价值</div>
          <div className="opportunity-cost-stats">
            <span><b>{bps(opportunityCost.veto_net_value_bps)}</b><em>今日净值</em></span>
            <span><b>{bps(opportunityCost.veto_net_value_bps_7d)}</b><em>7日净值</em></span>
            <span><b>{bps(opportunityCost.missed_profit_bps)}</b><em>错过收益</em></span>
            <span><b>{bps(opportunityCost.loss_saved_bps)}</b><em>保护亏损</em></span>
          </div>
          <div className="opportunity-cost-note" title={stringValue(opportunityCost.status, "NO_DATA")}>
            <span>{stringValue(opportunityCost.latest_day, "no-day")}</span>
            <em>{stringValue(opportunityCost.status, "NO_DATA")}</em>
            <strong>
              误杀 {shortNumber(opportunityCost.false_block_count)}
              {" / "}
              保护 {shortNumber(opportunityCost.loss_saved_count)}
              {" · "}
              高置信误杀 {shortNumber(opportunityCost.high_confidence_false_block_count_7d)}
              {" / "}
              保护 {shortNumber(opportunityCost.high_confidence_loss_saved_count_7d)}
            </strong>
          </div>
        </div>
        <div className="factor-factory-mini">
          <div className="candidate-title"><FlaskConical size={15} /> Factor Research v2</div>
          <div className="factor-factory-stats">
            <span><b>{shortNumber(factorFactory.independent_hypothesis_count)}</b><em>独立假设</em></span>
            <span><b>{shortNumber(factorFactory.active_hypothesis_count)}</b><em>活跃假设</em></span>
            <span><b>{shortNumber(factorFactory.trial_budget_usage_pct)}%</b><em>试验预算</em></span>
          </div>
          <div
            className="factor-generation-grid"
            data-testid="factor-generation-summary"
            title={[nasTaskId, generationId].filter(Boolean).join(" | ")}
          >
            <div className="factor-generation-cell">
              <span>NAS 任务</span>
              <b>{researchTaskState(stringValue(nasTask.state, "idle"))}</b>
              <em>{shortId(nasTaskId || generationId)}</em>
            </div>
            <div className="factor-generation-cell verdict">
              <span>本轮结论</span>
              <b>{researchVerdict(stringValue(factorFactory.current_generation_verdict, "NO_CURRENT_TRIALS"))}</b>
              <em>{stringValue(factorFactory.current_generation_verdict, "NO_CURRENT_TRIALS")}</em>
            </div>
            <div className="factor-generation-cell">
              <span>试验审查</span>
              <b>{shortNumber(factorFactory.current_trial_count)} 个试验</b>
              <em>
                质量拒绝 {shortNumber(factorFactory.current_data_quality_rejected_count)}
                {" · FDR 通过 "}{shortNumber(factorFactory.multiple_testing_pass_count)}
                {" · PIT成本 "}{pct(factorFactory.minimum_point_in_time_cost_coverage, 1)}
                {" · 可信 "}{pct(factorFactory.minimum_trusted_cost_coverage, 1)}
              </em>
            </div>
          </div>
        </div>
        <div className="candidate-list">
          <div className="candidate-title"><Rocket size={15} /> 策略候选（只读）</div>
          <div className="candidate-list-body">
            {topCandidates.slice(0, 6).map((candidate, i) => (
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
      <div className="strategy-secondary-grid">
        <ResearchMiniPanel
          title="假设"
          rows={hypothesisRows.slice(0, 4).map((row) => [
            stringValue(row.hypothesis_id, "hypothesis"),
            stringValue(row.factor_family, "family"),
            stringValue(row.status, "DRAFT")
          ])}
        />
        <ResearchMiniPanel
          title="试验"
          rows={trialRows.slice(0, 4).map((row) => [
            stringValue(row.trial_id, "trial"),
            stringValue(row.trial_kind, "EXPLORATORY"),
            stringValue(row.decision ?? row.status, "SUBMITTED")
          ])}
        />
        <ResearchMiniPanel
          title="归因"
          rows={attributionRows.slice(0, 4).map((row) => [
            stringValue(row.factor_id, "factor"),
            stringValue(row.attribution_type, "unknown"),
            `residual ${shortNumber(row.joint_residual_rank_ic)}`
          ])}
        />
        <ResearchMiniPanel
          title="组合"
          rows={portfolioRows.slice(0, 4).map((row) => [
            stringValue(row.factor_id, "factor"),
            stringValue(row.portfolio_validity, "UNKNOWN"),
            `${stringValue(row.decision, "RESEARCH")} · PIT ${pct(row.cost_coverage, 0)} · trusted ${pct(row.trusted_cost_coverage, 0)}`
          ])}
        />
      </div>
    </section>
  );
}

function LifecycleStage({ label, value }: { label: string; value: number | undefined }) {
  return (
    <span className="paper-lifecycle-stage">
      <b>{shortNumber(value ?? 0)}</b>
      <em>{label}</em>
    </span>
  );
}

function ResearchMiniPanel({ title, rows }: { title: string; rows: string[][] }) {
  return (
    <div className="research-mini-panel">
      <div className="research-mini-title">{title}</div>
      <div className="research-mini-list">
        {rows.map((row, i) => (
          <div className="research-mini-row" key={`${title}-${i}`} title={row.join(" · ")}>
            <strong>{row[0]}</strong>
            <span>{row[1]}</span>
            <em>{row[2]}</em>
          </div>
        ))}
        {!rows.length && <div className="research-mini-empty">not_observable</div>}
      </div>
    </div>
  );
}

function shortId(value: string): string {
  if (!value) return "no-task";
  const compact = value.replace(/^factor-research-/, "");
  return compact.length > 10 ? `${compact.slice(0, 10)}…` : compact;
}

function researchTaskState(value: string): string {
  const state = value.trim().toLowerCase();
  if (state === "completed") return "计算完成";
  if (state === "running" || state === "computing") return "计算中";
  if (state === "pending" || state === "queued") return "等待计算";
  if (state === "failed") return "计算失败";
  return state === "idle" ? "暂无任务" : value;
}

function researchVerdict(value: string): string {
  const verdicts: Record<string, string> = {
    DATA_QUALITY_BLOCKED: "数据质量阻塞",
    MULTIPLE_TESTING_BLOCKED: "统计检验阻塞",
    NO_CURRENT_TRIALS: "暂无本轮试验",
    RESEARCH_ONLY: "仅限研究",
    READY_FOR_REVIEW: "等待人工复核"
  };
  return verdicts[value] ?? value.replace(/_/g, " ");
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
