import {
  ArrowRight,
  BrainCircuit,
  Clock3,
  FileSearch,
  FlaskConical,
  GitBranch,
  Layers3,
  ListChecks,
  ShieldCheck,
  Sparkles
} from "lucide-react";
import { delay, safeRows, shortNumber, statusClass, stringValue } from "../lib/api";

export function AIResearchPanel({ research }: { research: Record<string, unknown> }) {
  const counts = (research.counts ?? {}) as Record<string, unknown>;
  const queue = (research.queue ?? {}) as Record<string, unknown>;
  const queueCounts = (queue.counts ?? {}) as Record<string, unknown>;
  const latest = (research.latest_run ?? {}) as Record<string, unknown>;
  const continuity = (research.continuity ?? {}) as Record<string, unknown>;
  const status = stringValue(research.status, "WAITING_FOR_FIRST_RESULT");
  const findings = safeRows(research.findings);
  const rootCause = safeRows(research.root_cause_tree);
  const nextActions = safeRows(research.next_actions);
  const hypotheses = safeRows(research.research_hypothesis_drafts);
  const dataProposals = safeRows(research.data_collection_proposals);
  const attributionExperiments = safeRows(research.attribution_experiments);
  const factorProposals = safeRows(research.factor_proposals);
  const experimentProposals = safeRows(research.experiment_proposals);
  const codeTargets = safeRows(research.code_review_targets);
  const validationEvents = safeRows(research.validation_events);
  const primaryId = stringValue(research.primary_bottleneck_id, "");
  const primaryFinding = findings.find((row) => stringValue(row.finding_id, "") === primaryId);
  const pending = Number(queueCounts.pending ?? 0);
  const running = Number(queueCounts.running ?? 0);
  const stage2Allowed = latest.stage2_allowed === true || String(latest.stage2_allowed).toLowerCase() === "true";
  const stage1Attempts = Number(latest.stage1_attempts ?? 0);
  const stage2Attempts = Number(latest.stage2_attempts ?? 0);
  const hasResult = Boolean(stringValue(latest.task_id, ""));
  const sourcePackFreshness = stringValue(research.source_pack_freshness_status, "NOT_OBSERVABLE");
  const sourcePackIsStale = sourcePackFreshness === "STALE_SOURCE_PACK";
  const latestAvailablePackName = stringValue(research.latest_available_pack_name, "");
  const stage1State = hasResult ? "COMPLETE" : "WAITING";
  const stage2State = !hasResult ? "WAITING" : stage2Allowed ? (stage2Attempts > 0 ? "COMPLETE" : "PENDING") : "GATED";
  const preflightStatus = stringValue(latest.preflight_status, "NOT_AVAILABLE");
  const preflightDisplay = preflightStatus === "NOT_AVAILABLE" ? "LEGACY NOT RECORDED" : preflightStatus;
  const routeSections = jsonStringList(latest.route_sections_json);
  const usesFactorProposalFallback = !hypotheses.length && factorProposals.length > 0;
  const usesExperimentProposalFallback = !attributionExperiments.length && experimentProposals.length > 0;
  const hypothesisCount = metricCount(
    usesFactorProposalFallback ? latest.factor_proposal_count : latest.hypothesis_draft_count,
    usesFactorProposalFallback ? factorProposals.length : hypotheses.length
  );
  const dataProposalCount = metricCount(latest.data_collection_proposal_count, dataProposals.length);
  const experimentCount = metricCount(
    usesExperimentProposalFallback ? latest.experiment_count : latest.attribution_experiment_count,
    usesExperimentProposalFallback ? experimentProposals.length : attributionExperiments.length
  );
  const statusLabel = running > 0 && hasResult
    ? `RUNNING · ${sourcePackIsStale ? "OLD RESULT" : "LAST RESULT"}`
    : pending > 0 && hasResult
      ? `PENDING · ${sourcePackIsStale ? "OLD RESULT" : "LAST RESULT"}`
      : sourcePackIsStale
        ? "STALE SOURCE PACK"
      : status.replace(/_/g, " ");

  return (
    <section className="card pad ai-research">
      <div className="ai-head">
        <div title={`${stringValue(primaryFinding?.summary, stringValue(latest.system_state, "等待首个结果"))} · ${stringValue(latest.executive_summary, stringValue(latest.task_id, "尚无导入结果"))}`}>
          <h2 className="section-title icon-title"><BrainCircuit size={24} />AI 研究工作台</h2>
          <p className="sub">确定性预检 → 诊断 → 研究假设 → 验证实验 → 人工复核</p>
        </div>
        <span className={`ai-status ${sourcePackIsStale ? "warning" : statusClass(status)}`}><ShieldCheck size={15} />{statusLabel}</span>
      </div>

      <div className={`ai-safety-strip ${sourcePackIsStale ? "stale" : ""}`}>
        <ShieldCheck size={17} />
        <b>{sourcePackIsStale ? "只读历史结果" : "只读研究"}</b>
        <span>{sourcePackIsStale
          ? `最新权威包 ${latestAvailablePackName || "已生成"} 正在等待或进行 AI 消费；下方旧结论不代表当前状态。`
          : `不生成交易信号 · 不修改 V5 · 不自动晋级 · live effect ${stringValue(research.live_order_effect, "none")}`}</span>
      </div>

      <div className="ai-metrics">
        <Metric icon={Layers3} label="历史运行" value={counts.run_count} />
        <Metric icon={FileSearch} label="本轮发现" value={latest.finding_count} />
        <Metric icon={Sparkles} label={usesFactorProposalFallback ? "本轮因子草案" : "本轮研究假设"} value={hypothesisCount} />
        <Metric icon={FileSearch} label="数据采集建议" value={dataProposalCount} />
        <Metric icon={FlaskConical} label={usesExperimentProposalFallback ? "验证实验草案" : "归因实验"} value={experimentCount} />
        <Metric icon={Clock3} label="队列" value={`${pending} / ${running}`} sub="pending / running" />
      </div>

      <div className="ai-latest">
        <div>
          <span>{sourcePackIsStale ? "上次研究主要矛盾" : "当前主要矛盾"}</span>
          <b>{stringValue(primaryFinding?.summary, stringValue(latest.system_state, "等待首个结果"))}</b>
          <small>{stringValue(latest.executive_summary, stringValue(latest.task_id, "尚无导入结果"))}</small>
        </div>
        <div>
          <span>证据来源与连续性</span>
          <b>{preflightDisplay} · {stringValue(continuity.status, "FIRST RUN")}</b>
          <small title={stringValue(continuity.summary, "尚无上一轮研究上下文")}>
            {preflightStatus === "NOT_AVAILABLE" ? "旧结果未回传确定性预检 · " : ""}证据包 {stringValue(latest.source_pack_name, "尚无来源包")} · {delay(research.latest_run_age_seconds)}
            {sourcePackIsStale && latestAvailablePackName ? ` · 最新 ${latestAvailablePackName}` : ""}
          </small>
        </div>
        <div>
          <span>两阶段状态</span>
          <b>Stage 1 {stage1State} · Stage 2 {stage2State}</b>
          <small>{stage1Attempts || stage2Attempts ? `尝试 ${stage1Attempts} / ${stage2Attempts}` : "尚未运行"} · 路由 {routeSections.length ? routeSections.join(", ") : "—"} · {validationEvents.length ? `${validationEvents.length} 次重试` : "无重试"}</small>
        </div>
      </div>

      <div className="ai-work-grid">
        <section className="ai-diagnosis">
          <div className="ai-block-head"><GitBranch size={17} /><h3>根因链</h3><span>{rootCause.length}</span></div>
          <div className="ai-root-chain">
            {rootCause.length ? rootCause.slice(0, 6).map((node, index) => (
              <article key={stringValue(node.node_id, `root-${index}`)} title={stringValue(node.label, "未命名根因")}>
                <em>{stringValue(node.causal_role, "unknown")}</em>
                <b>{stringValue(node.label, "未命名根因")}</b>
                {index < Math.min(rootCause.length, 6) - 1 ? <ArrowRight size={14} /> : null}
              </article>
            )) : <p>等待 Stage 1 生成证据化根因链</p>}
          </div>

          <div className="ai-block-head"><ListChecks size={17} /><h3>下一步最小动作</h3><span>{nextActions.length}</span></div>
          <div className="ai-action-list">
            {nextActions.length ? nextActions.slice(0, 6).map((action, index) => (
              <article key={stringValue(action.action_id, `action-${index}`)} title={`${stringValue(action.title, "未命名动作")} · ${stringValue(action.rationale, "—")}`}>
                <span>{stringValue(action.priority, "P2")}</span>
                <div>
                  <b>{stringValue(action.title, "未命名动作")}</b>
                  <small>{stringValue(action.rationale, "—")}</small>
                </div>
                <em>{stringValue(action.action_type, "review").replace(/_/g, " ")}</em>
              </article>
            )) : <p>暂无动作；若证据被阻断，应先修复数据而不是生成策略。</p>}
          </div>
        </section>

        <div className="ai-evidence-grid">
          <EvidenceList title="诊断发现" rows={findings} primary="summary" secondary="category" trailing="severity" empty="等待 Stage 1 诊断" />
          {usesFactorProposalFallback
            ? <EvidenceList title="因子研究草案" rows={factorProposals} primary="factor_name" fallbackPrimary="proposal_id" secondary="hypothesis" trailing="proposal_state" empty="暂无因子研究草案" />
            : <EvidenceList title="研究假设草案" rows={hypotheses} primary="title" fallbackPrimary="hypothesis_id" secondary="economic_return_payer" trailing="proposal_state" empty="暂无通过证据门槛的研究假设" />}
          <EvidenceList title="数据采集建议" rows={dataProposals} primary="title" fallbackPrimary="proposal_id" secondary="observed_data_gap" trailing="proposal_state" empty="当前没有新增数据采集建议" />
          {usesExperimentProposalFallback
            ? <EvidenceList title="验证实验草案" rows={experimentProposals} primary="objective" fallbackPrimary="proposal_id" secondary="hypothesis" trailing="proposal_state" empty="暂无验证实验草案" />
            : <EvidenceList title="归因实验" rows={attributionExperiments} primary="title" fallbackPrimary="experiment_id" secondary="attribution_question" trailing="proposal_state" empty="暂无可执行的只读归因实验" />}
          <EvidenceList title="代码复核目标" rows={codeTargets} primary="path_or_component" secondary="reason" trailing="priority" empty="暂无代码复核目标" />
        </div>
      </div>
    </section>
  );
}

function jsonStringList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map((item) => String(item)).filter(Boolean);
  if (typeof value !== "string" || !value.trim()) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map((item) => String(item)).filter(Boolean) : [];
  } catch {
    return [];
  }
}

function metricCount(value: unknown, fallback: number): number {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? Math.max(parsed, fallback) : fallback;
}

function Metric({ icon: Icon, label, value, sub }: { icon: typeof BrainCircuit; label: string; value: unknown; sub?: string }) {
  return (
    <article className="ai-metric">
      <Icon size={17} /><span>{label}</span><b>{typeof value === "string" ? value : shortNumber(value)}</b>
      {sub ? <small>{sub}</small> : null}
    </article>
  );
}

function EvidenceList({ title, rows, primary, fallbackPrimary, secondary, trailing, empty }: {
  title: string;
  rows: Record<string, unknown>[];
  primary: string;
  fallbackPrimary?: string;
  secondary: string;
  trailing: string;
  empty: string;
}) {
  return (
    <section className="ai-evidence-list">
      <h3>{title}<span>{rows.length}</span></h3>
      <div>
        {rows.length ? rows.slice(0, 6).map((row, index) => (
          <article
            key={`${stringValue(row[primary], stringValue(row[fallbackPrimary ?? ""], "row"))}-${index}`}
            title={`${stringValue(row[primary], stringValue(row[fallbackPrimary ?? ""], "—"))} · ${stringValue(row[secondary], "—")}`}
          >
            <b>{stringValue(row[primary], stringValue(row[fallbackPrimary ?? ""], "—"))}</b>
            <small>{stringValue(row[secondary], "—")}</small>
            <em>{stringValue(row[trailing], "—")}</em>
          </article>
        )) : <p>{empty}</p>}
      </div>
    </section>
  );
}
