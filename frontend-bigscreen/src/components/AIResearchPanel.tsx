import {
  BrainCircuit,
  Clock3,
  FileSearch,
  FlaskConical,
  Layers3,
  ShieldCheck,
  Sparkles
} from "lucide-react";
import { delay, safeRows, shortNumber, statusClass, stringValue } from "../lib/api";

export function AIResearchPanel({ research }: { research: Record<string, unknown> }) {
  const counts = (research.counts ?? {}) as Record<string, unknown>;
  const queue = (research.queue ?? {}) as Record<string, unknown>;
  const queueCounts = (queue.counts ?? {}) as Record<string, unknown>;
  const latest = (research.latest_run ?? {}) as Record<string, unknown>;
  const status = stringValue(research.status, "WAITING_FOR_FIRST_RESULT");
  const findings = safeRows(research.findings);
  const factors = safeRows(research.factor_proposals);
  const paperDrafts = safeRows(research.paper_strategy_drafts);
  const experiments = safeRows(research.experiment_proposals);
  const codeTargets = safeRows(research.code_review_targets);
  const pending = Number(queueCounts.pending ?? 0);
  const running = Number(queueCounts.running ?? 0);

  return (
    <section className="card pad ai-research">
      <div className="ai-head">
        <div>
          <h2 className="section-title icon-title"><BrainCircuit size={24} />AI 研究工作台</h2>
          <p className="sub">NAS Worker · Stage 1 / Stage 2 · 人工复核</p>
        </div>
        <span className={`ai-status ${statusClass(status)}`}><ShieldCheck size={15} />{status.replace(/_/g, " ")}</span>
      </div>

      <div className="ai-metrics">
        <Metric icon={Layers3} label="研究运行" value={counts.run_count} />
        <Metric icon={FileSearch} label="诊断发现" value={counts.finding_count} />
        <Metric icon={Sparkles} label="因子草案" value={counts.factor_proposal_count} />
        <Metric icon={BrainCircuit} label="Paper 草案" value={counts.paper_draft_count} />
        <Metric icon={FlaskConical} label="实验草案" value={counts.experiment_count} />
        <Metric icon={Clock3} label="队列" value={`${pending} / ${running}`} sub="pending / running" />
      </div>

      <div className="ai-latest">
        <div>
          <span>最新任务</span>
          <b>{stringValue(latest.task_id, "尚无导入结果")}</b>
          <small>{stringValue(latest.model, "—")} · {stringValue(latest.reasoning_effort, "—")} · {delay(research.latest_run_age_seconds)}</small>
        </div>
        <div>
          <span>证据状态</span>
          <b>{stringValue(latest.system_state, "等待首个结果")}</b>
          <small>Stage 2 {latest.stage2_allowed === true ? "已运行" : "未运行"} · live effect {stringValue(research.live_order_effect, "none")}</small>
        </div>
      </div>

      <div className="ai-evidence-grid">
        <EvidenceList
          title="诊断发现"
          rows={findings}
          primary="summary"
          secondary="category"
          trailing="severity"
          empty="等待 Stage 1 诊断"
        />
        <EvidenceList
          title="因子研究草案"
          rows={factors}
          primary="factor_name"
          secondary="factor_family"
          trailing="proposal_state"
          empty="暂无因子草案"
        />
        <EvidenceList
          title="Paper / 实验草案"
          rows={paperDrafts.concat(experiments)}
          primary="draft_id"
          fallbackPrimary="proposal_id"
          secondary="hypothesis"
          trailing="mode"
          empty="暂无 Paper 或实验草案"
        />
        <EvidenceList
          title="代码复核目标"
          rows={codeTargets}
          primary="path_or_component"
          secondary="reason"
          trailing="priority"
          empty="暂无代码复核目标"
        />
      </div>
    </section>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  sub
}: {
  icon: typeof BrainCircuit;
  label: string;
  value: unknown;
  sub?: string;
}) {
  return (
    <article className="ai-metric">
      <Icon size={17} />
      <span>{label}</span>
      <b>{typeof value === "string" ? value : shortNumber(value)}</b>
      {sub ? <small>{sub}</small> : null}
    </article>
  );
}

function EvidenceList({
  title,
  rows,
  primary,
  fallbackPrimary,
  secondary,
  trailing,
  empty
}: {
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
          <article key={`${stringValue(row[primary], stringValue(row[fallbackPrimary ?? ""], "row"))}-${index}`}>
            <b>{stringValue(row[primary], stringValue(row[fallbackPrimary ?? ""], "—"))}</b>
            <small>{stringValue(row[secondary], "—")}</small>
            <em>{stringValue(row[trailing], "—")}</em>
          </article>
        )) : <p>{empty}</p>}
      </div>
    </section>
  );
}
