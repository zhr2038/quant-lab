import { Activity, Clock3, Cpu, Database, ShieldCheck } from "lucide-react";
import { stringValue } from "../lib/api";

type Props = {
  status: Record<string, unknown>;
};

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function bytes(value: unknown): string {
  const number = Number(value ?? 0);
  if (!Number.isFinite(number) || number <= 0) return "0 B";
  if (number >= 1024 ** 3) return `${(number / 1024 ** 3).toFixed(1)} GB`;
  if (number >= 1024 ** 2) return `${(number / 1024 ** 2).toFixed(1)} MB`;
  if (number >= 1024) return `${(number / 1024).toFixed(1)} KB`;
  return `${Math.round(number)} B`;
}

function timeValue(value: unknown): string {
  const raw = stringValue(value, "");
  if (!raw) return "—";
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return raw;
  return parsed.toLocaleString("zh-CN", { hour12: false });
}

function durationValue(value: unknown): string {
  const seconds = Number(value ?? 0);
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)} h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)} min`;
  return `${seconds.toFixed(1)} s`;
}

function stateTone(state: string): string {
  const normalized = state.toLowerCase();
  if (["completed", "up_to_date", "already_current", "already_current_no_update"].includes(normalized)) return "ok";
  if (["rejected", "failed", "expired"].includes(normalized)) return "critical";
  if (["idle", "not_observable"].includes(normalized)) return "info";
  return "warning";
}

function TaskStatus({ label, status }: { label: string; status: Record<string, unknown> }) {
  const task = objectValue(status.task);
  const state = stringValue(status.state, "idle");
  const inputBytes = Number(task.input_bytes ?? 0);
  const downloadedBytes = Number(task.downloaded_bytes ?? 0);
  const cacheHitBytes = Number(task.cache_hit_bytes ?? 0);
  const cacheRate = inputBytes > 0 ? `${((cacheHitBytes / inputBytes) * 100).toFixed(1)}%` : "—";
  const error = stringValue(task.last_error ?? status.last_error, "");
  const isFactorFactory = stringValue(task.task_type) === "factor_factory"
    || status.request_outcome !== undefined;
  const request = objectValue(status.request);
  const inputFingerprint = objectValue(status.input_fingerprint ?? request.input_fingerprint);
  const requestOutcome = stringValue(status.request_outcome ?? request.request_outcome, "—");
  const fingerprintDigest = stringValue(inputFingerprint.combined_input_digest, "");
  const fingerprintMatches = Boolean(status.fingerprint_matches_generation);
  const payloadState = stringValue(status.snapshot_payload_state, "—");
  const snapshotMaterialized = Boolean(status.snapshot_materialized);
  const snapshotRehydrated = Boolean(status.snapshot_rehydrated);
  const noUpdateReason = stringValue(status.no_update_reason, "");

  return (
    <div className="research-compute-task">
      <div className="research-task-head">
        <b>{label}</b>
        <span className={`research-state ${stateTone(state)}`}><Activity size={13} />{state}</span>
      </div>
      <div className="research-compute-grid">
        <div><span>任务</span><b title={stringValue(task.task_id)}>{stringValue(task.task_id, "—")}</b></div>
        <div><span>Snapshot</span><b title={stringValue(task.snapshot_id)}>{stringValue(task.snapshot_id, "—")}</b></div>
        <div><span>窗口</span><b>{stringValue(task.start_date, "—")} → {stringValue(task.end_date, "—")}</b></div>
        <div><span>Worker / 心跳</span><b><Clock3 size={12} />{stringValue(task.worker_id, "—")} · {timeValue(task.worker_heartbeat_at ?? task.heartbeat_at)}</b></div>
        <div><span>输入 / 下载</span><b><Database size={12} />{bytes(inputBytes)} / {bytes(downloadedBytes)}</b></div>
        <div><span>缓存命中</span><b>{bytes(cacheHitBytes)} · {cacheRate}</b></div>
        <div><span>输出</span><b>{stringValue(task.output_rows, "0")} 行 · {bytes(task.output_bytes)}</b></div>
        <div><span>峰值内存 / 计算</span><b>{bytes(task.peak_rss_bytes)} · {durationValue(task.compute_duration_seconds)}</b></div>
        <div><span>Anti-Leakage</span><b><ShieldCheck size={12} />{stringValue(task.anti_leakage_status, "—")}</b></div>
        <div><span>云端导入</span><b>{stringValue(task.import_status, "—")}</b></div>
        <div><span>Gold generation</span><b title={stringValue(task.gold_generation_id ?? task.generation_id)}>{stringValue(task.gold_generation_id ?? task.generation_id, "—")}</b></div>
        {isFactorFactory ? <>
          <div><span>Factor Plan</span><b title={stringValue(task.factor_plan_digest)}>{stringValue(task.factor_plan_digest, "—")}</b></div>
          <div><span>Scope</span><b>{stringValue(task.feature_set, "—")} / {stringValue(task.feature_version, "—")} → {stringValue(task.factor_version, "—")} · {stringValue(task.timeframe, "—")}</b></div>
          <div><span>Horizon / Factors</span><b>{Array.isArray(task.horizon_bars) ? task.horizon_bars.join(",") : "—"} · {stringValue(task.factor_count, "0")}</b></div>
          <div><span>Value / Evidence / Corr</span><b>{stringValue(task.value_rows, "0")} / {stringValue(task.evidence_rows, "0")} / {stringValue(task.correlation_rows, "0")}</b></div>
          <div><span>Generation age</span><b>{durationValue(task.generation_age_seconds)}</b></div>
          <div><span>Request / Fingerprint</span><b>{requestOutcome} · {fingerprintMatches ? "match" : "not matched"}</b></div>
          <div><span>Input fingerprint</span><b title={fingerprintDigest}>{fingerprintDigest ? fingerprintDigest.slice(0, 16) : "—"}</b></div>
          <div><span>Snapshot payload</span><b>{payloadState} · materialized={String(snapshotMaterialized)} · rehydrated={String(snapshotRehydrated)}</b></div>
          <div><span>压缩 / 估算未压缩输入</span><b>{bytes(status.compressed_input_bytes)} / {bytes(status.estimated_uncompressed_input_bytes)}</b></div>
          <div><span>Already current</span><b>{timeValue(status.already_current_at)}{noUpdateReason ? ` · ${noUpdateReason}` : ""}</b></div>
        </> : null}
      </div>
      {error ? <div className="research-compute-error">{error}</div> : null}
    </div>
  );
}

export function ResearchComputeStatus({ status }: Props) {
  const tasks = objectValue(status.tasks);
  const entryQuality = objectValue(tasks.entry_quality_history);
  const alphaFactory = objectValue(tasks.alpha_factory);
  const factorResearch = objectValue(tasks.factor_research);
  const factorFactory = objectValue(tasks.factor_factory);
  const aggregateState = stringValue(status.state, "idle");

  return (
    <section className="research-compute-panel">
      <div className="research-compute-head">
        <div>
          <h3><Cpu size={18} />NAS Research Compute</h3>
          <p>共享 Research Plane · research-only · live_order_effect: none</p>
        </div>
        <span className={`research-state ${stateTone(aggregateState)}`}><Activity size={14} />{aggregateState}</span>
      </div>
      <div className="research-task-list">
        <TaskStatus label="Entry Quality History" status={entryQuality} />
        <TaskStatus label="Alpha Factory" status={alphaFactory} />
        <TaskStatus label="Factor Research v2" status={factorResearch} />
        <TaskStatus label="Factor Factory" status={factorFactory} />
      </div>
    </section>
  );
}
