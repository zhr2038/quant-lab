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

export function ResearchComputeStatus({ status }: Props) {
  const task = objectValue(status.task);
  const state = stringValue(status.state, "idle");
  const normalized = state.toLowerCase();
  const tone = normalized === "completed"
    ? "ok"
    : ["rejected", "failed", "expired"].includes(normalized)
      ? "critical"
      : ["idle", "not_observable"].includes(normalized)
        ? "info"
        : "warning";
  const inputBytes = Number(task.input_bytes ?? 0);
  const downloadedBytes = Number(task.downloaded_bytes ?? 0);
  const cacheHitBytes = Number(task.cache_hit_bytes ?? 0);
  const cacheRate = inputBytes > 0 ? `${((cacheHitBytes / inputBytes) * 100).toFixed(1)}%` : "—";
  const error = stringValue(task.last_error ?? status.last_error, "");

  return (
    <section className="research-compute-panel">
      <div className="research-compute-head">
        <div>
          <h3><Cpu size={18} />Entry Quality History · NAS</h3>
          <p>research-only · live_order_effect: none</p>
        </div>
        <span className={`research-state ${tone}`}><Activity size={14} />{state}</span>
      </div>
      <div className="research-compute-grid">
        <div><span>任务</span><b title={stringValue(task.task_id)}>{stringValue(task.task_id)}</b></div>
        <div><span>Snapshot</span><b title={stringValue(task.snapshot_id)}>{stringValue(task.snapshot_id)}</b></div>
        <div><span>窗口</span><b>{stringValue(task.start_date)} → {stringValue(task.end_date)}</b></div>
        <div><span>口径</span><b>{stringValue(task.mode)} · {stringValue(task.cost_mode)}</b></div>
        <div><span>Worker</span><b>{stringValue(task.worker_id)}</b></div>
        <div><span>心跳</span><b><Clock3 size={12} />{timeValue(task.heartbeat_at)}</b></div>
        <div><span>输入 / 下载</span><b><Database size={12} />{bytes(inputBytes)} / {bytes(downloadedBytes)}</b></div>
        <div><span>缓存命中</span><b>{bytes(cacheHitBytes)} · {cacheRate}</b></div>
        <div><span>输出行</span><b>{stringValue(task.output_rows, "0")}</b></div>
        <div><span>Anti-Leakage</span><b><ShieldCheck size={12} />{stringValue(task.anti_leakage_status)}</b></div>
        <div><span>云端导入</span><b>{stringValue(task.import_status)}</b></div>
        <div><span>Gold generation</span><b title={stringValue(task.gold_generation_id)}>{stringValue(task.gold_generation_id)}</b></div>
      </div>
      {error ? <div className="research-compute-error">{error}</div> : null}
    </section>
  );
}
