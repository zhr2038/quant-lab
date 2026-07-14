import { z } from "zod";

export const StatusSchema = z.enum(["OK", "WARNING", "CRITICAL", "UNKNOWN", "INFO"]);
export type Status = z.infer<typeof StatusSchema>;

export const SnapshotSchema = z.object({
  generated_at: z.string(),
  lake_root: z.string(),
  mode: z.literal("read-only"),
  status: z.string(),
  health_score: z.number().min(0).max(100),
  kpis: z.record(z.string(), z.unknown()).default({}),
  actions: z.array(z.record(z.string(), z.unknown())).default([]),
  data_matrix: z.object({
    columns: z.array(z.string()),
    rows: z.array(z.record(z.string(), z.unknown()))
  }).default({ columns: [], rows: [] }),
  strategy_flow: z.record(z.string(), z.unknown()).default({}),
  ai_research: z.record(z.string(), z.unknown()).default({}),
  v5: z.record(z.string(), z.unknown()).default({}),
  cost: z.record(z.string(), z.unknown()).default({}),
  market: z.record(z.string(), z.unknown()).default({}),
  collectors: z.record(z.string(), z.unknown()).default({}),
  data_health: z.record(z.string(), z.unknown()).default({}),
  legacy_anomalies: z.record(z.string(), z.unknown()).default({}),
  server_resources: z.record(z.string(), z.unknown()).default({}),
  web_perf: z.record(z.string(), z.unknown()).default({}),
  consumers: z.record(z.string(), z.unknown()).default({}),
  exports: z.record(z.string(), z.unknown()).default({}),
  warnings: z.array(z.string()).default([])
});

export type BigscreenSnapshot = z.infer<typeof SnapshotSchema>;

export const ExpertPackStatusSchema = z.object({
  mode: z.string().default("read_only_export"),
  live_order_effect: z.string().default("none"),
  export_date: z.string(),
  exports_root: z.string().optional(),
  state: z.string(),
  status: z.record(z.string(), z.unknown()).default({}),
  requested_date_pack: z.string().nullable().optional(),
  requested_date_pack_name: z.string().nullable().optional(),
  available_pack: z.string().nullable().optional(),
  available_pack_name: z.string().nullable().optional(),
  available_download_url: z.string().nullable().optional(),
  manual_latest_pack: z.string().nullable().optional(),
  manual_latest_pack_name: z.string().nullable().optional(),
  manual_latest_download_url: z.string().nullable().optional(),
  latest_pack: z.string().nullable().optional(),
  latest_pack_name: z.string().nullable().optional(),
  latest_download_url: z.string().nullable().optional(),
  latest_size_bytes: z.number().nullable().optional(),
  latest_modified_at: z.string().nullable().optional(),
  latest_pack_quant_lab_git_commit: z.string().nullable().optional(),
  current_quant_lab_git_commit: z.string().nullable().optional(),
  latest_pack_matches_current_quant_lab_commit: z.boolean().nullable().optional(),
  latest_pack_code_lag_status: z.string().nullable().optional(),
  regenerate_cooldown_seconds: z.number().nullable().optional(),
  regenerate_cooldown_remaining_seconds: z.number().nullable().optional(),
  regenerate_available_at: z.string().nullable().optional(),
  regenerate_reuse_pack_name: z.string().nullable().optional(),
  packs: z.array(z.record(z.string(), z.unknown())).default([]),
  pack_count: z.number().default(0)
});

export type ExpertPackStatus = z.infer<typeof ExpertPackStatusSchema>;

function apiBase(): string {
  return import.meta.env.VITE_QUANT_LAB_API_BASE ?? "";
}

function tokenHeaders(): HeadersInit | undefined {
  const token = import.meta.env.VITE_QUANT_LAB_API_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : undefined;
}

export async function fetchBigscreenSnapshot(): Promise<BigscreenSnapshot> {
  const base = apiBase();
  const path = import.meta.env.VITE_QUANT_LAB_SNAPSHOT_PATH ?? "/web-v2/snapshot";
  const res = await fetch(`${base}${path}`, {
    headers: tokenHeaders(),
    cache: "no-store"
  });
  if (!res.ok) {
    throw new Error(`bigscreen snapshot failed: ${res.status}`);
  }
  return SnapshotSchema.parse(await res.json());
}

export async function triggerExpertPackGenerate(): Promise<ExpertPackStatus> {
  const res = await fetch(`${apiBase()}/web-v2/expert-pack/generate`, {
    method: "POST",
    headers: tokenHeaders(),
    cache: "no-store"
  });
  if (!res.ok) {
    throw new Error(`expert pack generate failed: ${res.status}`);
  }
  return ExpertPackStatusSchema.parse(await res.json());
}

export async function fetchExpertPackStatus(): Promise<ExpertPackStatus> {
  const res = await fetch(`${apiBase()}/web-v2/expert-pack/status`, {
    headers: tokenHeaders(),
    cache: "no-store"
  });
  if (!res.ok) {
    throw new Error(`expert pack status failed: ${res.status}`);
  }
  return ExpertPackStatusSchema.parse(await res.json());
}

export function expertPackDownloadUrl(nameOrUrl: unknown): string {
  const text = stringValue(nameOrUrl, "");
  if (!text) return "#";
  if (text.startsWith("/web-v2/expert-pack/download/")) return `${apiBase()}${text}`;
  return `${apiBase()}/web-v2/expert-pack/download/${encodeURIComponent(text)}`;
}

export function statusClass(value: unknown): "ok" | "warning" | "critical" | "info" {
  const text = String(value ?? "").toUpperCase();
  if (["OK", "ALLOW", "FRESH", "PASS", "TRUE"].includes(text)) return "ok";
  if (["WARNING", "DELAYED", "UNKNOWN", "SELL_ONLY"].includes(text)) return "warning";
  if (["CRITICAL", "FAIL", "STALE", "ABORT", "KILL", "FALSE"].includes(text)) return "critical";
  return "info";
}

export function statusText(value: unknown): string {
  const text = String(value ?? "").toUpperCase();
  if (["OK", "ALLOW", "FRESH", "PASS", "TRUE", "RUNNING"].includes(text)) return "正常";
  if (["WARNING", "DELAYED", "UNKNOWN", "SELL_ONLY"].includes(text)) return "注意";
  if (["CRITICAL", "FAIL", "STALE", "ABORT", "KILL", "FALSE"].includes(text)) return "异常";
  if (!text || text === "NULL" || text === "NONE") return "未知";
  return String(value);
}

export function permissionDisplay(value: unknown): { value: string; sub: string; tone: string } {
  const text = String(value ?? "UNKNOWN").toUpperCase();
  if (text === "ACTIVE_ABORT") {
    return { value: "只读 ABORT", sub: "advisory 未强制", tone: "WARNING" };
  }
  if (text === "ACTIVE_SELL_ONLY") {
    return { value: "只读 SELL", sub: "advisory 未强制", tone: "WARNING" };
  }
  return {
    value: text.replace(/_/g, " "),
    sub: "只读 advisory",
    tone: text
  };
}

export function pct(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

export function shortNumber(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function bps(value: unknown, digits = 1): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${n.toFixed(digits)}bps`;
}

export function ms(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${Math.round(n)}ms`;
}

export function delay(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "—";
  if (n < 60) return `${Math.round(n)}s`;
  if (n < 3600) return `${Math.round(n / 60)}m`;
  return `${(n / 3600).toFixed(1)}h`;
}

export function safeRows(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((row): row is Record<string, unknown> => !!row && typeof row === "object" && !Array.isArray(row)) : [];
}

export function stringValue(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}
