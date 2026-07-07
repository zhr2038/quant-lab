import { AlertTriangle, CheckCircle2, Cpu, HardDrive, MemoryStick, Server, Siren } from "lucide-react";
import { stringValue } from "../lib/api";

function numberValue(value: unknown): number | null {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function percent(value: unknown): string {
  const n = numberValue(value);
  if (n === null) return "—";
  return `${n.toFixed(1)}%`;
}

function clampPercent(value: unknown): number {
  const n = numberValue(value);
  if (n === null) return 0;
  return Math.max(0, Math.min(100, n));
}

function bytes(value: unknown): string {
  const n = numberValue(value);
  if (n === null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let scaled = n;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  const digits = scaled >= 100 || unit === 0 ? 0 : 1;
  return `${scaled.toFixed(digits)}${units[unit]}`;
}

function duration(value: unknown): string {
  const seconds = numberValue(value);
  if (seconds === null) return "—";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function resourceStatus(resources: Record<string, unknown>): "ok" | "warning" | "critical" {
  const status = stringValue(resources.status, "OK").toUpperCase();
  if (status === "CRITICAL") return "critical";
  if (status === "WARNING") return "warning";
  return "ok";
}

export function ServerResourcesPanel({ resources }: { resources: Record<string, unknown> }) {
  const severity = resourceStatus(resources);
  const Icon = severity === "critical" ? Siren : severity === "warning" ? AlertTriangle : CheckCircle2;
  const badge = severity === "critical" ? "紧张" : severity === "warning" ? "关注" : "OK";
  const host = stringValue(resources.hostname, "qyun2");
  const diskPath = stringValue(resources.disk_path, "/var/lib/quant-lab");
  const cpuValue = resources.cpu_usage_percent ?? resources.cpu_load_percent ?? resources.cpu_display_percent;
  const rows = [
    {
      label: "CPU",
      value: percent(cpuValue),
      sub: `load1 ${stringValue(resources.load_1m, "—")} / ${stringValue(resources.cpu_count, "—")}c`,
      icon: Cpu,
      percent: clampPercent(cpuValue)
    },
    {
      label: "内存",
      value: percent(resources.memory_used_percent),
      sub: `可用 ${bytes(resources.memory_available_bytes)} / ${bytes(resources.memory_total_bytes)}`,
      icon: MemoryStick,
      percent: clampPercent(resources.memory_used_percent)
    },
    {
      label: "磁盘",
      value: percent(resources.disk_used_percent),
      sub: `空闲 ${bytes(resources.disk_free_bytes)} / ${bytes(resources.disk_total_bytes)}`,
      icon: HardDrive,
      percent: clampPercent(resources.disk_used_percent)
    }
  ];

  return (
    <section className={`card pad server-resources ${severity}`}>
      <div className="server-resource-head">
        <div>
          <h2 className="section-title icon-title">
            <Server size={22} />
            服务器资源
          </h2>
          <p className="sub">{host} · {diskPath}</p>
        </div>
        <div className={`server-resource-badge ${severity}`}>
          <Icon size={16} aria-hidden="true" />
          {badge}
        </div>
      </div>
      <div className="server-resource-list">
        {rows.map((row) => {
          const RowIcon = row.icon;
          return (
            <article key={row.label} className="server-resource-row">
              <RowIcon size={17} aria-hidden="true" />
              <div>
                <span>{row.label}</span>
                <strong>{row.value}</strong>
                <em>{row.sub}</em>
              </div>
              <div className="resource-meter" aria-hidden="true">
                <i style={{ width: `${row.percent}%` }} />
              </div>
            </article>
          );
        })}
      </div>
      <div className="server-resource-meta">
        <span><b>{duration(resources.uptime_seconds)}</b><em>uptime</em></span>
        <span><b>{stringValue(resources.load_5m, "—")}</b><em>load 5m</em></span>
        <span><b>{percent(resources.swap_used_percent)}</b><em>swap</em></span>
      </div>
    </section>
  );
}
