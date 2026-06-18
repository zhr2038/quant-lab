import { AlertTriangle, CheckCircle2, DatabaseZap, Siren } from "lucide-react";
import { safeRows, shortNumber, stringValue } from "../lib/api";

function numberValue(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : 0;
}

export function LegacyAnomalyPanel({ anomalies }: { anomalies: Record<string, unknown> }) {
  const items = safeRows(anomalies.items);
  const total = numberValue(anomalies.total_count);
  const critical = numberValue(anomalies.critical_count);
  const warnings = numberValue(anomalies.warning_count);
  const hasAnomalies = Boolean(anomalies.has_anomalies) || total > 0;
  const Icon = critical > 0 ? Siren : hasAnomalies ? AlertTriangle : CheckCircle2;
  const severity = critical > 0 ? "critical" : hasAnomalies ? "warning" : "ok";
  const title = hasAnomalies ? "旧页面异常" : "旧页面同步正常";
  const subtitle = hasAnomalies
    ? "Streamlit 数据健康 / overview 异常同步到新首页"
    : "Streamlit 数据健康 / overview 当前口径一致";

  return (
    <section className={`card pad legacy-anomalies ${severity}`}>
      <div className="legacy-anomaly-head">
        <div>
          <h2 className="section-title icon-title">
            <DatabaseZap size={22} />
            {title}
          </h2>
          <p className="sub">{subtitle}</p>
        </div>
        <div className={`legacy-anomaly-badge ${severity}`}>
          <Icon size={16} aria-hidden="true" />
          {hasAnomalies ? `${shortNumber(total)} 项` : "OK"}
        </div>
      </div>
      <div className="legacy-anomaly-stats">
        <span>
          <b>{shortNumber(critical)}</b>
          <em>CRITICAL</em>
        </span>
        <span>
          <b>{shortNumber(warnings)}</b>
          <em>WARNING</em>
        </span>
        <span>
          <b>{shortNumber(anomalies.stale_dataset_count)}</b>
          <em>过期/缺失</em>
        </span>
      </div>
      <div className="legacy-anomaly-list">
        {items.length ? (
          items.slice(0, 3).map((item, i) => {
            const itemSeverity = stringValue(item.severity, "INFO").toLowerCase();
            return (
              <article key={`${item.source}-${i}`} className={`legacy-anomaly-row ${itemSeverity}`}>
                <strong>{stringValue(item.title, "旧页面异常")}</strong>
                <span>{stringValue(item.summary, "")}</span>
                <small>{stringValue(item.next_action, "查看数据健康页明细")}</small>
              </article>
            );
          })
        ) : (
          <div className="legacy-anomaly-empty">
            老页面当前没有可见异常，新旧首页口径一致。
          </div>
        )}
      </div>
    </section>
  );
}
