import { AlertTriangle, CheckCircle2, CircleDot, Siren } from "lucide-react";

export function ActionQueue({ actions }: { actions: Record<string, unknown>[] }) {
  return (
    <section className="card pad action-queue">
      <h2 className="section-title">行动队列</h2>
      <div className="action-list" aria-label="当前行动项">
        {actions.slice(0, 4).map((action, i) => {
          const severity = String(action.severity ?? "INFO").toUpperCase();
          const title = String(action.title ?? "未命名行动");
          const detail = String(action.next_action ?? action.summary ?? "");
          const Icon = severity === "CRITICAL" ? Siren : severity === "WARNING" ? AlertTriangle : severity === "OK" ? CheckCircle2 : CircleDot;
          return (
            <article key={`${action.title}-${i}`} className={`queue-item ${severity.toLowerCase()}`} title={`${title} · ${detail}`}>
              <Icon size={17} aria-hidden="true" />
              <b>{title}</b>
              <span>{detail}</span>
            </article>
          );
        })}
      </div>
    </section>
  );
}
