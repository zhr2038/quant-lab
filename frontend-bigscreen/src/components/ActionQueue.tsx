import { AlertTriangle, CheckCircle2, CircleDot, Siren } from "lucide-react";

export function ActionQueue({ actions }: { actions: Record<string, unknown>[] }) {
  return (
    <section className="card pad action-queue">
      <h2 className="section-title">行动队列</h2>
      {actions.slice(0, 4).map((action, i) => {
        const severity = String(action.severity ?? "INFO").toUpperCase();
        const Icon = severity === "CRITICAL" ? Siren : severity === "WARNING" ? AlertTriangle : severity === "OK" ? CheckCircle2 : CircleDot;
        return (
          <article key={`${action.title}-${i}`} className={`queue-item ${severity.toLowerCase()}`}>
            <Icon size={17} aria-hidden="true" />
            <b>{String(action.title ?? "未命名行动")}</b>
            <span>{String(action.next_action ?? action.summary ?? "")}</span>
          </article>
        );
      })}
    </section>
  );
}
