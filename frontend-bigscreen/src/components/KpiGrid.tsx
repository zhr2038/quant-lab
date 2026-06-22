import { motion } from "framer-motion";
import { Activity, Gauge, RadioTower, ShieldCheck, Timer, WalletCards, Waves, Zap } from "lucide-react";
import { delay, ms, pct, shortNumber, type BigscreenSnapshot } from "../lib/api";

export function KpiGrid({ snapshot }: { snapshot: BigscreenSnapshot }) {
  const k = snapshot.kpis;
  const v5Permission = permissionDisplay(k.v5_permission);
  const cards = [
    { label: "中台状态", value: String(k.platform_status ?? snapshot.status), sub: "风控边界", icon: ShieldCheck, tone: snapshot.status },
    { label: "V5 权限", value: v5Permission.value, sub: v5Permission.sub, icon: WalletCards, tone: v5Permission.tone },
    { label: "行情延迟", value: delay(k.market_delay_seconds), sub: "market_bar", icon: Timer, tone: Number(k.market_delay_seconds) > 1800 ? "WARNING" : "OK" },
    { label: "V5 Bundle", value: delay(k.v5_bundle_delay_seconds), sub: "latest bundle", icon: Activity, tone: Number(k.v5_bundle_delay_seconds) > 3600 ? "WARNING" : "OK" },
    { label: "硬回退", value: pct(k.cost_hard_fallback_ratio, 1), sub: "global default", icon: Zap, tone: Number(k.cost_hard_fallback_ratio) > 0.25 ? "CRITICAL" : "OK" },
    { label: "软回退", value: pct(k.cost_soft_fallback_ratio, 1), sub: "public proxy", icon: Waves, tone: Number(k.cost_soft_fallback_ratio) > 0.8 ? "WARNING" : "OK" },
    { label: "WS 消息", value: shortNumber(k.ws_message_count), sub: "books / trades", icon: RadioTower, tone: "INFO" },
    { label: "API P95", value: ms(k.api_p95_ms), sub: "真实接口", icon: Gauge, tone: Number(k.api_p95_ms) > 800 ? "WARNING" : "OK" }
  ];
  return (
    <section className="card kpis">
      <h2 className="section-title">核心指标</h2>
      {cards.map((card, i) => {
        const Icon = card.icon;
        const displayValue = card.value.replace(/_/g, " ");
        const isLongValue = displayValue.length > 8;
        return (
          <motion.div
            className={`kpi ${toneClass(card.tone)}${isLongValue ? " long-value" : ""}`}
            key={card.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.045, duration: 0.45 }}
          >
            <Icon size={18} aria-hidden="true" />
            <label>{card.label}</label>
            <strong title={card.value}>{displayValue}</strong>
            <span>{card.sub}</span>
          </motion.div>
        );
      })}
    </section>
  );
}

function permissionDisplay(value: unknown): { value: string; sub: string; tone: string } {
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

function toneClass(value: unknown): string {
  const text = String(value ?? "").toUpperCase();
  if (["CRITICAL", "ABORT", "KILL", "FAIL"].includes(text)) return "critical";
  if (["WARNING", "SELL_ONLY", "UNKNOWN"].includes(text)) return "warn";
  if (["OK", "ALLOW", "RUNNING"].includes(text)) return "ok";
  return "info";
}
