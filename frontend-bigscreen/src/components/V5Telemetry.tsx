import { Cpu, Radio } from "lucide-react";
import { permissionDisplay } from "../lib/api";
import { ReactECharts } from "./EChart";

export function V5Telemetry({
  v5,
  consumers
}: {
  v5: Record<string, unknown>;
  consumers: Record<string, unknown>;
}) {
  const option = {
    backgroundColor: "transparent",
    radar: {
      center: ["50%", "52%"],
      radius: "40%",
      splitNumber: 3,
      axisNameGap: 8,
      axisName: { color: "#8aa4be", fontSize: 11, lineHeight: 13, overflow: "break", width: 68 },
      splitLine: { lineStyle: { color: "rgba(80,169,255,.22)" } },
      splitArea: { areaStyle: { color: ["rgba(80,169,255,.04)", "rgba(80,169,255,.08)"] } },
      axisLine: { lineStyle: { color: "rgba(80,169,255,.22)" } },
      indicator: [
        { name: "遥测批次", max: 1 },
        { name: "reconcile", max: 1 },
        { name: "ledger", max: 1 },
        { name: "risk", max: 1 },
        { name: "config", max: 1 },
        { name: "position", max: 1 }
      ]
    },
    series: [
      {
        type: "radar",
        data: [
          {
            value: [
              v5.latest_bundle_ts ? 0.96 : 0.25,
              v5.reconcile_ok === false ? 0.15 : 1,
              v5.ledger_ok === false ? 0.15 : 1,
              Number(v5.high_issue_count ?? 0) > 0 ? 0.35 : 0.92,
              Number(v5.config_not_consumed_count ?? 0) > 0 ? 0.55 : 0.96,
              Number(v5.open_position_count ?? 0) > 0 ? 0.85 : 1
            ],
            areaStyle: { color: "rgba(80,169,255,.52)" },
            lineStyle: { color: "#50A9FF" }
          }
        ]
      }
    ]
  };
  const permissions = (consumers.permissions ?? {}) as Record<string, unknown>;
  const p3 = (v5.cost_probe_p3_preflight ?? {}) as Record<string, unknown>;
  const p3State = String(p3.state ?? "not_observable");
  const p3Ready = p3.ready_to_request_manual_live_probe === true ? "可申请" : "未就绪";
  const p3Approved = p3.approved_live_order_execution === true ? "已批准" : "未批准";
  const v5Permission = permissionDisplay(permissions.v5);
  const items = [
    ["72h 运行", v5.run_count_72h],
    ["24h 决策", v5.decision_audit_count_24h],
    ["交易笔数", v5.trade_count_24h],
    ["持仓", v5.open_position_count],
    ["kill-switch", v5.kill_switch_enabled ? "ON" : "OFF"],
    ["V5 权限", v5Permission.value]
  ];
  return (
    <section className="card v5 pad">
      <h2 className="section-title icon-title"><Cpu size={23} />V5 遥测与消费者</h2>
      <p className="sub">遥测批次、账本、对账、风控权限、blocked opportunity 合并展示。</p>
      <div className="v5-grid">
        <div className="v5-radar" aria-label="V5 telemetry radar">
          <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />
        </div>
        <div className="metric-list">
          {items.map(([key, value]) => <div className="mini" key={String(key)}><span>{String(key)}</span><strong>{String(value ?? "—")}</strong></div>)}
        </div>
      </div>
      <div className="v5-p3-strip">
        <span>P3 {p3State}</span>
        <span>人工授权 {p3Ready}</span>
        <span>实盘 {p3Approved}</span>
      </div>
      <div className="footnote"><Radio size={12} /> 最新遥测 SHA · <code>{String(v5.latest_bundle_sha256_short ?? "not_observable")}</code></div>
    </section>
  );
}
