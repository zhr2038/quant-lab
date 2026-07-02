import { CircleDollarSign } from "lucide-react";
import { pct, shortNumber } from "../lib/api";
import { ReactECharts } from "./EChart";

export function CostQuality({ cost }: { cost: Record<string, unknown> }) {
  const basis = String(cost.cost_quality_basis ?? "");
  const unitLabel = basis === "effective_symbol_source" ? "标的" : "rows";
  const rows = [
    { name: "真实", value: Number(cost.actual_rows ?? 0), color: "#2DE8A6" },
    { name: "混合", value: Number(cost.mixed_rows ?? 0), color: "#50A9FF" },
    { name: "探针", value: Number(cost.bootstrap_probe_rows ?? 0), color: "#A77DFF" },
    { name: "代理", value: Number(cost.proxy_rows ?? 0), color: "#FFC457" },
    { name: "全局默认", value: Number(cost.global_default_rows ?? 0), color: "#FF5D7D" }
  ];
  const total = rows.reduce((sum, row) => sum + row.value, 0);
  const option = {
    backgroundColor: "transparent",
    color: rows.map((row) => row.color),
    series: [
      {
        type: "pie",
        radius: ["62%", "78%"],
        label: { color: "#eaf6ff" },
        data: rows
      }
    ],
    graphic: [
      {
        type: "text",
        left: "center",
        top: "42%",
        style: {
          text: `${shortNumber(total)}\n${unitLabel}`,
          fill: "#eaf6ff",
          fontSize: 22,
          fontWeight: 900,
          lineHeight: 25,
          align: "center"
        }
      }
    ]
  };
  return (
    <section className="card cost pad">
      <h2 className="section-title icon-title"><CircleDollarSign size={23} />成本质量</h2>
      <p className="sub">按当前 symbol 最高质量来源汇总；bootstrap probe 不作 live 覆盖。</p>
      <div className="cost-body">
        <ReactECharts option={option} style={{ height: 168, width: 168 }} />
        <div>
          {rows.map((row) => <Bar key={row.name} label={`${row.name}成本`} value={row.value} total={Math.max(total, 1)} color={row.color} />)}
        </div>
      </div>
      <div className="footnote yellowText">硬回退 {pct(cost.hard_fallback_ratio)} · 软回退 {pct(cost.soft_fallback_ratio)} · 探针只作 bootstrap，不作 live 覆盖</div>
    </section>
  );
}

function Bar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  return (
    <div className="barline">
      <span>{label}</span>
      <div className="bar"><span style={{ width: `${Math.min(100, (value / total) * 100)}%`, background: color }} /></div>
      <b>{value}</b>
    </div>
  );
}
