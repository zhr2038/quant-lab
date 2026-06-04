import ReactECharts from "echarts-for-react";
import { ShieldPlus } from "lucide-react";

export function HealthPanel({
  score,
  status,
  warnings
}: {
  score: number;
  status: string;
  warnings: string[];
}) {
  const color = status === "CRITICAL" ? "#FF5D7D" : status === "WARNING" ? "#FFC457" : "#2DE8A6";
  const option = {
    backgroundColor: "transparent",
    animationDuration: 900,
    series: [
      {
        type: "gauge",
        startAngle: 90,
        endAngle: -270,
        min: 0,
        max: 100,
        progress: { show: true, width: 18, roundCap: true, itemStyle: { color } },
        axisLine: { lineStyle: { width: 18, color: [[1, "rgba(93,132,178,.34)"]] } },
        pointer: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        detail: {
          valueAnimation: true,
          formatter: "{value}",
          color: "#eaf6ff",
          fontSize: 58,
          fontWeight: 900,
          offsetCenter: [0, 0]
        },
        data: [{ value: score }]
      }
    ]
  };
  return (
    <section className="card status-ring pad">
      <h2 className="section-title icon-title"><ShieldPlus size={24} />系统主状态</h2>
      <p className="sub">综合数据新鲜度 / 成本可信 / V5 遥测 / 消费者权限</p>
      <div className="ring-chart">
        <ReactECharts option={option} style={{ height: 208, width: 230 }} />
        <span>HEALTH SCORE</span>
      </div>
      <div className={`health-pill ${status.toLowerCase()}`}>{status} · {warnings.length} 个注意项</div>
    </section>
  );
}
