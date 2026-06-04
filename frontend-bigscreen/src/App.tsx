import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import {
  Archive,
  BarChart3,
  Boxes,
  Database,
  DownloadCloud,
  Gauge,
  LayoutDashboard,
  MonitorDot,
  RadioTower,
  ShieldCheck,
  Table2,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ActionQueue } from "./components/ActionQueue";
import { CostQuality } from "./components/CostQuality";
import { DataMatrix } from "./components/DataMatrix";
import { HealthPanel } from "./components/HealthPanel";
import { KpiGrid } from "./components/KpiGrid";
import { MarketLiquidity } from "./components/MarketLiquidity";
import { PerfConsumers } from "./components/PerfConsumers";
import { StrategyFlow } from "./components/StrategyFlow";
import { V5Telemetry } from "./components/V5Telemetry";
import { fetchBigscreenSnapshot, safeRows, stringValue, type BigscreenSnapshot } from "./lib/api";
import "./styles.css";

const queryClient = new QueryClient();
type ViewKey = "strategy" | "data" | "v5" | "exports" | "raw";

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Bigscreen />
    </QueryClientProvider>
  );
}

function Bigscreen() {
  const [view, setView] = useState<ViewKey | null>(null);
  useEffect(() => {
    const fit = () => {
      const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
      document.documentElement.style.setProperty("--screen-scale", String(scale));
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);
  const query = useQuery({
    queryKey: ["bigscreen-snapshot"],
    queryFn: fetchBigscreenSnapshot,
    refetchInterval: 30_000,
    staleTime: 25_000,
    retry: 1
  });
  const data = query.data;
  return (
    <div className="viewport">
      <AnimatedBackground />
      <main className="app-shell">
        <header className="header">
          <div className="title">
            <h1><MonitorDot size={34} />quant-lab CONTROL CENTER</h1>
            <p>只读研究中台 · 大屏战情室 · 不下单 / 不撤单 / 不改状态</p>
          </div>
          <div className="pills">
            <span className="pill green">READ ONLY</span>
            <span className={`pill ${data?.status?.toLowerCase() ?? "yellow"}`}>{data?.status ?? "LOADING"}</span>
            <span className="pill">REFRESH 30s · Asia/Shanghai</span>
          </div>
        </header>

        <AnimatePresence mode="popLayout">
          {!data ? (
            <motion.section className="loading-card" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              {query.isError ? "Snapshot 加载失败：检查 /web-v2/snapshot 或 API token" : "加载 bigscreen snapshot…"}
            </motion.section>
          ) : (
            <Dashboard data={data} view={view} setView={setView} />
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function Dashboard({
  data,
  view,
  setView
}: {
  data: BigscreenSnapshot;
  view: ViewKey | null;
  setView: (view: ViewKey | null) => void;
}) {
  return (
    <motion.div className="layout" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
      <section className="left">
        <HealthPanel score={data.health_score} status={data.status} warnings={data.warnings} />
        <ActionQueue actions={data.actions} />
        <FunctionMap setView={setView} />
      </section>
      <KpiGrid snapshot={data} />
      <DataMatrix matrix={data.data_matrix} />
      <StrategyFlow flow={data.strategy_flow} />
      <V5Telemetry v5={data.v5} consumers={data.consumers} />
      <CostQuality cost={data.cost} />
      <MarketLiquidity market={data.market} />
      <PerfConsumers perf={data.web_perf} consumers={data.consumers} />
      <Drilldown view={view} data={data} onClose={() => setView(null)} />
    </motion.div>
  );
}

function FunctionMap({ setView }: { setView: (view: ViewKey) => void }) {
  const rows: Array<[ViewKey, string, string, typeof LayoutDashboard]> = [
    ["strategy", "策略", "研究组合 / advisory / Alpha Factory / 扩展币池", BarChart3],
    ["data", "数据运维", "市场状态 / OKX 采集器 / lake freshness", Database],
    ["v5", "消费性能", "风险权限 / API metrics / Web reader / V5", Gauge],
    ["exports", "专家包", "导出 / 下载 / manifest / data quality", DownloadCloud],
    ["raw", "原始表格", "保留旧 Streamlit 明细入口", Table2]
  ];
  return (
    <section className="card pad compact-map">
      <h2 className="section-title icon-title"><Boxes size={23} />功能整合地图</h2>
      <p className="sub">旧 10 页 → 新 5 个区域，主屏不再逐页翻。</p>
      {rows.map(([key, label, text, Icon]) => (
        <button className="map-row" key={key} onClick={() => setView(key)}>
          <Icon size={15} />
          <b>{label}</b>
          <span>{text}</span>
        </button>
      ))}
    </section>
  );
}

function Drilldown({
  view,
  data,
  onClose
}: {
  view: ViewKey | null;
  data: BigscreenSnapshot;
  onClose: () => void;
}) {
  if (!view) return null;
  const content = drilldownContent(view, data);
  return (
    <motion.aside className="detail-drawer" initial={{ opacity: 0, x: 42 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 42 }}>
      <div className="drawer-head">
        <h2>{content.title}</h2>
        <button onClick={onClose} aria-label="关闭二级页"><X size={20} /></button>
      </div>
      <p>{content.subtitle}</p>
      {content.blocks}
    </motion.aside>
  );
}

function drilldownContent(view: ViewKey, data: BigscreenSnapshot) {
  if (view === "strategy") {
    const candidates = safeRows(data.strategy_flow.top_candidates);
    const alphaFactory = safeRows(data.strategy_flow.alpha_factory);
    const riskOn = safeRows(data.strategy_flow.risk_on_multi_buy);
    return {
      title: "策略驾驶舱",
      subtitle: "聚合 advisory、Alpha Factory、risk-on multi-buy 和研究组合；全部为 read-only。",
      blocks: (
        <>
          <MiniTable title="最可能上线候选" rows={safeRows(data.strategy_flow.top_live_candidates).concat(candidates).slice(0, 8)} columns={["strategy_candidate", "symbol", "recommended_mode", "decision", "avg_net_bps", "p25_net_bps"]} />
          <MiniTable title="Alpha Factory" rows={alphaFactory} columns={["strategy_candidate", "promotion_state", "recommended_mode", "alpha_factory_score", "horizon_hours"]} />
          <MiniTable title="Risk-on multi-buy shadow" rows={riskOn} columns={["selected_symbols", "would_buy_symbols", "top_k", "decision_ts"]} />
        </>
      )
    };
  }
  if (view === "data") {
    return {
      title: "数据运维",
      subtitle: "市场状态、OKX 采集器、stale/missing datasets 与 file-index 状态。",
      blocks: (
        <>
          <MiniTable title="Stale datasets" rows={safeRows(data.data_health.stale_datasets)} columns={["dataset", "freshness_status", "rows", "path", "latest_timestamp"]} />
          <MiniTable title="Collectors" rows={safeRows(data.collectors.collectors)} columns={["collector", "status", "success_count", "latest_success_ts", "lag"]} />
          <MiniTable title="Latest per symbol" rows={safeRows(data.data_health.latest_per_symbol)} columns={["symbol", "timeframe", "latest_ts", "rows"]} />
        </>
      )
    };
  }
  if (view === "v5") {
    return {
      title: "V5 / 消费者",
      subtitle: "V5 bundle、risk_permission、fallback audit 和 API/Web 性能。",
      blocks: (
        <>
          <KeyValueGrid data={data.v5} />
          <MiniTable title="Risk permission rows" rows={safeRows(data.consumers.permission_rows)} columns={["strategy", "permission", "permission_status", "as_of_ts", "source"]} />
          <MiniTable title="API slow paths" rows={safeRows(data.web_perf.slow_paths)} columns={["path", "count", "p95", "max", "server_error_count"]} />
        </>
      )
    };
  }
  if (view === "exports") {
    return {
      title: "专家包",
      subtitle: "最新导出、manifest、data_quality 和专家问题摘要。",
      blocks: (
        <>
          <MiniTable title="Export packs" rows={safeRows(data.exports.packs)} columns={["name", "size_bytes", "modified_at", "path"]} />
          <KeyValueGrid data={data.exports.manifest_summary as Record<string, unknown>} />
          <MiniList title="Expert questions" rows={(data.exports.expert_questions as string[] | undefined) ?? []} />
        </>
      )
    };
  }
  return {
    title: "原始表格 / 旧版明细",
    subtitle: "V2 不删除 Streamlit；原始长表仍由旧 Web 明细页承载。",
    blocks: (
      <div className="raw-links">
        <a href="/" target="_blank" rel="noreferrer"><Archive size={16} />打开旧版 Streamlit 入口</a>
        <a href="/docs" target="_blank" rel="noreferrer"><RadioTower size={16} />打开 FastAPI Docs</a>
        <a href="/v1/web/bigscreen-snapshot" target="_blank" rel="noreferrer"><ShieldCheck size={16} />查看只读 snapshot JSON</a>
      </div>
    )
  };
}

function MiniTable({
  title,
  rows,
  columns
}: {
  title: string;
  rows: Record<string, unknown>[];
  columns: string[];
}) {
  return (
    <section className="mini-table-block">
      <h3>{title}</h3>
      <div className="mini-table">
        <div className="mini-table-row head">
          {columns.map((column) => <span key={column}>{column}</span>)}
        </div>
        {rows.slice(0, 8).map((row, i) => (
          <div className="mini-table-row" key={`${title}-${i}`}>
            {columns.map((column) => <span key={column}>{stringValue(row[column])}</span>)}
          </div>
        ))}
        {!rows.length && <div className="empty">not_observable</div>}
      </div>
    </section>
  );
}

function KeyValueGrid({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data ?? {}).slice(0, 18);
  return (
    <section className="kv-grid">
      {entries.map(([key, value]) => <div key={key}><span>{key}</span><b>{stringValue(value)}</b></div>)}
    </section>
  );
}

function MiniList({ title, rows }: { title: string; rows: string[] }) {
  return (
    <section className="mini-table-block">
      <h3>{title}</h3>
      <ul className="question-list">{rows.slice(0, 8).map((row, i) => <li key={i}>{row}</li>)}</ul>
    </section>
  );
}

function AnimatedBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particles = useMemo(() => Array.from({ length: 110 }, () => ({
    x: Math.random(),
    y: Math.random(),
    v: 0.0007 + Math.random() * 0.0012,
    r: 0.4 + Math.random() * 1.8
  })), []);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let frame = 0;
    let animation = 0;
    const draw = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      ctx.clearRect(0, 0, width, height);
      frame += 1;
      for (const particle of particles) {
        particle.x += particle.v;
        if (particle.x > 1) particle.x = 0;
        const x = particle.x * width;
        const y = (particle.y + Math.sin(frame / 180 + particle.x * 6) * 0.008) * height;
        ctx.fillStyle = "rgba(80,169,255,.55)";
        ctx.beginPath();
        ctx.arc(x, y, particle.r, 0, Math.PI * 2);
        ctx.fill();
      }
      animation = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animation);
  }, [particles]);
  return (
    <>
      <div id="bg" />
      <canvas ref={canvasRef} id="particles" aria-hidden="true" />
    </>
  );
}
