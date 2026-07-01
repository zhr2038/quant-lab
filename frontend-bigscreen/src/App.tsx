import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import {
  Archive,
  BarChart3,
  Boxes,
  ChevronLeft,
  ChevronRight,
  Database,
  DownloadCloud,
  Gauge,
  LayoutDashboard,
  MonitorDot,
  PackagePlus,
  RefreshCw,
  Sparkles,
  ShieldCheck,
  Table2,
  X
} from "lucide-react";
import { type CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import { ActionQueue } from "./components/ActionQueue";
import { CostQuality } from "./components/CostQuality";
import { DataMatrix } from "./components/DataMatrix";
import { HealthPanel } from "./components/HealthPanel";
import { KpiGrid } from "./components/KpiGrid";
import { LegacyAnomalyPanel } from "./components/LegacyAnomalyPanel";
import { MarketLiquidity } from "./components/MarketLiquidity";
import { PerfConsumers } from "./components/PerfConsumers";
import { StrategyFlow } from "./components/StrategyFlow";
import { V5Telemetry } from "./components/V5Telemetry";
import {
  expertPackDownloadUrl,
  fetchBigscreenSnapshot,
  fetchExpertPackStatus,
  safeRows,
  shortNumber,
  stringValue,
  triggerExpertPackGenerate,
  type BigscreenSnapshot
} from "./lib/api";
import "./styles.css";

const queryClient = new QueryClient();
type ViewKey = "strategy" | "data" | "v5" | "exports" | "raw";
type PageKey = "overview" | "strategy" | "data" | "ops";

const PAGES: Array<{ key: PageKey; label: string; description: string }> = [
  { key: "overview", label: "总览", description: "健康 / KPI / 数据矩阵" },
  { key: "strategy", label: "策略研究", description: "Factor Factory / 策略候选 / 市场" },
  { key: "data", label: "数据成本", description: "市场 / 成本 / 数据矩阵" },
  { key: "ops", label: "运行导出", description: "服务 / API / V5 / 专家包" }
];
const PAGE_KEYS = new Set<PageKey>(PAGES.map((item) => item.key));

function pageFromHash(): PageKey {
  if (typeof window === "undefined") return "overview";
  const key = window.location.hash.replace("#", "").trim() as PageKey;
  return PAGE_KEYS.has(key) ? key : "overview";
}

function formatPackTime(value: unknown): string {
  const raw = stringValue(value, "");
  if (!raw) return "—";
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return raw;
  const pad = (part: number) => String(part).padStart(2, "0");
  const offsetMinutes = -parsed.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const absOffset = Math.abs(offsetMinutes);
  const offset = `${sign}${pad(Math.floor(absOffset / 60))}${pad(absOffset % 60)}`;
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())} ${pad(
    parsed.getHours()
  )}:${pad(parsed.getMinutes())}:${pad(parsed.getSeconds())} ${offset}`;
}

function formatExpertPackStamp(fileName: string): string {
  const match = fileName.match(
    /^quant_lab_expert_pack_(\d{4}-\d{2}-\d{2})_(\d{8})T(\d{2})(\d{2})(\d{2})(?:\d{1,6})?([+-]\d{4}|Z)?\.zip$/
  );
  if (!match) return "";
  const [, exportDate, generatedDay, hour, minute, second, zone = ""] = match;
  const generatedDate = `${generatedDay.slice(0, 4)}-${generatedDay.slice(4, 6)}-${generatedDay.slice(6, 8)}`;
  const generatedTime = `${generatedDate} ${hour}:${minute}:${second}${zone ? ` ${zone}` : ""}`;
  return `生成 ${generatedTime} · 数据日期 ${exportDate}`;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Bigscreen />
    </QueryClientProvider>
  );
}

function Bigscreen() {
  const [view, setView] = useState<ViewKey | null>(null);
  const [page, setPageState] = useState<PageKey>(() => pageFromHash());
  const setPage = (nextPage: PageKey) => {
    setPageState(nextPage);
    const nextHash = `#${nextPage}`;
    if (window.location.hash !== nextHash) {
      window.history.replaceState(null, "", nextHash);
    }
  };
  useEffect(() => {
    const fit = () => {
      const viewport = window.visualViewport;
      const width = viewport?.width ?? window.innerWidth;
      const height = viewport?.height ?? window.innerHeight;
      const scale = Math.min(width / 1920, height / 1080);
      document.documentElement.style.setProperty("--screen-scale", String(scale));
    };
    fit();
    window.addEventListener("resize", fit);
    window.visualViewport?.addEventListener("resize", fit);
    return () => {
      window.removeEventListener("resize", fit);
      window.visualViewport?.removeEventListener("resize", fit);
    };
  }, []);
  useEffect(() => {
    const onHashChange = () => setPageState(pageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const index = PAGES.findIndex((item) => item.key === page);
      const nextIndex = event.key === "ArrowRight"
        ? (index + 1) % PAGES.length
        : (index + PAGES.length - 1) % PAGES.length;
      setPage(PAGES[nextIndex].key);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [page]);
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
      <div className="app-frame">
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
              <Dashboard data={data} view={view} setView={setView} page={page} setPage={setPage} />
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

function Dashboard({
  data,
  view,
  setView,
  page,
  setPage
}: {
  data: BigscreenSnapshot;
  view: ViewKey | null;
  setView: (view: ViewKey | null) => void;
  page: PageKey;
  setPage: (page: PageKey) => void;
}) {
  const pageContent = (() => {
    switch (page) {
      case "overview":
        return (
          <motion.section className="page-grid page-overview" key="overview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <KpiGrid snapshot={data} />
            <div className="overview-side">
              <HealthPanel score={data.health_score} status={data.status} warnings={data.warnings} />
              <LegacyAnomalyPanel anomalies={data.legacy_anomalies} />
              <ActionQueue actions={data.actions} />
            </div>
            <DataMatrix matrix={data.data_matrix} variant="overview" />
          </motion.section>
        );
      case "strategy":
        return (
          <motion.section className="page-grid page-strategy" key="strategy" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <StrategyFlow flow={data.strategy_flow} />
            <MarketLiquidity market={data.market} matrix={data.data_matrix} density="compact" />
          </motion.section>
        );
      case "data":
        return (
          <motion.section className="page-grid page-data" key="data" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <DataMatrix matrix={data.data_matrix} />
            <MarketLiquidity market={data.market} matrix={data.data_matrix} density="full" />
            <CostQuality cost={data.cost} />
          </motion.section>
        );
      case "ops":
        return (
          <motion.section className="page-grid page-ops" key="ops" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <PerfConsumers perf={data.web_perf} consumers={data.consumers} />
            <V5Telemetry v5={data.v5} consumers={data.consumers} />
            <section className="card pad ops-export-card">
              <ExpertPackControls exports={data.exports as Record<string, unknown>} />
            </section>
            <FunctionMap setView={setView} />
          </motion.section>
        );
    }
  })();

  return (
    <motion.div className="dashboard-shell" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
      <PageControls page={page} setPage={setPage} />
      {pageContent}
      <Drilldown view={view} data={data} onClose={() => setView(null)} />
    </motion.div>
  );
}

function PageControls({
  page,
  setPage
}: {
  page: PageKey;
  setPage: (page: PageKey) => void;
}) {
  const index = PAGES.findIndex((item) => item.key === page);
  const go = (offset: number) => {
    const nextIndex = (index + offset + PAGES.length) % PAGES.length;
    setPage(PAGES[nextIndex].key);
  };
  return (
    <nav className="page-controls" aria-label="Web v2 dashboard pages">
      <button className="page-arrow" onClick={() => go(-1)} aria-label="上一页"><ChevronLeft size={20} /></button>
      <div className="page-tabs">
        {PAGES.map((item) => (
          <button
            className={`page-tab ${item.key === page ? "active" : ""}`}
            key={item.key}
            onClick={() => setPage(item.key)}
          >
            <b>{item.label}</b>
            <span>{item.description}</span>
          </button>
        ))}
      </div>
      <button className="page-arrow" onClick={() => go(1)} aria-label="下一页"><ChevronRight size={20} /></button>
    </nav>
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
      <div className="drawer-scroll">
        <p>{content.subtitle}</p>
        {content.blocks}
      </div>
    </motion.aside>
  );
}

function drilldownContent(view: ViewKey, data: BigscreenSnapshot) {
  if (view === "strategy") {
    const candidates = dedupeRowsByKeys(safeRows(data.strategy_flow.top_candidates), [
      "symbol",
      "horizon_hours",
      "recommended_mode",
      "decision"
    ]);
    const alphaFactory = safeRows(data.strategy_flow.alpha_factory);
    const riskOn = safeRows(data.strategy_flow.risk_on_multi_buy);
    const factorFactory = (data.strategy_flow.factor_factory ?? {}) as Record<string, unknown>;
    const factorPaper = safeRows(factorFactory.paper_ready_candidates);
    const factorTop = safeRows(factorFactory.top_candidates);
    const factorRows = factorPaper.length ? factorPaper : factorTop;
    const factorPaperQueue = safeRows(factorFactory.paper_review_queue);
    const factorDedupe = safeRows(factorFactory.dedupe_decisions);
    const factorLeaderboard = safeRows(factorFactory.family_leaderboard);
    const compositeCandidates = safeRows(factorFactory.composite_candidates);
    const regimeEffectiveness = safeRows(factorFactory.regime_effectiveness);
    const bridgeCandidates = safeRows(factorFactory.strategy_bridge_candidates);
    return {
      title: "策略驾驶舱",
      subtitle: "聚合 advisory、Alpha Factory、Factor Factory、risk-on multi-buy 和研究组合；全部为 read-only。",
      blocks: (
        <>
          <section className="factor-factory-drawer">
            <div>
              <h3><Sparkles size={17} /> Factor Factory</h3>
              <p>自动发现与测试因子，只做研究展示；PAPER_READY 仅代表 paper review，不代表 live eligibility。</p>
            </div>
            <KeyValueGrid
              data={{
                live_order_effect: factorFactory.live_order_effect,
                candidate_count: factorFactory.candidate_count,
                paper_ready_count: factorFactory.paper_ready_count,
                high_correlation_pair_count: factorFactory.high_correlation_pair_count,
                paper_review_queue_count: factorPaperQueue.length,
                dedupe_decision_count: factorDedupe.length,
                composite_candidate_count: compositeCandidates.length,
                bridge_candidate_count: bridgeCandidates.length,
                latest_candidate_created_at: factorFactory.latest_candidate_created_at
              }}
            />
          </section>
          <MiniTable
            title="Factor paper review queue"
            rows={factorPaperQueue}
            columns={["factor_id", "factor_family", "candidate_state", "best_horizon_bars", "best_rank_ic_mean", "best_rank_ic_tstat", "best_long_short_mean_bps", "sample_count", "recommendation"]}
          />
          <MiniTable
            title="Factor dedupe decisions"
            rows={factorDedupe}
            columns={["factor_id", "correlation_cluster_id", "leader_factor_id", "dedupe_decision", "dedupe_reason", "cluster_size"]}
          />
          <MiniTable
            title="Factor family leaderboard"
            rows={factorLeaderboard}
            columns={["factor_family", "leader_factor_id", "leader_candidate_state", "factor_count", "paper_ready_count", "leader_best_rank_ic_tstat", "leader_best_long_short_mean_bps"]}
          />
          <MiniTable
            title="Composite factor candidates"
            rows={compositeCandidates}
            columns={["composite_factor_id", "factor_terms", "available_term_count", "max_terms", "interpretable_only", "missing_terms", "recommendation"]}
          />
          <MiniTable
            title="Factor regime effectiveness"
            rows={regimeEffectiveness}
            columns={["factor_id", "regime", "horizon", "rank_ic", "long_short_bps", "win_rate", "sample_count", "recommendation"]}
          />
          <MiniTable
            title="Strategy bridge candidates"
            rows={bridgeCandidates}
            columns={["factor_id", "symbol", "regime", "horizon", "bridge_candidate_id", "eligible_for_alpha_factory", "recommended_action", "blocking_reasons", "live_order_effect"]}
          />
          <MiniTable
            title="Factor Factory candidates"
            rows={factorRows}
            columns={["factor_id", "factor_family", "candidate_state", "best_horizon_bars", "best_score", "best_rank_ic_mean", "best_long_short_mean_bps", "recommended_action"]}
          />
          <MiniTable
            title="Factor evidence by horizon"
            rows={safeRows(factorFactory.evidence_by_horizon)}
            columns={["horizon_bars", "factor_count", "paper_ready_count", "avg_score", "avg_rank_ic_mean", "avg_long_short_mean_bps"]}
          />
          <MiniTable
            title="High correlation factor pairs"
            rows={safeRows(factorFactory.high_correlation_pairs)}
            columns={["factor_id_left", "factor_id_right", "correlation", "sample_count", "timeframe"]}
          />
          <MiniTable title="策略候选（只读）" rows={dedupeRowsByKeys(safeRows(data.strategy_flow.top_live_candidates).concat(candidates), ["symbol", "horizon_hours", "recommended_mode", "decision"]).slice(0, 8)} columns={["strategy_candidate", "symbol", "recommended_mode", "decision", "avg_net_bps", "p25_net_bps"]} />
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
          <MiniTable title="Stale datasets" rows={safeRows(data.data_health.stale_datasets)} columns={["dataset", "freshness_status", "rows", "path", "latest_timestamp"]} emptyLabel="无过期或缺失数据集" />
          <MiniTable title="Collectors" rows={safeRows(data.collectors.collectors)} columns={["collector", "status", "success_count", "latest_success_ts", "lag"]} />
          <MiniTable title="Latest per symbol" rows={safeRows(data.data_health.latest_per_symbol)} columns={["symbol", "timeframe", "latest_ts", "rows"]} />
        </>
      )
    };
  }
  if (view === "v5") {
    return {
      title: "V5 / 消费者",
      subtitle: "V5 遥测、risk_permission、fallback audit 和 API/Web 性能。",
      blocks: (
        <>
          <KeyValueGrid data={data.v5} />
          <MiniTable title="Risk permission rows" rows={safeRows(data.consumers.permission_rows)} columns={["strategy", "permission", "permission_status", "as_of_ts", "source"]} />
          <MiniTable
            title="API slow paths"
            rows={safeRows(data.web_perf.slow_paths)}
            columns={[
              "path",
              "count",
              "success_p95",
              "p95",
              "max",
              "error_count",
              "auth_error_count",
              "client_error_count",
              "server_error_count"
            ]}
          />
        </>
      )
    };
  }
  if (view === "exports") {
    return {
      title: "专家包",
      subtitle: "直接生成、轮询并下载今日专家包；该入口只写导出文件，不影响交易。",
      blocks: (
        <>
          <ExpertPackControls exports={data.exports as Record<string, unknown>} />
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
        <a href={legacyWebUrl()} target="_blank" rel="noreferrer"><Archive size={16} />打开旧版 Streamlit 入口</a>
        <a href="/web-v2/snapshot" target="_blank" rel="noreferrer"><ShieldCheck size={16} />查看只读 snapshot JSON</a>
      </div>
    )
  };
}

function legacyWebUrl(): string {
  if (typeof window === "undefined") return "/web-v2/legacy";
  const protocol = window.location.protocol || "http:";
  const hostname = window.location.hostname || "qyun2.hrhome.top";
  return `${protocol}//${hostname}:8501/`;
}

function ExpertPackControls({ exports }: { exports: Record<string, unknown> }) {
  const queryClient = useQueryClient();
  const statusQuery = useQuery({
    queryKey: ["expert-pack-status"],
    queryFn: fetchExpertPackStatus,
    refetchInterval: (query) => {
      const state = String(query.state.data?.state ?? "").toLowerCase();
      return ["running", "starting"].includes(state) ? 3000 : false;
    },
    retry: 1
  });
  const generateMutation = useMutation({
    mutationFn: triggerExpertPackGenerate,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["expert-pack-status"] });
      void queryClient.invalidateQueries({ queryKey: ["bigscreen-snapshot"] });
    }
  });
  const status = statusQuery.data;
  const packs = status?.packs ?? [];
  const latestName = stringValue(status?.latest_pack_name, "");
  const latestUrl = status?.latest_download_url ?? "";
  const latestFileName = fileNameFromPath(latestName || latestUrl);
  const availableName = stringValue(status?.available_pack_name, status?.available_pack ?? "");
  const availableFileName = fileNameFromPath(availableName);
  const latestPack = packs.find((pack) => {
    const name = stringValue(pack.name, stringValue(pack.path, ""));
    return latestFileName && fileNameFromPath(name) === latestFileName;
  });
  const availablePack = packs.find((pack) => {
    const name = stringValue(pack.name, stringValue(pack.path, ""));
    return availableFileName && fileNameFromPath(name) === availableFileName;
  });
  const fallbackPack = packs[0];
  const displayPack = latestPack ?? availablePack ?? fallbackPack;
  const displayFileName = latestFileName || availableFileName || fileNameFromPath(
    stringValue(displayPack?.name, stringValue(displayPack?.path, ""))
  );
  const displayUrl = latestUrl || stringValue(displayPack?.download_url, displayFileName);
  const displayGeneratedAt = formatExpertPackStamp(displayFileName);
  const displayPrimaryText = displayGeneratedAt || displayFileName || "latest.zip";
  const displayModifiedAt = formatPackTime(displayPack?.modified_at);
  const requestedDate = stringValue(status?.export_date, "");
  const requestedPackPrefix = requestedDate ? `quant_lab_expert_pack_${requestedDate}_` : "";
  const displayPackMatchesRequestedDate = Boolean(
    displayFileName && requestedPackPrefix && displayFileName.startsWith(requestedPackPrefix)
  );
  const previousPackOnly = Boolean(displayUrl && requestedDate && !displayPackMatchesRequestedDate);
  const downloadLatestLabel = displayPackMatchesRequestedDate
    ? "下载今日专家包"
    : "下载上一份专家包";
  const latestMeta = [
    displayGeneratedAt ? displayFileName : "",
    displayPack?.modified_at ? `mtime ${displayModifiedAt}` : "",
    status?.latest_size_bytes || displayPack?.size_bytes ? `${shortNumber(status?.latest_size_bytes || displayPack?.size_bytes)}B` : ""
  ].filter(Boolean);
  const visiblePacks = packs.filter((pack) => {
    const name = stringValue(pack.name, stringValue(pack.path, ""));
    return !displayFileName || fileNameFromPath(name) !== displayFileName;
  });
  const state = String(status?.state ?? (statusQuery.isLoading ? "loading" : "not_observable"));
  const statusBody = status?.status ?? {};
  const isRunning = ["running", "starting"].includes(state.toLowerCase());
  const cooldownRemaining = Math.max(
    0,
    Math.ceil(numberValue(status?.regenerate_cooldown_remaining_seconds))
  );
  const isCoolingDown = cooldownRemaining > 0;
  const v5LagStatus = stringValue(exports?.latest_pack_v5_lag_status, "");
  const v5LagMinutes = numberValue(exports?.latest_pack_v5_lag_minutes);
  const v5LagWarning = v5LagStatus.toUpperCase() === "WARNING" && v5LagMinutes > 0;
  const packCommit = stringValue(status?.latest_pack_quant_lab_git_commit, "");
  const currentCommit = stringValue(status?.current_quant_lab_git_commit, "");
  const codeLagStatus = stringValue(status?.latest_pack_code_lag_status, "");
  const codeLagWarning = codeLagStatus.toUpperCase() === "WARNING";
  const rawState = state.toLowerCase();
  const displayState = !isRunning && displayUrl && rawState === "manual_missing"
    ? "PACK_AVAILABLE"
    : !isRunning && previousPackOnly && rawState === "missing_requested_date"
      ? "PREVIOUS_PACK_AVAILABLE"
      : state;
  const lastError = stringValue(statusBody.error, "");
  const generateLabel = isRunning
    ? "生成中"
    : isCoolingDown
      ? `刚生成 ${cooldownRemaining}s`
      : "生成今日专家包";

  return (
    <section className="export-console">
      <div className="export-console-head">
        <div>
          <h3><Archive size={18} />今日专家包</h3>
          <p>read-only export · live_order_effect: none · 生成后可直接下载 ZIP</p>
        </div>
        <div className="export-actions">
          <button
            className="ghost-action"
            onClick={() => void statusQuery.refetch()}
            disabled={statusQuery.isFetching}
          >
            <RefreshCw size={15} className={statusQuery.isFetching ? "spin" : ""} />刷新状态
          </button>
          <button
            className="primary-action"
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending || isRunning || isCoolingDown}
          >
            <PackagePlus size={16} />{generateLabel}
          </button>
        </div>
      </div>
      <div className={`export-status-line ${displayState.toLowerCase()}`}>
        <span className="status-dot" />
        <b>{displayState}</b>
        <em>日期 {stringValue(status?.export_date, "today")}</em>
        <em>历史包 {status?.pack_count ?? packs.length}</em>
        {status?.latest_size_bytes || displayPack?.size_bytes ? (
          <em>包大小 {shortNumber(status?.latest_size_bytes || displayPack?.size_bytes)}B</em>
        ) : null}
        {isCoolingDown ? <em>可重新生成 {cooldownRemaining}s</em> : null}
      </div>
      {generateMutation.error || statusQuery.error || lastError ? (
        <div className="export-error">
          {lastError || String(generateMutation.error || statusQuery.error)}
        </div>
      ) : null}
      {previousPackOnly ? (
        <div className="export-warning">
          今日 {requestedDate} 专家包尚未生成；当前下载入口是上一份可用专家包。
        </div>
      ) : null}
      {v5LagWarning ? (
        <div className="export-warning">
          当前可下载包内 V5 证据落后最新遥测 {Math.round(v5LagMinutes)} 分钟；需要最新证据时请手动重新生成。
        </div>
      ) : null}
      {codeLagWarning ? (
        <div className="export-warning">
          当前可下载包由旧 quant-lab 代码生成（包 {shortCommit(packCommit)} · 当前 {shortCommit(currentCommit)}）；需要最新代码证据时请手动重新生成。
        </div>
      ) : null}
      {displayUrl ? (
        <a className="download-latest" href={expertPackDownloadUrl(displayUrl)} download title={displayFileName}>
          <DownloadCloud size={18} />
          <span className="download-latest-label">{downloadLatestLabel}</span>
          <span className="download-latest-name">{displayPrimaryText}</span>
          {latestMeta.length ? <span className="download-latest-meta">{latestMeta.join(" · ")}</span> : null}
        </a>
      ) : (
        <div className="export-empty">未找到可下载专家包，点击“生成今日专家包”提交后台导出。</div>
      )}
      <div className="pack-list">
        {visiblePacks.slice(0, 8).map((pack, index) => {
          const name = stringValue(pack.name, stringValue(pack.path, `pack-${index}`));
          const url = stringValue(pack.download_url, "");
          const modifiedAt = formatPackTime(pack.modified_at);
          const rawModifiedAt = stringValue(pack.modified_at);
          return (
            <div className="pack-row" key={`${name}-${index}`} title={`${name} | ${rawModifiedAt}`}>
              <span className="pack-name">{name}</span>
              <em className="pack-meta">
                <span>{shortNumber(pack.size_bytes)}B</span>
                <span>{modifiedAt}</span>
              </em>
              <a className="pack-download" href={expertPackDownloadUrl(url || name)} download>
                <DownloadCloud size={14} />下载
              </a>
            </div>
          );
        })}
        {!visiblePacks.length && <div className="export-empty">{packs.length ? "最新包已在上方显示" : "not_observable"}</div>}
      </div>
    </section>
  );
}

function dedupeRowsByKeys(rows: Record<string, unknown>[], keys: string[]): Record<string, unknown>[] {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = keys.map((name) => stringValue(row[name], "")).join("|");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function fileNameFromPath(value: unknown): string {
  const text = stringValue(value, "");
  if (!text) return "";
  const clean = text.split("?")[0].replace(/\\/g, "/");
  return clean.split("/").filter(Boolean).pop() ?? "";
}

function numberValue(value: unknown): number {
  if (value === null || value === undefined || value === "") return 0;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function shortCommit(value: unknown): string {
  const text = stringValue(value, "").trim();
  return text ? text.slice(0, 7) : "unknown";
}

function MiniTable({
  title,
  rows,
  columns,
  emptyLabel = "not_observable"
}: {
  title: string;
  rows: Record<string, unknown>[];
  columns: string[];
  emptyLabel?: string;
}) {
  return (
    <section className="mini-table-block">
      <h3>{title}</h3>
      <div className="mini-table">
        <div className="mini-table-row head" style={{ "--cols": columns.length } as CSSProperties}>
          {columns.map((column) => <span key={column}>{column}</span>)}
        </div>
        {rows.slice(0, 8).map((row, i) => (
          <div className="mini-table-row" key={`${title}-${i}`} style={{ "--cols": columns.length } as CSSProperties}>
            {columns.map((column) => <span key={column}>{stringValue(row[column])}</span>)}
          </div>
        ))}
        {!rows.length && <div className="empty">{emptyLabel}</div>}
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
