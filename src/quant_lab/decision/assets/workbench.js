"use strict";
const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];
const LABELS = {DEFER: "等待", REVIEW_ENTRY: "复核入场", KEEP_BASELINE: "保持原规则", NO_VIEW: "暂无观点"};
const TITLES = {reference: "交易参考", forward: "效果观察", experiment: "实验与数据", servers: "服务器状态"};
const REASONS = {
  CURRENT_MARKET_MISSING: "缺少当前行情", CURRENT_MARKET_STALE: "当前行情已过期",
  HISTORY_CURRENT_MISMATCH: "历史与当前输入不一致", CONTEXT_WINDOW_INCOMPLETE: "趋势窗口有缺口",
  COST_MISSING: "缺少成本估计", COST_STALE_OR_UNDATED: "成本时间过旧或缺失",
  COST_TIMESTAMP_FUTURE: "成本时间异常", HISTORICAL_SAMPLE_INSUFFICIENT: "历史非重叠样本不足",
  RECENT_DIAGNOSTIC_SAMPLE_INSUFFICIENT: "近期诊断窗口样本不足", COST_NOT_PAPER_TRUSTED: "成本尚未用正常成交校准",
  HISTORICAL_COST_DRAG: "相似窗口的成本压力较大", POSITIVE_HISTORICAL_REFERENCE: "历史参考偏正",
  COST_REQUIRES_CALIBRATION: "需要进一步校准成交成本", HISTORICAL_REFERENCE_MIXED: "历史与近期结果不一致",
  FORWARD_VALUE_NOT_ESTABLISHED: "前向交易增益尚未建立", ADVICE_EXPIRED: "建议已到期",
  CURRENT_COST_EXPIRED_OR_UNAVAILABLE: "当前费率或盘口成本缺失、过期", CURRENT_COST_IS_ESTIMATE: "当前成本为估计值",
  COST_ESTIMATE_UNCALIBRATED: "尚未用正常成交校准", EXIT_BOOK_ASSUMED_CURRENT: "退出成本按当前盘口估计",
  ACCOUNT_FEE_UNAVAILABLE: "缺少账户费率", ACCOUNT_FEE_EXPIRED: "账户费率已过期",
  READONLY_FEE_ACCESS_UNAVAILABLE: "账户费率读取权限或连接不可用", ACCOUNT_FEE_REFRESH_FAILED: "账户费率本次读取失败",
  ACCOUNT_FEE_USING_PREVIOUS_OBSERVATION: "沿用此前读取的费率，未更新其时间",
  CURRENT_BOOK_FETCH_FAILED: "本次盘口读取失败", CURRENT_BOOK_OR_INSTRUMENT_UNAVAILABLE: "盘口或交易规格校验未通过",
  REBATE_NOT_CREDITED_IN_ESTIMATE: "潜在返佣未计为成本折扣", INSUFFICIENT_DEPTH: "盘口深度不足", BELOW_MINIMUM_SIZE: "低于最小数量"
};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, x => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
const num = (value, digits = 1) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN", {maximumFractionDigits: digits, minimumFractionDigits: digits}) : "—";
const bps = value => typeof value === "number" && Number.isFinite(value) ? `${value > 0 ? "+" : ""}${num(value)} bps` : "—";
const bookBps = value => typeof value === "number" && Number.isFinite(value) ? `${num(value, 3)} bps` : "—";
const time = (value, full = false) => {
  if (!value || !Number.isFinite(Date.parse(value))) return "—";
  return new Intl.DateTimeFormat("zh-CN", {timeZone: "Asia/Shanghai", ...(full ? {month:"2-digit", day:"2-digit"} : {}), hour:"2-digit", minute:"2-digit", hour12:false}).format(new Date(value));
};
const state = {data: null, symbol: "BTCUSDT", horizon: 4, page: "reference", busy: false, error: "", offset: 0, expiryTimer: null};
const serverState = {data: null, busy: false, error: "", receivedAt: 0};
try { sessionStorage.removeItem("qyun2.access"); } catch (_) { /* Clear the retired login credential when storage is available. */ }
const effective = advice => !advice || state.error || Date.now() + state.offset >= Date.parse(advice.expires_at) ? "NO_VIEW" : (advice.effective_action || advice.action);
const badge = action => `<span class="action ${esc(action)}">${esc(LABELS[action] || "暂无观点")}</span>`;
const metric = (title, value) => `<div class="metric"><dt>${esc(title)}</dt><dd class="number">${esc(value)}</dd></div>`;

function setPage(page) {
  state.page = page;
  $("page-title").textContent = TITLES[page];
  $("breadcrumb-page").textContent = TITLES[page];
  document.querySelectorAll(".page").forEach(node => { node.hidden = node.id !== `page-${page}`; });
  document.querySelectorAll(".nav").forEach(node => {
    node.classList.toggle("active", node.dataset.page === page);
    if (node.dataset.page === page) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current");
  });
  $("notice").hidden = page === "servers";
  $("export").hidden = page === "servers";
  syncRefreshButton();
  if (page === "servers") {renderServers(); refreshServers();}
}

const HOST_LABELS = {qyun2: ["qyun2", "行情采集 · 参考发布 · Web"], nas: ["NAS", "历史归档 · 按计划分析"]};
const SERVICE_LABELS = {api: "中台 API", market: "实时行情采集", https: "网页 HTTPS", decision: "输入与结果发布", backfill: "行情补齐", compaction: "数据整理", analysis: "分析任务"};
const STATUS_LABELS = {ok: "正常", warning: "需关注", stale: "状态已过期", unknown: "状态未知", running: "运行中", scheduled: "等待下一轮", stopped: "已停止", failed: "失败", restarting: "正在重启", overdue: "任务未及时完成", missing: "服务缺失", exited: "已退出", created: "已创建", paused: "已暂停", dead: "异常退出", removing: "删除中"};
const STATUS_WARNINGS = {SNAPSHOT_MISSING: "尚未收到采样", SNAPSHOT_INVALID: "采样校验失败", SNAPSHOT_STALE: "超过 3 分钟未更新", RESOURCE_COLLECTION_FAILED: "资源采集失败", SERVICE_COLLECTION_FAILED: "服务采集失败", CONTAINER_COLLECTION_FAILED: "容器采集失败", WORKER_COLLECTION_FAILED: "分析任务状态采集失败", CPU_UNAVAILABLE: "CPU 采样不可用", CPU_HIGH: "CPU 占用较高", MEMORY_LOW: "可用内存不足 10%", NAS_MEMORY_RESERVE_LOW: "NAS 可用内存低于分析任务的 6 GiB 预留要求", DISK_LOW: "磁盘剩余不足 10%", SERVICE_ATTENTION: "关键服务或任务需关注", CONTAINER_ATTENTION: "有容器未正常运行"};
const statusBadge = (value, text) => `<span class="status-badge ${esc(value)}">${esc(text || STATUS_LABELS[value] || "状态未知")}</span>`;
const capacity = value => typeof value === "number" ? `${num(value / 1024 ** (value >= 1024 ** 4 ? 4 : 3))} ${value >= 1024 ** 4 ? "TiB" : "GiB"}` : "—";
const usageMetric = (label, percent, caption) => `<div class="usage-metric"><dt>${esc(label)}</dt><dd class="number">${num(percent)}<small>%</small></dd>${Number.isFinite(percent) ? `<meter min="0" max="100" low="75" high="90" optimum="20" value="${Math.max(0, Math.min(100, percent))}" aria-label="${esc(label)}占用">${num(percent)}%</meter>` : ""}<p>${esc(caption)}</p></div>`;

function renderServers() {
  const elapsed = serverState.receivedAt ? (Date.now() - serverState.receivedAt) / 1000 : 0;
  const hosts = (serverState.data?.hosts || ["qyun2", "nas"].map(host => ({host, state: "unknown", warnings: ["SNAPSHOT_MISSING"]}))).map(host => {
    const age = host.age_seconds === null ? null : host.age_seconds + elapsed;
    return {...host, age_seconds: age, state: serverState.error ? "unknown" : age > 180 ? "stale" : host.state};
  });
  const attention = hosts.filter(host => host.state !== "ok").length;
  $("server-notice").classList.toggle("warning", Boolean(serverState.error) || attention > 0);
  $("server-notice").textContent = serverState.error || (serverState.data ? `${hosts.length} 台主机 · ${attention ? `${attention} 台需关注` : "当前采样正常"} · 最近读取 ${time(serverState.data.viewed_at, true)}` : "正在读取服务器状态…");
  $("server-hosts").innerHTML = hosts.map(host => {
    const labels = HOST_LABELS[host.host] || ["主机", ""];
    const r = host.resources, stale = ["unknown", "stale"].includes(host.state);
    const metrics = r ? `<dl class="server-usage">${usageMetric("CPU", r.cpu_percent, `${num(r.cpu_cores, 0)} 核 · 1 分钟负载 ${num(r.load_1m, 2)}`)}${usageMetric("内存", 100 * (1 - r.memory_available_bytes / r.memory_total_bytes), `可用 ${capacity(r.memory_available_bytes)} / 共 ${capacity(r.memory_total_bytes)}`)}</dl><div class="disk-list">${r.disks.map(d => {
      const used = 100 * (1 - d.free_bytes / d.total_bytes);
      return `<div class="disk-row"><div><strong>${esc({system: "系统盘", ssd: "SSD 工作盘", hdd: "HDD 归档盘"}[d.id] || "磁盘")}</strong><span>剩余 ${capacity(d.free_bytes)} / ${capacity(d.total_bytes)}</span></div><meter min="0" max="100" low="75" high="90" optimum="20" value="${used}" aria-label="${esc(d.id)}磁盘占用">${num(used)}%</meter><span class="number">已用 ${num(used)}%</span></div>`;
    }).join("")}</div><p class="host-footnote">连续运行 ${num(r.uptime_seconds / 86400)} 天 · Swap 已用 ${capacity(r.swap_used_bytes)} / ${capacity(r.swap_total_bytes)}</p>` : `<p class="empty">资源采样暂不可用。</p>`;
    const services = (host.services || []).map(s => `<tr><td>${esc(SERVICE_LABELS[s.id] || s.id)}<span class="subline">${s.interval_seconds ? `每 ${num(s.interval_seconds / 60, 0)} 分钟` : "常驻服务"}</span></td><td>${statusBadge(stale ? "unknown" : s.state, stale ? "当前未知" : null)}</td><td class="number">${s.interval_seconds ? time(s.last_finished_at, true) : `${num(s.restart_count, 0)} 次重启`}</td></tr>`).join("");
    const containers = host.host === "nas" ? `<div class="host-subheading"><h3>容器</h3><span>${num((host.containers || []).length, 0)} 个 · 重启次数为创建以来累计</span></div>${host.warnings?.includes("CONTAINER_COLLECTION_FAILED") ? `<p class="empty">容器状态采集失败。</p>` : `<div class="container-list">${(host.containers || []).map(c => `<div class="container-row"><span>${esc(c.name)}<small>${c.health ? esc({healthy: "健康检查通过", unhealthy: "健康检查异常", starting: "健康检查启动中"}[c.health]) : "未配置健康检查"}</small></span>${statusBadge(stale ? "unknown" : c.health === "unhealthy" ? "failed" : c.state)}<span class="number">${num(c.restart_count, 0)} 次</span></div>`).join("") || `<p class="empty">采样时没有容器。</p>`}</div>`}` : "";
    const task = host.services?.find(s => s.id === "analysis");
    const warnings = [...new Set([...(host.warnings || []), ...(host.state === "stale" ? ["SNAPSHOT_STALE"] : [])])];
    return `<article class="host-panel ${stale ? "unconfirmed" : ""}"><div class="host-heading"><div><h3>${esc(labels[0])}</h3><p>${esc(labels[1])}</p></div>${statusBadge(host.state)}</div><p class="sample-time">${stale && r ? "以下为上次采样 · " : ""}采样时间 ${time(host.observed_at, true)}${Number.isFinite(host.age_seconds) ? ` · ${num(Math.max(0, host.age_seconds), 0)} 秒前` : ""}</p>${warnings.length ? `<p class="host-warning">${warnings.map(w => esc(STATUS_WARNINGS[w] || "状态需关注")).join(" · ")}</p>` : ""}${metrics}<div class="host-subheading"><h3>关键服务与任务</h3><span>最近记录 / 重启累计</span></div>${services ? `<div class="table-scroll"><table class="simple-table service-table"><thead><tr><th>服务 / 频率</th><th>状态</th><th>最近记录</th></tr></thead><tbody>${services}</tbody></table></div>` : `<p class="empty">服务状态暂不可用。</p>`}${task ? `<p class="host-footnote">上次分析耗时 ${num(task.runtime_seconds, 2)} 秒${typeof task.peak_rss_mib === "number" ? ` · 进程峰值 ${num(task.peak_rss_mib)} MiB` : ""}。分析结果以云端校验发布为准。</p>` : ""}${containers}</article>`;
  }).join("");
}

function syncRefreshButton() {
  const busy = state.page === "servers" ? serverState.busy : state.busy;
  $("refresh").disabled = busy; $("refresh").textContent = busy ? "更新中…" : "刷新";
}

async function refreshServers() {
  if (serverState.busy) return;
  serverState.busy = true; syncRefreshButton();
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch("/v1/server-status", {cache: "no-store", signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.hosts) || data.hosts.length !== 2) throw new Error("状态数据格式异常");
    serverState.data = data; serverState.receivedAt = Date.now(); serverState.error = "";
  } catch (error) {
    serverState.error = `服务器状态读取失败，当前状态未知。${error.name === "AbortError" ? "请求超时。" : error.message}`;
  } finally {
    clearTimeout(timer); serverState.busy = false; renderServers(); syncRefreshButton();
  }
}

function plot(distribution) {
  const values = [distribution.net_p10_bps, distribution.net_p50_bps, distribution.net_p90_bps];
  if (!values.every(x => typeof x === "number" && Number.isFinite(x))) return `<p class="empty">样本尚不足以形成分布。</p>`;
  const lo = Math.min(0, values[0]), hi = Math.max(0, values[2]);
  const span = Math.max(hi - lo, 1), x = value => 25 + (value - lo) / span * 550;
  return `<svg viewBox="0 0 600 96" role="img" aria-label="历史净结果第十、五十和九十分位"><title>当前成本假设下的历史分布</title>
    <line x1="25" y1="48" x2="575" y2="48" stroke="var(--line)" stroke-width="4"/>
    <line x1="${x(0)}" y1="16" x2="${x(0)}" y2="80" stroke="var(--plot-zero)" stroke-dasharray="3 3"/>
    <line x1="${x(values[0])}" y1="48" x2="${x(values[2])}" y2="48" stroke="var(--plot-range)" stroke-width="5"/>
    <circle cx="${x(values[1])}" cy="48" r="5" fill="var(--accent)"/></svg>
    <div class="distribution-quantiles">${values.map((value, i) => `<span>${["P10", "P50", "P90"][i]}<strong class="number">${esc(num(value))}</strong></span>`).join("")}</div>
    <p class="distribution-note">单位 bps · 虚线为零</p>`;
}

function renderReference() {
  const advice = state.data?.advice || [];
  $("reference-rows").innerHTML = SYMBOLS.map(symbol => {
    const row = advice.find(a => a.symbol === symbol && a.horizon_hours === state.horizon);
    return `<tr class="${symbol === state.symbol ? "selected" : ""}"><td><button class="coin-button" data-symbol="${symbol}" aria-label="查看 ${symbol.replace("USDT", "")} 依据">${symbol.replace("USDT", "")}<span>/ USDT</span></button><span class="subline number">${num(row?.last_close, 2)}</span></td><td>${badge(effective(row))}</td><td class="number">${bps(row?.distribution?.net_mean_bps)}</td><td class="number">${num(row?.distribution?.non_overlapping_samples, 0)}</td><td class="number">${time(row?.expires_at)}</td></tr>`;
  }).join("");
  $("reference-rows").querySelectorAll("[data-symbol]").forEach(button => button.addEventListener("click", () => {state.symbol = button.dataset.symbol; renderReference();}));
  const row = advice.find(a => a.symbol === state.symbol && a.horizon_hours === state.horizon);
  if (!row) {
    $("advice-detail").innerHTML = `<div class="detail-title"><h3>${state.symbol.replace("USDT", "")}</h3>${badge("NO_VIEW")}</div><div class="empty"><strong>等待已校验的分析结果</strong>当前没有可显示的建议。结果发布后，这里会展示成本、样本和有效时间。</div>`;
    return;
  }
  const reasons = [...(row.effective_reason_codes || row.reason_codes)];
  if (Date.now() + state.offset >= Date.parse(row.expires_at) && !reasons.includes("ADVICE_EXPIRED")) reasons.push("ADVICE_EXPIRED");
  $("advice-detail").innerHTML = `<div class="detail-title"><h3>${esc(row.symbol.replace("USDT", ""))}</h3>${badge(effective(row))}</div>
    <p class="explanation">${esc(row.explanation)}</p>
    <dl class="metrics">${metric("24h 趋势", bps(row.trend_24h_bps))}${metric("24h 实现波动", bps(row.volatility_24h_bps))}${metric("往返成本假设", bps(row.cost.roundtrip_bps))}${metric("参考金额", `${num(row.cost.notional_usdt, 0)} USDT`)}</dl>
    ${renderCost(row.cost)}
    <div class="distribution"><p>相似行情 · 当前成本下的历史分布</p>${plot(row.distribution)}</div>
    <div class="detail-list"><p><strong>行情截至</strong>　${time(row.market_asof, true)}</p><p><strong>历史范围</strong>　${time(row.distribution.first_signal_at, true)} — ${time(row.distribution.last_signal_at, true)}</p><p><strong>成本前历史均值</strong>　${bps(row.distribution.gross_mean_bps)}</p><p><strong>近期窗口净均值</strong>　${bps(row.distribution.chronological_tail_net_mean_bps)}</p><p><strong>双倍成本净均值</strong>　${bps(row.distribution.double_cost_mean_bps)}</p><p><strong>成本观测时间</strong>　${time(row.cost.as_of, true)}</p><p><strong>依据与限制</strong><br>${reasons.map(r => esc(REASONS[r] || r)).join(" · ")}</p><p><strong>失效条件</strong><br>${row.invalidation_conditions.map(esc).join(" · ")}</p></div>
    <details><summary>查看证据标识</summary><p>建议：<code>${esc(row.advice_id)}</code></p><p>输入：<code>${esc(row.input_snapshot_id)}</code></p><p>数据：<code>${esc(row.data_snapshot_hash)}</code></p></details>`;
}

function renderCost(cost) {
  const c = cost.current;
  if (!c) return `<p class="flat-note">当前显示旧版历史成本场景，观测时间 ${time(cost.as_of, true)}。等待当前费率与盘口快照。</p>`;
  const s = c.sizes.find(s => s.notional_usdt === cost.notional_usdt);
  const stale = !c.valid_until || Date.now() + state.offset >= Date.parse(c.valid_until);
  const label = state.error ? "连接失败 · 上次观测" : c.status !== "ESTIMATED" ? "当前成本不可用" : stale ? "成本快照已过期" : "当前成本 · 估计未校准";
  return `<section class="cost-detail" aria-label="成本拆解"><h3>${label}</h3>
    <dl class="cost-components">${metric("往返手续费", bps(s?.fee_roundtrip_bps))}${metric("盘口影响（含价差）", bookBps(s?.book_roundtrip_bps))}${metric("预留误差", bps(s?.uncertainty_bps))}</dl>
    <p>合计 ${bps(cost.roundtrip_bps)}。以吃单进出估计，盘口影响已含价差，只扣一次。预留误差为固定假设，不是已验证的滑点上限。</p>
    <div class="table-scroll"><table class="simple-table cost-table"><thead><tr><th>参考金额</th><th>盘口影响</th><th>往返合计</th></tr></thead><tbody>${c.sizes.map(v => `<tr><td>${num(v.notional_usdt,0)} USDT</td><td>${bookBps(v.book_roundtrip_bps)}</td><td>${v.status === "ESTIMATED" ? bps(v.roundtrip_bps) : esc(REASONS[v.status] || v.status)}</td></tr>`).join("") || `<tr><td colspan="3">缺少完整成本输入，暂不测算。</td></tr>`}</tbody></table></div>
    <p>费率：吃单 ${bps(c.fee?.taker_bps)} / 挂单 ${bps(c.fee?.maker_bps)}（单边）。读取于 ${time(c.fee?.fetched_at, true)}。<br>盘口 ${time(c.book_as_of, true)} · 成本有效至 ${time(c.valid_until)}。金额按中间价折算数量；未来退出盘口仍有不确定性，挂单成交尚未假定。</p>
    <details><summary>成本依据与历史留存</summary><p>${cost.missing_reasons.map(r => esc(REASONS[r] || r)).join(" · ")}</p><p>成本版本：${esc(cost.version)}。当前估计没有计入真实成交校准样本。</p>${c.historical_anchor ? `<p>历史留存：${time(c.historical_anchor.as_of, true)}，旧模型往返场景 ${bps(c.historical_anchor.roundtrip_bps)}。仅供核对，不用它代替当前成本。</p>` : "<p>本快照无旧模型锚点；原始历史数据继续保存在 NAS。</p>"}</details></section>`;
}

function renderForward() {
  const f = state.data?.forward;
  const items = [["已登记机会", f?.registered_opportunities], ["观察维度", f?.registered_horizon_observations], ["已成熟窗口", f?.matured_observations], ["等待观察", f?.waiting_observations]];
  $("forward-summary").innerHTML = `<dl class="summary-strip">${items.map(([k,v]) => `<div class="summary-item"><dt>${esc(k)}</dt><dd class="number">${num(v, 0)}</dd></div>`).join("")}</dl>
    <p class="flat-note">前向起点：${time(f?.started_at, true)}。同一机会的 4h 与 24h 是两个观察维度，不是两次独立交易。到期但缺行情的窗口：${num(f?.missing_label_observations, 0)}。<br>V5 消费回执：尚未接入。V5 账户增量收益：尚无同资金对照证据。</p>`;
  const groups = f?.by_group || [];
  $("forward-groups").innerHTML = groups.length ? `<div class="table-scroll"><table class="simple-table"><thead><tr><th>时域</th><th>参考动作</th><th>已成熟</th><th>成本场景净均值</th></tr></thead><tbody>${groups.map(g => `<tr><td>${esc(g.horizon_hours)}h</td><td>${esc(LABELS[g.action] || g.action)}</td><td>${num(g.observations,0)}</td><td>${bps(g.net_mean_bps)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="empty"><strong>尚无成熟的前向观察</strong>参考必须先发布，再经历固定的观察窗口；不会用旧历史结果填充这里。</div>`;
}

function renderRuntime() {
  const data = state.data;
  const pairs = [["最近分析", time(data?.generated_at, true)], ["历史小时线", `${num(data?.history_rows, 0)} 条`], ["分析进程峰值", `${num(data?.peak_rss_mib)} MiB`], ["本次计算", `${num(data?.runtime_seconds, 2)} 秒`], ["运行位置", data ? "NAS · 按次运行" : "—"], ["当前参考状态", data?.effective_status === "AVAILABLE" ? "已读取结果，按各条时效判断" : data?.effective_status === "EXPIRED" ? "结果已过期" : "等待结果"]];
  $("runtime-details").innerHTML = `<dl class="facts">${pairs.map(([k,v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl><details><summary>版本与任务证据</summary><p>实验：<code>${esc(data?.experiment_version || "trend-reference-1h-v1")}</code></p><p>版本：<code>${esc(data?.worker_commit || "尚未收到")}</code></p><p>结果：<code>${esc(data?.result_id || "尚未收到")}</code></p></details>`;
}

function render() {
  const data = state.data;
  const active = (data?.advice || []).filter(a => a.horizon_hours === state.horizon && effective(a) !== "NO_VIEW").length;
  $("notice").classList.toggle("warning", Boolean(state.error) || !active);
  $("notice").textContent = state.error || (data?.result_id ? `${state.horizon}h 参考：${active}/4 币当前有观点。最近分析 ${time(data.generated_at, true)}，请结合每条依据与失效时间。` : "尚未收到 NAS 的已校验结果；当前无观点。");
  renderReference(); renderForward(); renderRuntime();
  $("export").disabled = !data?.result_id;
  clearTimeout(state.expiryTimer);
  const now = Date.now() + state.offset;
  const next = Math.min(...(data?.advice || []).map(a => Date.parse(a.expires_at)).filter(t => t > now));
  if (Number.isFinite(next)) state.expiryTimer = setTimeout(render, Math.max(5, next - now + 5));
}

async function fetchReference(signal) {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      return await fetch("/v1/trade-advice/latest", {cache: "no-store", signal});
    } catch (error) {
      if (attempt || signal.aborted) throw error;
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }
}

async function refresh() {
  if (state.busy) return;
  state.busy = true; syncRefreshButton();
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetchReference(controller.signal);
    if (!response.ok) throw new Error(`接口暂不可用（HTTP ${response.status}）`);
    const data = await response.json();
    if (!Array.isArray(data.advice)) throw new Error("接口返回了无法识别的结果");
    state.data = data; state.error = "";
    state.offset = data.viewed_at ? Date.parse(data.viewed_at) - Date.now() : 0;
    render();
  } catch (error) {
    state.error = `更新失败，当前暂停显示有效观点。${error.name === "AbortError" ? "请求超时。" : error.message}`;
    render();
  } finally {
    clearTimeout(timer); state.busy = false; syncRefreshButton();
  }
}

document.querySelectorAll(".nav").forEach(button => button.addEventListener("click", () => setPage(button.dataset.page)));
document.querySelectorAll("[data-horizon]").forEach(button => button.addEventListener("click", () => {
  state.horizon = Number(button.dataset.horizon);
  document.querySelectorAll("[data-horizon]").forEach(node => {const selected = Number(node.dataset.horizon) === state.horizon; node.classList.toggle("selected", selected); node.setAttribute("aria-pressed", String(selected));});
  render();
}));
function syncFullscreen() {
  const active = Boolean(document.fullscreenElement);
  $("fullscreen").textContent = active ? "退出全屏" : "全屏";
  $("fullscreen").setAttribute("aria-pressed", String(active));
}
$("fullscreen").addEventListener("click", async () => {
  $("display-status").hidden = true;
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (document.fullscreenEnabled && document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
    else throw new Error("Fullscreen unavailable");
  } catch (_) {
    $("display-status").textContent = "浏览器未允许全屏，可使用浏览器菜单中的全屏功能。";
    $("display-status").hidden = false;
  }
  syncFullscreen();
});
document.addEventListener("fullscreenchange", syncFullscreen);
$("refresh").addEventListener("click", () => state.page === "servers" ? refreshServers() : refresh());
$("export").addEventListener("click", () => {
  if (!state.data?.result_id) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.data, null, 2)], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = `${state.data.result_id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
setInterval(() => {if (!document.hidden) {refresh(); if (state.page === "servers") refreshServers();}}, 30000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) {render(); refresh(); if (state.page === "servers") {renderServers(); refreshServers();}} });
refresh();
