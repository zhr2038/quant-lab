"use strict";
const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];
const LABELS = {DEFER: "等待", REVIEW_ENTRY: "复核入场", KEEP_BASELINE: "保持原规则", NO_VIEW: "暂无观点"};
const TITLES = {reference: "交易参考", forward: "效果观察", experiment: "实验与数据"};
const REASONS = {
  CURRENT_MARKET_MISSING: "缺少当前行情", CURRENT_MARKET_STALE: "当前行情已过期",
  HISTORY_CURRENT_MISMATCH: "历史与当前输入不一致", CONTEXT_WINDOW_INCOMPLETE: "趋势窗口有缺口",
  COST_MISSING: "缺少成本估计", COST_STALE_OR_UNDATED: "成本时间过旧或缺失",
  COST_TIMESTAMP_FUTURE: "成本时间异常", HISTORICAL_SAMPLE_INSUFFICIENT: "历史非重叠样本不足",
  RECENT_DIAGNOSTIC_SAMPLE_INSUFFICIENT: "近期诊断窗口样本不足", COST_NOT_PAPER_TRUSTED: "成本尚未通过模拟验证要求",
  HISTORICAL_COST_DRAG: "相似窗口的成本压力较大", POSITIVE_HISTORICAL_REFERENCE: "历史参考偏正",
  COST_REQUIRES_CALIBRATION: "需要进一步校准成交成本", HISTORICAL_REFERENCE_MIXED: "历史与近期结果不一致",
  FORWARD_VALUE_NOT_ESTABLISHED: "前向交易增益尚未建立", ADVICE_EXPIRED: "建议已到期"
};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, x => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
const num = (value, digits = 1) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN", {maximumFractionDigits: digits, minimumFractionDigits: digits}) : "—";
const bps = value => typeof value === "number" && Number.isFinite(value) ? `${value > 0 ? "+" : ""}${num(value)} bps` : "—";
const time = (value, full = false) => {
  if (!value || !Number.isFinite(Date.parse(value))) return "—";
  return new Intl.DateTimeFormat("zh-CN", {timeZone: "Asia/Shanghai", ...(full ? {month:"2-digit", day:"2-digit"} : {}), hour:"2-digit", minute:"2-digit", hour12:false}).format(new Date(value));
};
const state = {data: null, symbol: "BTCUSDT", horizon: 4, page: "reference", busy: false, error: "", offset: 0, expiryTimer: null, token: "", authRequired: false};
try { state.token = sessionStorage.getItem("qyun2.access") || ""; } catch (_) { /* Session storage may be disabled. */ }
const effective = advice => !advice || state.error || Date.now() + state.offset >= Date.parse(advice.expires_at) ? "NO_VIEW" : (advice.effective_action || advice.action);
const badge = action => `<span class="action ${esc(action)}">${esc(LABELS[action] || "暂无观点")}</span>`;
const metric = (title, value) => `<div class="metric"><dt>${esc(title)}</dt><dd class="number">${esc(value)}</dd></div>`;

function setPage(page) {
  state.page = page;
  $("page-title").textContent = TITLES[page];
  document.querySelectorAll(".page").forEach(node => { node.hidden = node.id !== `page-${page}`; });
  document.querySelectorAll(".nav").forEach(node => {
    node.classList.toggle("active", node.dataset.page === page);
    if (node.dataset.page === page) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current");
  });
}

function plot(distribution) {
  const values = [distribution.net_p10_bps, distribution.net_p50_bps, distribution.net_p90_bps];
  if (!values.every(x => typeof x === "number" && Number.isFinite(x))) return `<p class="empty">样本尚不足以形成分布。</p>`;
  const lo = Math.min(0, values[0]), hi = Math.max(0, values[2]);
  const span = Math.max(hi - lo, 1), x = value => 25 + (value - lo) / span * 240;
  return `<svg viewBox="0 0 290 105" role="img" aria-label="历史净结果第十、五十和九十分位"><title>当前成本假设下的历史分布</title>
    <line x1="25" y1="36" x2="265" y2="36" stroke="#e4e8ee" stroke-width="4"/>
    <line x1="${x(0)}" y1="12" x2="${x(0)}" y2="54" stroke="#c5cdd7" stroke-dasharray="3 3"/>
    <line x1="${x(values[0])}" y1="36" x2="${x(values[2])}" y2="36" stroke="#789be0" stroke-width="5"/>
    <circle cx="${x(values[1])}" cy="36" r="5" fill="#235cda"/>
    <text x="25" y="75" fill="#7c8797" font-size="10">P10 ${esc(num(values[0]))}</text>
    <text x="145" y="75" text-anchor="middle" fill="#365b9b" font-size="10">P50 ${esc(num(values[1]))}</text>
    <text x="265" y="75" text-anchor="end" fill="#7c8797" font-size="10">P90 ${esc(num(values[2]))}</text>
    <text x="145" y="96" text-anchor="middle" fill="#97a1af" font-size="9">单位 bps · 虚线为零</text></svg>`;
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
    <div class="distribution"><p>相似行情 · 当前成本下的历史分布</p>${plot(row.distribution)}</div>
    <div class="detail-list"><p><strong>行情截至</strong>　${time(row.market_asof, true)}</p><p><strong>历史范围</strong>　${time(row.distribution.first_signal_at, true)} — ${time(row.distribution.last_signal_at, true)}</p><p><strong>近期窗口净均值</strong>　${bps(row.distribution.chronological_tail_net_mean_bps)}</p><p><strong>双倍成本净均值</strong>　${bps(row.distribution.double_cost_mean_bps)}</p><p><strong>成本观测时间</strong>　${time(row.cost.as_of, true)}</p><p><strong>成本来源</strong>　${esc(row.cost.source)} · ${esc(row.cost.quality)}</p><p><strong>依据与限制</strong><br>${reasons.map(r => esc(REASONS[r] || r)).join(" · ")}</p><p><strong>失效条件</strong><br>${row.invalidation_conditions.map(esc).join(" · ")}</p></div>
    <details><summary>查看证据标识</summary><p>建议：<code>${esc(row.advice_id)}</code></p><p>输入：<code>${esc(row.input_snapshot_id)}</code></p><p>数据：<code>${esc(row.data_snapshot_hash)}</code></p></details>`;
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

async function refresh() {
  if (state.busy) return;
  state.busy = true; $("refresh").disabled = true; $("refresh").textContent = "更新中…";
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch("/v1/trade-advice/latest", {headers: state.token ? {Authorization: `Bearer ${state.token}`} : {}, cache: "no-store", signal: controller.signal});
    if (response.status === 401 || response.status === 403) {
      state.authRequired = true;
      state.data = null; state.error = ""; render();
      $("auth-panel").hidden = false; $("workspace").hidden = true; $("disconnect").hidden = true;
      $("auth-error").textContent = state.token ? (response.status === 403 ? "当前网络没有访问权限。" : "访问密钥未通过验证。") : "";
      return;
    }
    if (!response.ok) throw new Error(`接口暂不可用（HTTP ${response.status}）`);
    const data = await response.json();
    if (!Array.isArray(data.advice)) throw new Error("接口返回了无法识别的结果");
    state.data = data; state.error = ""; state.authRequired = false;
    state.offset = data.viewed_at ? Date.parse(data.viewed_at) - Date.now() : 0;
    $("auth-panel").hidden = true; $("workspace").hidden = false; $("disconnect").hidden = !state.token;
    $("token").value = ""; render();
  } catch (error) {
    state.error = `更新失败，当前暂停显示有效观点。${error.name === "AbortError" ? "请求超时。" : error.message}`;
    render();
  } finally {
    clearTimeout(timer); state.busy = false; $("refresh").disabled = false; $("refresh").textContent = "刷新";
  }
}

document.querySelectorAll(".nav").forEach(button => button.addEventListener("click", () => setPage(button.dataset.page)));
document.querySelectorAll("[data-horizon]").forEach(button => button.addEventListener("click", () => {
  state.horizon = Number(button.dataset.horizon);
  document.querySelectorAll("[data-horizon]").forEach(node => {const selected = Number(node.dataset.horizon) === state.horizon; node.classList.toggle("selected", selected); node.setAttribute("aria-pressed", String(selected));});
  render();
}));
$("refresh").addEventListener("click", refresh);
$("auth-form").addEventListener("submit", event => {event.preventDefault(); state.token = $("token").value.trim(); try {sessionStorage.setItem("qyun2.access", state.token);} catch (_) {} refresh();});
$("disconnect").addEventListener("click", () => {state.token = ""; state.data = null; try {sessionStorage.removeItem("qyun2.access");} catch (_) {} refresh();});
$("export").addEventListener("click", () => {
  if (!state.data?.result_id) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.data, null, 2)], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = `${state.data.result_id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
setInterval(() => {if (!document.hidden && !state.authRequired) refresh();}, 30000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) {render(); if (!state.authRequired) refresh();} });
refresh();
