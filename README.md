# quant-lab 交易参考中台

2026-09-05 完成精简后，首版重建聚焦一个问题：四币在相似趋势和当前往返成本下，值得复核候选还是等待。NAS 计算历史参考并记录发布后的观察，qyun2 提供有来源、有效期与限制的研究建议。旧 Web、Alpha/Factor Factory、专题研究、自动 AI、自动专家包及专用 NAS worker 已退役。

中台不下单、不撤单、不维护真实仓位，不替代 V5 的执行、reconcile 或 kill-switch。当前兼容接口不代表研究有效、实盘就绪或盈利能力。

## 当前职责

| 位置 | 保留的工作 |
| --- | --- |
| qyun2 | BTC、ETH、SOL、BNB 的公开行情；只读成交/账单；有界 V5 遥测导入；闭合 K 线特征；成本与风险许可；HTTP 读接口 |
| NAS SSD `/volume1/docker/quant-decision` | 四币一年以内小时线工作集，512 MiB 上限；不存完整历史任务目录 |
| NAS HDD `/volume2/quant-lab/archive` | 长期历史、退役任务及结果、校验清单、软件版本和受限恢复资料 |
| V5 | 交易执行、账户与真实仓位、风控、已有 Paper 状态及退出 |

云端保留 API、WebSocket 两项常驻业务服务、8 项底座定时任务，加 1 项轻量参考发布任务。Web 复用 API 的静态页面。NAS 只有一类按次退出的分析容器，最多 3 CPU / 4 GiB，不增加 swap 配额，预留至少 6 GiB 可用内存才启动。输入、历史快照、结果、前向观察存于 HDD `/volume2/quant-lab/decision/archive`。

## API 与旧消费者

- `/v1/health`、`/v1/catalog/datasets`、行情/特征/成本/风险许可接口继续提供服务。
- 旧 advisory、Paper proposal/status/promotion 和 canary 读接口作为过渡兼容层保留。它们读取历史发布结果，保持原始身份、过期时间及安全语义；不续期、不生成新提案、不升级 live 权限。响应头 `X-Quant-Lab-Legacy-Producer: retired-2026-09-05` 标明生产者已退役。
- `/` 为新交易参考 Web；`GET /v1/trade-advice/latest` 和按 `advice_id` 查询的详情复用现有鉴权。在线请求只读 Gold 紧凑发布结果。旧 `/web-v2` 页面和专家包操作仍返回 HTTP 410。
- 遥测仅沉淀事实和运行健康。显式请求旧 candidate Gold 生成会在写入前报错。

新参考有 4h / 24h 两个观察时域；动作为等待、复核入场、保持原规则或暂无观点。全部为 `research_only`、`live_order_effect=none`。历史净均值扣除当前 20 USDT 名义金额的往返成本假设，不是预测胜率或账户收益。2026-09-06 的 v2 契约区分研究可评估、成本已校准、真实执行资格；完整且有效的估算成本可支持独立研究候选，仍不冒充实盘成交样本。V5 新消费器只落盘，自己的同资金对照独立记账；中台未接入 V5 回执上传和账户收益，页面不能显示已自动采纳。设计、部署及回滚见 [decision-workbench.md](docs/decision-workbench.md)。

## 数据与恢复

本次归档批次：`192.168.1.15:/volume2/quant-lab/archive/retirement-20260905`。

- `nas-ssd/quant-research`、`quant-export`、`quant-ai`、`quant-runtime`：旧运行目录的数据副本，包含历史结果、队列和失败记录，保留硬链接与元数据。
- `nas-ssd/quant-archive.tar-parts/`：旧高频和 V5 归档按日期及目录分批存为 TAR，避免在机械盘重复铺开约 177 万个小文件。`index.json` 将原始相对路径映射到归档包；每包附逐成员校验清单和完成回执。
- `qyun2/data/`：云端同一截止点的完整数据副本；截止时间以 `snapshot-status.json` 为准。
- `*-manifest.jsonl`、`*-verified.json`：逐文件 SHA256、源/目标映射和校验结果。TAR 内每个文件重新读取校验，扩展属性另外保存在成员清单中。文件存在不等于已经通过校验，以完成回执为准。
- `private/` 及云端 `private-config/`：受限的运行配置与镜像恢复资料，不对 Web 或下载服务公开。
- 清理前代码：Git 提交 `33af8e23eec8018f3786dc46755b4035778053cf` 与 `quant-lab-before.bundle`。

正在运行的源数据不随代码删除。SSD 原件清理须同时具有完整校验回执、读取恢复验证以及无活跃写入者的证据。归档中的旧质量等级和旧 PASS 结论不升级为当前可用建议。

长期归档任务改用 `/volume2/quant-lab/archive/current/qyun2`，独立于冻结的退役批次；不再自动删除 45 天前的 NAS 历史。高频源清理仍必须逐批匹配校验清单，并遵守原有锁和范围检查。

## 开发与验证

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest -q
qlab --help
```

采集、成本、数据格式、时间规则、权限安全与既有 V5 契约的测试继续保留。退役产品的测试随对应代码移出当前版本，可从旧提交完整恢复。

精确退役单元、容器和回滚提交见 `deploy/retirement-20260905.json`。`docs/` 中旧设计与验收记录是历史资料，不能据此启动已退役的服务。生产回滚需先恢复代码、环境和目标模块配置，再选择性恢复服务；不要覆盖当前湖数据或 V5 状态。
