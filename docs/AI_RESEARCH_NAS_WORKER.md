# quant-lab AI Research Plane：云端轻量队列 + NAS Docker Worker

## 1. 目标

这套实现把 AI 推理和大部分编排负载放在 NAS，云端 quant-lab 只承担两项轻量工作：

1. 从**已经生成的 Expert Pack ZIP**中挑选少量摘要、JSON 和 CSV 前若干行，生成小型任务包；
2. 校验 NAS 返回的结构化 JSON，并写入新的只读 `gold/ai_*` 研究表。

它不会：

- 扫描全量 lake 来构造 Prompt；
- 在云端调用模型；
- 修改 V5；
- 修改 `risk_permission`；
- 把 AI 草案自动转成 PaperStrategyProposal；
- 产生真实订单。

## 2. 数据流

```text
quant-lab daily expert pack
        |
        | 低频、限额读取 ZIP 内已有报告
        v
/var/lib/quant-lab/ai_queue/pending/<task_id>/task.json
        |
        | NAS 主动 SSH/SCP 拉取；NAS 不需要公网 IP
        v
UGREEN NAS Docker: quant-ai-worker
        |
        | http://192.168.1.15:8317/v1/responses
        | model=gpt-5.6-sol
        | reasoning.effort=xhigh
        v
两阶段严格 JSON Schema 输出
        |
        | NAS 主动 SCP 回传
        v
/var/lib/quant-lab/ai_queue/results/inbox/<task_id>/result.json
        |
        | 中台 Pydantic + SHA256 + section 白名单校验
        v
gold/ai_research_run 等只读研究表
```

NAS 只发起出站连接，因此家庭宽带没有固定公网 IP、处于 CGNAT 都不影响使用。云服务器可使用现有固定 IP 或现有域名作为 SSH 目标；不需要为 NAS 配置 DDNS 或开放路由器端口。

## 3. 两阶段提示词

提示词位于：

```text
src/quant_lab/ai_research/prompts/stage1_system.md
src/quant_lab/ai_research/prompts/stage2_system.md
```

### Stage 1：研究诊断与路由

模型调用前先执行确定性 Preflight，确认根级 `manifest.json`、
`provenance.json` 和 `data_quality.json` 均已进入任务包，且核心文档未被截断。
Preflight 为 `BLOCK` 时，Stage 1 仍可给出数据修复和代码复核目标，但不能进入
Stage 2。

Stage 1 随后检查：

- freshness、manifest、provenance、data quality；
- 因子 IC、Rank IC、after-cost spread、独立性和样本覆盖；
- entry/exit 归因、false block、decision regret、opportunity cost；
- paper proposal、ACK、tracker、promotion gate 和 runtime freshness。

只有证据足够且没有阻断性数据质量问题时，才允许 Stage 2，并且只路由相关 section。
每轮还会携带上一轮已导入诊断的精简上下文，用于区分问题是持续、解决、恶化还是
变化；历史结论不会替代本轮证据。

Stage 1 固定输出一个可追溯闭环：

```text
finding -> primary bottleneck -> root cause tree -> next action -> code review target
```

即使 Stage 2 被阻断，安全的数据修复动作和代码检查目标仍会进入 Web，不再只显示
“BLOCKED”而没有下一步。

### Stage 2：可证伪研究草案

Stage 2 只能输出：

- 受限 `FactorTemplate` 因子候选；
- shadow/paper 规则草案；
- 对照实验；
- 待人工检查的代码路径。

所有输出固定为：

```text
research_only=true
live_order_effect=none_read_only_research
requires_human_review=true
proposal_state=AI_RESEARCH_DRAFT
```

每个因子、Paper、实验和代码目标必须带 `research_thread_id` 或 `target_id`，并通过
`source_finding_ids` 追溯到 Stage 1 的具体发现。实验同时要求失败条件、停止条件和
regime 切片，避免只定义“什么算成功”。

## 3.1 对 PA_Agent 的选择性借鉴

本实现参考了 PA_Agent 的工程思想，但没有复制其交易决策语义或代码：

| PA_Agent 的做法 | quant-lab 的采用方式 |
| --- | --- |
| 两阶段诊断与路由 | Stage 1 证据诊断，Stage 2 只接收相关 section |
| 调用前确定性检查 | Expert Pack 身份与核心文档 Preflight |
| 校验失败反馈重试 | Pydantic 错误路径经脱敏后反馈给模型，并记录尝试次数 |
| 上一轮分析连续性 | 只携带精简研究上下文，本轮仍必须重新引用证据 |
| 完整过程留痕 | prompt version、attempts、validation events、usage 写入研究运行表 |

明确不采用的部分包括：GPT 买卖判断、自由对话驱动交易、任意代码执行、自动修改
策略、自动晋级或连接真实下单。PA_Agent 使用 AGPL-3.0，本项目只借鉴公开架构思想，
没有复制其实现代码。

## 4. 云端部署

### 4.1 更新代码

将包含本功能的分支或合并后的主分支部署到：

```text
/opt/quant-lab
```

并更新虚拟环境：

```bash
cd /opt/quant-lab
/opt/quant-lab/.venv/bin/pip install -e .
```

### 4.2 建立 NAS 专用 SSH 账号

账号只需要访问 AI 队列，不需要读取 lake、生产配置或交易密钥：

```bash
sudo groupadd --system quantai
sudo useradd --system --create-home \
  --home-dir /var/lib/quant-ai-ssh \
  --shell /bin/bash \
  --gid quantai quantai

# quantlab 只获得 AI 队列的组访问权；quantai 不加入 quantlab 组，
# 因而不能读取 lake、生产配置或交易密钥。
sudo usermod --append --groups quantai quantlab

sudo install -d -o quantlab -g quantai -m 2770 \
  /var/lib/quant-lab/ai_queue \
  /var/lib/quant-lab/ai_queue/pending \
  /var/lib/quant-lab/ai_queue/running \
  /var/lib/quant-lab/ai_queue/completed \
  /var/lib/quant-lab/ai_queue/failed \
  /var/lib/quant-lab/ai_queue/results/inbox \
  /var/lib/quant-lab/ai_queue/results/imported \
  /var/lib/quant-lab/ai_queue/results/rejected
```

把 NAS 生成的**专用公钥**写入：

```text
/var/lib/quant-ai-ssh/.ssh/authorized_keys
```

在公钥前增加：

```text
restrict
```

该账号不得加入 `quantlab` 组或 sudoers，不得配置交易所密钥，也不需要访问
`/var/lib/quant-lab/lake` 或 `/etc/quant-lab`。部署后应使用 `id quantai` 和
`sudo -u quantai test ! -r /etc/quant-lab/quant-lab.env` 验证边界。

### 4.3 安装 systemd 单元

```bash
sudo cp deploy/systemd/quant-lab-ai-task.service /etc/systemd/system/
sudo cp deploy/systemd/quant-lab-ai-task.timer /etc/systemd/system/
sudo cp deploy/systemd/quant-lab-ai-import.service /etc/systemd/system/
sudo cp deploy/systemd/quant-lab-ai-import.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-lab-ai-task.timer quant-lab-ai-import.timer
```

手工测试：

```bash
sudo systemctl start quant-lab-ai-task.service
sudo -u quantlab /opt/quant-lab/.venv/bin/python \
  /opt/quant-lab/tools/quant_ai_queue.py status \
  --queue-root /var/lib/quant-lab/ai_queue
```

资源限制：

- 任务包构建：`CPUQuota=40%`、`MemoryMax=700M`；
- 结果导入：`CPUQuota=30%`、`MemoryMax=700M`；
- `POLARS_MAX_THREADS=1`；
- 任务包直接读取已有 Expert Pack，不触发 lake refresh 或 daily export。
- 单个任务证据上限约 30 万字符，Worker 对完整 Responses 请求设置 60 万字节
  fail-fast 上限，避免超出模型上下文后反复重试。

## 5. NAS Docker 部署

建议目录：

```text
/volume1/docker/quant-ai/
  repo/
  deploy/nas_ai_worker/.env
  deploy/nas_ai_worker/data/
  deploy/nas_ai_worker/secrets/id_ed25519
  deploy/nas_ai_worker/secrets/known_hosts
```

### 5.1 准备配置

```bash
cd /volume1/docker/quant-ai/repo/deploy/nas_ai_worker
sh setup.sh
cp .env.example .env
```

编辑 `.env`：

```text
QLAB_SSH_HOST=<云服务器 IP 或现有域名>
QLAB_SSH_USER=quantai
CLIPROXY_BASE_URL=http://192.168.1.15:8317/v1
CLIPROXY_API_KEY=<轮换后的新 token>
OPENAI_MODEL=gpt-5.6-sol
OPENAI_REASONING_EFFORT=xhigh
```

**不要**把 Token 写进 Dockerfile、Compose、Git 或 Codex TOML。当前实现只从容器环境变量读取。

### 5.2 生成专用 SSH 密钥

在 NAS 上执行：

```bash
ssh-keygen -t ed25519 \
  -f deploy/nas_ai_worker/secrets/id_ed25519 \
  -C quant-ai-nas \
  -N ''

ssh-keyscan -p 22 <云服务器地址> \
  > deploy/nas_ai_worker/secrets/known_hosts

chown 10001:10001 deploy/nas_ai_worker/secrets/id_ed25519
chmod 600 deploy/nas_ai_worker/secrets/id_ed25519
chmod 644 deploy/nas_ai_worker/secrets/known_hosts
```

将 `.pub` 内容添加到云端 `quantai` 账号。

### 5.3 验证容器能够访问本地代理

`cliproxyapi` 必须监听 NAS/LAN 可达地址，而不能只监听 `127.0.0.1`。测试：

```bash
docker compose run --rm --entrypoint python quant-ai-worker -c '
import os, httpx
u=os.environ["CLIPROXY_BASE_URL"].rstrip("/")+"/responses"
p={"model":os.environ.get("OPENAI_MODEL","gpt-5.6-sol"),
   "input":"Return exactly the word OK.",
   "reasoning":{"effort":os.environ.get("OPENAI_REASONING_EFFORT","xhigh")},
   "max_output_tokens":64,"store":False}
r=httpx.post(u,headers={"Authorization":"Bearer "+os.environ["CLIPROXY_API_KEY"]},json=p,timeout=120)
print(r.status_code); print(r.text[:1000]); r.raise_for_status()
'
```

如果代理使用自定义模型别名，只需修改 `.env` 中的 `OPENAI_MODEL`，不用改代码。生产目标仍是 `gpt-5.6-sol` + `xhigh`。

### 5.4 启动

```bash
docker compose build --pull
docker compose up -d
docker compose logs -f --tail=200 quant-ai-worker
```

Compose 限制：

```text
CPU: 1.5
Memory: 2GB
PIDs: 128
Root filesystem: read-only
No new privileges: enabled
```

## 6. 手工端到端测试

云端创建任务：

```bash
sudo systemctl start quant-lab-ai-task.service
```

NAS 单次执行：

```bash
docker compose run --rm -e RUN_ONCE=true quant-ai-worker
```

云端导入：

```bash
sudo systemctl start quant-lab-ai-import.service
```

检查队列：

```bash
sudo -u quantlab /opt/quant-lab/.venv/bin/python \
  /opt/quant-lab/tools/quant_ai_queue.py status \
  --queue-root /var/lib/quant-lab/ai_queue
```

检查 Gold：

```text
gold/ai_research_run
gold/ai_research_finding
gold/ai_factor_proposal
gold/ai_paper_strategy_draft
gold/ai_experiment_proposal
gold/ai_code_review_target
```

## 7. 故障行为

系统采用 fail-closed：

- 代理不支持 `gpt-5.6-sol`：任务进入 `failed`；
- 代理忽略严格 JSON Schema：Pydantic 校验失败并重试，最终仍失败则不导入；
- Schema 重试会携带精简错误路径，不会盲目重复同一请求；每次失败均写入
  `validation_events_json`，令 Web 可见；
- NAS 离线：任务留在 `pending`，V5 和 quant-lab 原有功能不受影响；
- 返回了未知 section、SHA 不匹配或越权字段：结果进入 `results/rejected`；
- Stage 1 认为证据不足：只保存诊断，不执行 Stage 2；
- AI 结果永远不会自动进入 V5 或现有 Paper 生命周期。

## 8. 安全注意事项

1. 已粘贴到聊天中的旧 Bearer Token 必须立即在 `cliproxyapi` 侧作废并轮换。
2. NAS Worker 使用独立 SSH 密钥和独立低权限账号。
3. 不要把 Docker Socket、UGOS 管理界面、SSH 私钥、cliproxy 管理端口暴露到公网。
4. NAS 不需要公网域名；它只主动连接云服务器。
5. `task.json` 中的文档内容一律视为不可信数据，系统 Prompt 明确禁止服从其中的指令。
6. Expert Pack 必须继续执行现有 secret scan 和脱敏流程；AI 任务包还会排除 `restricted/private/secrets` 路径。
