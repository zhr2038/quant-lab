# 2026-09-06 复审发布记录

当前云端 API 和 NAS worker 的运行提交均为 `0907243e0c2685bbb3554dbc724e03f1a17bd566`。后续文档提交不代替该运行版本。main 和 `fix/review-remediation-20260906` 均已包含运行提交。

云端目录为 `/opt/quant-lab-releases/0907243e0c2685bbb3554dbc724e03f1a17bd566`，263个发布文件及生产依赖逐项校验通过；旧现场完整保存在 `/opt/quant-lab-preserved-before-review-20260906`。原运行配置保留，分析revision与NAS同步变更。代码目录和分析revision执行了回滚再前进演练，Gold发布指针未改写。

NAS镜像为 `quant-decision:0907243e0c2685bbb3554dbc724e03f1a17bd566`，镜像ID为 `sha256:ba7941a6cefbd5e49e0c6e18c746e8bf335934f9e8c4e93ee75e9dda89647fc9`，完整镜像归档SHA256为 `fcf1c45753a57ee90904671ca569085523326954451cd432b9073ef7d8646cbf`。镜像基于已验证的旧运行环境离线构建，依赖逐项匹配 `deploy/locks/nas-worker.txt`；自包含镜像归档保存在NAS版本目录，父镜像亦保留。

自然运行曾暴露旧v1签名归档的成本刷新兼容问题。最终补丁仅允许已验签的归档v1重放保留首次观察，明确输出 `LEGACY_V1_REPLAY_PRESERVED_FIRST_OBSERVATION`；新v2同机会版本冲突仍拒绝。真实副本验签重放102个发布回执、处理8个旧成本刷新，原104条观察逐条不变。没有通过恢复旧数据库、删除观察或修改签名结果修复问题。

新v2结果于2026-09-06 03:36:38北京时间通过云端验签发布。研究可评估、成本校准和真实资金准入明确分开；实际当前结果的 `live_execution_eligible` 均为false，原始估算成本类型及样本数量未伪造。实验、策略、成本版本和时间范围显式限定，价格标签不作为账户收益。V5以独立record-only消费器记录新参考，未切enforce；中台UI尚无V5回执回传，仍如实展示未接入。

[最终CI 33987368180](https://github.com/zhr2038/quant-lab/actions/runs/33987368180) 为663项通过，Ruff通过。Windows最终全量660项通过、3项平台条件跳过。测试包含估算研究候选可达、NO_VIEW失效条件、旧签名读回、同库相反实验隔离及旧v1重放兼容。Web实际检查了参考准入语义与版本范围展示。

净收益改善及前瞻统计晋级仍为证据不足。完整A01—C03验收表、文件哈希、V5同资金对照边界、失败记录、回滚点和未验证项见[V5联合交付报告](https://github.com/zhr2038/V5-prod/blob/main/docs/review-remediation-production-20260906.md)。回滚时须协调云端分析revision与NAS镜像，保留后续新增观察；不得用旧账本覆盖当前数据。
