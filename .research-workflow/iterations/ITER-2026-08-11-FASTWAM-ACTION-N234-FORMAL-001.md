### ITER-2026-08-11-FASTWAM-ACTION-N234-FORMAL-001 — Action-only N=2/3/4 independent 1x8 formal training

- 时间：2026-08-11T13:00:34+08:00
- 类型：Code_Config_Validation_Experiment
- 状态：In Progress / Prelaunch
- 动机：按用户授权，把当前 action-only、VideoGen=0、HubToken=1、Gaussian=1、native variable-N 实现整理为可提交版本；先完成一次 1 Worker x 8 GPU 的 save/fresh-load Gate2，再并行启动 N=2、N=3、N=4 三条正式 1000-step 训练。
- 科学矩阵：N=2 使用 PlaceFood-rf；N=3 使用 ThreeRobotsPlaceShoes-rf 与 ThreeRobotsStackCube-rf 的 count-level pool；N=4 使用 FourRobotsStackCube-rf。三组 seed 与 split_seed 均为 42，不使用固定容量 masked agent set。
- 初始 Git：分支 `exp/multi-robot-gaudp`，HEAD `2823fd94f18c42d19646b350c13695e23afb60f3`，工作树包含此前未提交的 action-only/no-hash 实现；未经审查不丢弃、不覆盖。计划建立独立分支，完成定向测试后形成 clean pushed commit。
- 数据与资产：RoboFactory multi-robot 数据、N=2/3/4 seed-42 split、文本缓存、Gaussian compact/canonical cache 和 step-5000 初始化权重均已只读确认位于 OSS；正式源码与结果也使用唯一 OSS 前缀。
- 资源与提交：PAI workspace `270969`，resource `quotaksvqq2oh2pg`；每条作业固定 1 Worker x 8 GPU。当前 workspace 无 active 作业，但实时物理卡余量由 PAI 调度决定。
- Gate 顺序：先完成源码发布、配置/合同测试、Notion Planned 页和 Gate2；只有 Gate2 结构化终态 PASS 才提交三条正式训练。Running 只表示基础设施状态，不代表训练或科学结果完成。
- 预期产物：四个唯一实验身份（Gate2、N2、N3、N4）、Notion page IDs、DLC Job IDs、可续训 checkpoint、训练日志、终态状态与 task/count 级指标。禁止用日志渲染片段代替结构化 Gate2 receipt。
- 记录策略：遵循当前 no-hash 决策；只记录 Git commit、唯一 source/output 路径、时间、文件数量/大小、作业和运行 ID，不新增 hash 或 hash-chain gate。
- 当前门禁：N3/N4 配置、通用 native-agent 1x8 合同、Gate2 OSS source/VAE 绑定、正式三阶段 launcher 与 Notion Planned 记录均已完成。代码层独立审计 P0 已清零；正式 N=2/3/4 仍严格等待同挂载、同 1x8 拓扑 Gate2 的真实结构化终态 PASS，以及提交时平台权威可用容量不少于 190 GiB。
- Notion 预注册：已在写入前对 `FASTWAM-MR-FT-ACT-N234-GATE2-1X8-S42-R1-20260811`、`FASTWAM-MR-FT-ACT-N2-PLACEFOOD-1K-S42-R1-20260811`、`FASTWAM-MR-FT-ACT-N3-POOL-1K-S42-R1-20260811`、`FASTWAM-MR-FT-ACT-N4-STACKCUBE-1K-S42-R1-20260811` 完成 exact-ID 去重，待幂等创建为 `Planned`。Git commit、推送和 OSS 唯一源快照仍是明确的 prelaunch pending 项，不在页面中虚构。
- Notion 回读：四个 exact-ID 均唯一命中且状态为 `Planned`；页面正文均未截断、无未知块。Gate2 页面 ID `3b921e77-89cc-81ca-96d3-d23164b6a446`，N2/N3/N4 页面 ID 依次为 `3b921e77-89cc-8157-bc07-ece6281b4c8d`、`3b921e77-89cc-81c1-88a0-c1bc2d0fe4cf`、`3b921e77-89cc-81fd-98bb-e2a1e7effd47`。本次未调用 CreateJob，Git clean pushed commit、OSS source snapshot 和 DLC Job ID 仍保持 pending。
- 提交前验证：SSH970 上使用项目训练依赖环境，对当前完整 `tests/` 执行测试，结果 `390 passed, 2 warnings`；两个 warning 分别来自 `pynvml` 弃用提示与 Hydra `_self_` 顺序提示。formal launcher 的 Python compile、shell syntax、静态/动态 fail-close 测试均 PASS；Gate2 controller/publisher 定向测试均 PASS。远端临时 Git fixture 仅用于满足既有测试的 `git rev-parse` 前置，不作为正式源码身份。
- 审计结论：允许先提交一次 1 Worker x 8 GPU Gate2；禁止绕过 Gate2 直接提交 N2/N3/N4。Gate2 必须实际完成 CUDA save/resume/fresh-load 三个 8-rank world、OSS `O_EXCL` 发布后 close/reopen 逐字节回读，并以 `COMPLETE` 与结构化 receipts 判定。正式 suite 仅在该终态 PASS 后放行。
- 记录人：Codex for chengjuntao
