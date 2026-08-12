### ITER-2026-08-13-FASTWAM-R5-CLOSEDLOOP-ABLATIONS-001 — PlaceFood R5 closed-loop replanning, oracle, and checkpoint evaluation

- 时间：2026-08-13T00:00:00+08:00
- 类型：Code_Evaluation_Experiment
- 动机：按用户要求，不再用 action loss 代替闭环结果；用受控的 exec_horizon、oracle 干预和 checkpoint 面板定位 PlaceFood 失败是接近误差还是闭爪时序主导。
- 变更：已在既有固定 rollout runner 中实现 exec_horizon=1/5 受控对照、robot0 pose/robot0 gripper/robot1 action 三种 expert temporal replay 干预、真实仿真 grasp 状态与肉块最大抬升量统计；保持场景 seed、policy seed、初始状态、评测步数和其他配置不变。断点续跑同时严格绑定 checkpoint 路径/字节数/mtime 与 policy seed，避免误复用旧结果。
- Git：分支 `rollout-eval-r5`，起点 commit `4ba4143a6d2bf1ce3a5830fd9f287fc9ec13d891`；当前只有一个既有未跟踪 workflow outbox，本轮不修改也不纳入提交。
- 实验：DSW `cwam-dsw970` 4×RTX4090；固定 8 个 PlaceFood validation seeds。真实已有 R5 checkpoint 仅 step500/1000；step250/750 当前缺失，禁止重命名、插值或伪造评测点。
- 结果：Implementation_In_Progress；本地实现已完成，加入 checkpoint 路径/字节数/mtime 与 policy seed 身份门后，在 DSW Python 3.10 环境重跑两组定向测试，14/14 PASS。尚未创建正式 Notion 记录或启动 rollout。
- 决策：先单 seed 冒烟验证 oracle 作用于实际 env action 且历史状态一致，再运行 8-seed 面板；只有仿真闭环成功、真实抓取事件和肉块抬升量才记为科学结果。
- 下一步：创建唯一 Notion Planned 实验页；提交并冻结代码；发布到独立 CPFS 路径；随后运行单 seed pilot 和 8-seed 正式面板。
- 记录人：Codex for chengjuntao
