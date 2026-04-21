# Tasks: 02-double-dueling-player

## 1. 模型
- [x] 1.1 `DuelingDQN`：trunk `64→150→100`，分支 `V: 100→1`、`A: 100→4`
- [x] 1.2 Q = V + (A − mean_a A)

## 2. Target network 工具
- [x] 2.1 `hard_update(target, source)`
- [x] 2.2 `soft_update(target, source, tau)`（HW3-3 用）

## 3. 訓練流程
- [x] 3.1 三種 variant 共用一個訓練函式
- [x] 3.2 Double/Dueling 的目標：`a* = argmax Q_online(s')`，`y = r + γ Q_target(s', a*)`
- [x] 3.3 每 500 步 hard sync target

## 4. 結果
- [x] 4.1 `artifacts/variants_rewards.png`、`artifacts/variants_winrate.png`
- [x] 4.2 三種 variant 於 player mode 的最後 200 回合勝率 ≥ 0.90
- [x] 4.3 Dueling DQN 的 rollout GIF
