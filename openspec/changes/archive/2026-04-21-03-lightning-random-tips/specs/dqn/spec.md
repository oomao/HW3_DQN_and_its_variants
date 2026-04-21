# Spec Delta: dqn (ADDED)

## ADDED Requirements

### Requirement: Lightning Training Loop
系統 SHALL 以 `pytorch_lightning.LightningModule` 封裝 Dueling Double DQN，
使用 `pl.Trainer` 驅動訓練。

#### Scenario: Gradient clipping
- **WHEN** `pl.Trainer(gradient_clip_val=1.0)` 設定
- **THEN** 每步更新的梯度 L2 norm 被裁剪到 ≤ 1.0

#### Scenario: Cosine LR schedule
- **WHEN** 訓練達到 `total_steps`
- **THEN** 學習率由 1e-3 退火至 1e-5（CosineAnnealingLR）

### Requirement: Soft Target Update
系統 SHALL 在每個訓練步以 Polyak averaging 更新 target 網路。

#### Scenario: Tau
- **WHEN** `tau=0.005` 時
- **THEN** `target ← (1 − τ) target + τ online`

### Requirement: Exponential Epsilon Schedule
系統 SHALL 對隨機動作的機率以指數衰減方式降低，加速收斂但保留必要探索。

#### Scenario: Decay formula
- **WHEN** 環境互動步數 `t`
- **THEN** `ε(t) = ε_end + (ε_start − ε_end) · exp(−3 · min(1, t/decay_steps))`

### Requirement: Random Mode Performance
系統 SHALL 於 Gridworld random mode 下的最後 300 回合達到勝率 ≥ 0.60。

#### Scenario: Smoke-test rollout
- **WHEN** 使用訓練完成的網路在 random mode 跑 1 回合 greedy rollout
- **THEN** 能在 `max_steps` 內抵達 goal（贏）或避開 pit（至少不在第一步送頭）
