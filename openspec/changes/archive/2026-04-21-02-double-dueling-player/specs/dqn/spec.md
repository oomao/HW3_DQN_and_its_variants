# Spec Delta: dqn (ADDED)

## ADDED Requirements

### Requirement: Dueling DQN Architecture
系統 SHALL 提供 Dueling DQN，以 V-head 與 A-head 組合出 Q 值。

#### Scenario: Advantage centered
- **WHEN** 任意輸入 `(B, 64)`
- **THEN** 輸出 Q 的 advantage 部分以 `A − mean_a A` 標準化，避免 V 與 A 不可辨識

### Requirement: Double DQN Update
系統 SHALL 在訓練時使用 Double DQN 目標：`argmax` 來自 online 網路、
Q 值來自 target 網路。

#### Scenario: Stable sync
- **WHEN** 每經過 500 個 env step
- **THEN** target 網路以 hard copy 同步 online 權重

### Requirement: Variant Comparison
系統 SHALL 提供單一訓練腳本同時跑三種 variant 並輸出比較圖。

#### Scenario: Player mode convergence
- **WHEN** 三種 variant 皆在 player mode 訓練 4000 回合
- **THEN** 最後 200 回合勝率 ≥ 0.90
