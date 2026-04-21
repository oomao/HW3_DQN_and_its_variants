# Proposal: PyTorch Lightning + Training Tips on Gridworld (random)

## Why
HW3-3 要求把 DQN 改寫成 **PyTorch Lightning**（或 Keras），並在 **random** 模式
（全部物件位置皆隨機）下訓練。Random mode 明顯比 static / player 更難，
因此需要額外的穩定性技巧（梯度裁剪、學習率排程、target soft update、
ε 指數衰減等）才有機會收斂。

## What Changes
- 新增 `src/hw3/train_lightning.py`：
  - `DQNLightning(LightningModule)` 把 Dueling Double DQN 包進標準的
    `training_step` / `configure_optimizers`。
  - 自訂 `ExperienceStream(IterableDataset)` 搭配 `DataLoader` 餵 mini-batch；
    每個 training step 額外收 1 筆 env transition 推進 buffer。
  - 使用 `Trainer(gradient_clip_val=1.0)` 做梯度裁剪。
  - `CosineAnnealingLR` 讓學習率從 1e-3 退火到 1e-5。
  - Target network soft update（τ=0.005）每步更新。
  - ε 指數衰減：`ε = ε_end + (ε_start − ε_end) · exp(−3 · min(1, t / decay))`。
- 輸出 `artifacts/lightning_rewards.png`、`artifacts/lightning_winrate.png`、
  `artifacts/rollout_lightning_random.gif`。

## Impact
- 新 requirement：Lightning 封裝、穩定性技巧。
- 訓練時間 < 8 分鐘（CPU，12000 env step）。
