# Tasks: 03-lightning-random-tips

## 1. Lightning 封裝
- [x] 1.1 `DQNLightning(pl.LightningModule)`
- [x] 1.2 自訂 `ExperienceStream(IterableDataset)` + `DataLoader`
- [x] 1.3 `configure_optimizers` 回傳 Adam + CosineAnnealingLR

## 2. 訓練技巧
- [x] 2.1 `gradient_clip_val=1.0`
- [x] 2.2 Target network soft update（τ=0.005，每步）
- [x] 2.3 ε 指數衰減
- [x] 2.4 Warmup buffer 500 步後再開始更新

## 3. 結果
- [x] 3.1 `artifacts/lightning_rewards.png` + `artifacts/lightning_winrate.png`
- [x] 3.2 random mode 最後 300 回合勝率 ≥ 0.60（random mode 上限較低）
- [x] 3.3 rollout GIF 能從隨機起點走到 goal
