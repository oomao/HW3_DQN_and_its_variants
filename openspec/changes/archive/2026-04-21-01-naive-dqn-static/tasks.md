# Tasks: 01-naive-dqn-static

## 1. 環境包裝
- [x] 1.1 vendor Gridworld/GridBoard 並加上 attribution
- [x] 1.2 `GridworldEnv.reset()` / `step(int)` 回傳 `(state, reward, done, info)`

## 2. DQN 與 Replay Buffer
- [x] 2.1 Listing 3.2 的三層 DQN（`64→150→100→4`）
- [x] 2.2 `ReplayBuffer`：`push` / `sample(batch_size)`、capacity 1000

## 3. 訓練
- [x] 3.1 Linear ε 衰減、Adam(1e-3)、γ=0.9、batch=200
- [x] 3.2 同時比較「無 replay」與「有 replay」
- [x] 3.3 輸出 `artifacts/naive_rewards.png`、`artifacts/naive_winrate.png`
- [x] 3.4 rollout GIF `artifacts/rollout_naive_static.gif`

## 4. 指標
- [x] 4.1 有 replay 版本於最後 200 回合勝率 ≥ 0.95
