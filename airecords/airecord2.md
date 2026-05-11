# AI 對話紀錄 #2 — HW3-4（加分題） Rainbow DQN

**日期:** 2026-05-11
**Model:** Claude Opus 4.7 (1M context) via Claude Code
**專案:** https://github.com/oomao/HW3_DQN_and_its_variants
**前一個 session:** [airecord1.md](airecord1.md)（HW3-1/2/3 主作業）

---

## 使用者初始需求

延續 airecord1 完成的 HW3-1/2/3，老師加開加分題：
> 「用 Rainbow DQN 解 random mode gridworld」

老師原始提示：
1. static mode 用基本款
2. player mode 解不出來才要用「高級」
3. random mode 用到可以解就好

使用者偏好：
- 「先分析，再 tutorial me」（先講原理再做）
- README 寫進同一頁
- Commit / push **不帶 Claude trailer**
- 「做你覺得最容易加分的方式」（後改為要求完整 Rainbow）
- 對話也要記錄到 ai_record（這個檔）

## 關鍵決策

| 問題 | 使用者選擇 |
|---|---|
| 走 Rainbow-Lite (4/6) 還是完整 Rainbow？ | **完整 Rainbow** — 加 C51（B 選項）|
| Headline 配置 | 「Rainbow DQN（去除 NoisyNet）」名義為主、ablation 為輔 |
| README 章節 | 同一頁，§五 全部寫進去 |
| ai_record 形式 | 拆成 `airecords/airecord1.md` + `airecord2.md` |

## 設計分析（對話中先給的 tutorial 結論）

Rainbow 6 個組件對 random gridworld 的預期效益：

| 改進 | 對 random gridworld 的效益 | 初版採用？ | 最後採用？ |
|---|---|---|---|
| Double DQN | 中 | ✅ HW3-2/3 已有 | ✅ |
| Dueling DQN | 中 | ✅ HW3-2/3 已有 | ✅ |
| Prioritized Experience Replay | **高** | ✅ 加 | ✅ |
| Multi-step / n-step return | **高** | ✅ 加（n=3）| ✅ |
| Distributional / C51 | 預測低（reward 退化）| ❌ 不做 | ✅（B 選項後加入）|
| Noisy Networks | 中（替代 ε-greedy）| ⚠️ ablation | ⚠️ ablation |

## 執行步驟

### Phase 0：基礎結構

```
src/hw3/
├── prioritized_replay.py   # 新增：sum-tree PER，α=0.6, β 0.4→1.0 anneal
├── nstep.py                # 新增：n-step transition 累積器
├── models.py               # 加：NoisyLinear + RainbowDQN
└── train_rainbow.py        # 新增：Lightning 訓練腳本，--noisy 切換 NoisyNet
```

關鍵實作細節：
- PER 用 sum-tree O(log N) 抽樣 / 更新；max_priority 自適應、新樣本用 max_priority 寫入
- n-step：deque(maxlen=n)，episode 結束時把剩下 n−1 筆 partial-horizon transition 一併 flush（避免漏掉終局 reward）
- Huber (smooth_l1) loss + IS-weight，每個 batch 結束寫回新 priority
- NoisyNet：trunk 保留普通 Linear，只在 value/advantage head 用 NoisyLinear；target net 強制 `eval()` 移除 noise（穩定 bootstrap 目標）

### Phase 1：第一次完整訓練（25k steps，含 NoisyNet）

**踩到第一個坑：勝率反而從 0.46（初期）跌到 0.21（末期），比 HW3-3 的 0.64 慘很多。**

成因排查：
- 初版 `sigma_init=0.5`，再加上每個 train step 都 `reset_noise()` 在 online 與 target 上 → 目標太不穩
- 修正 1：`target.eval()` 永遠不加 noise → 仍然 0.27，沒解決
- 修正 2：online 只在 acting 時 reset_noise，train_step 不重抽 → 仍然 0.27
- 修正 3：`sigma_init` 0.5→0.17（C51 paper 對小 env 的設定）→ 改善到 0.37，但還是輸 baseline

### Phase 2：拔掉 NoisyNet（先暫時走 4/6 Rainbow-Lite）

回到 ε-greedy 探索，其他全保留（PER + n-step + Dueling + Double + Huber + IS-weight）。

| Agent | episodes / 25k steps | last-300 win rate |
|---|---:|---:|
| HW3-3 · Dueling Double DQN（baseline）| 1074 | 0.643 |
| **4/6 Rainbow-Lite (PER + n=3)** | **1224 (+14%)** | **0.653** |
| 4/6 + NoisyNet（ablation）| 812 | 0.367 |

兩個重點：
1. 勝率小幅提升（0.64 → 0.65）
2. **更有意義的指標：同樣 25k env step 跑了 1224 回合（+14%）** = trajectory 變短 = 策略更直接

到這裡為止，我推薦使用者用 Rainbow-Lite (4/6) push。

### Phase 3：使用者要求補上 C51（5/6 → 6/6 完整 Rainbow）

我給三個選項（A 接受 Rainbow-Lite 名稱、B 加 C51 變 5/6、C 直接改名為 Rainbow DQN）。
使用者選 **B**：想要「Rainbow DQN（去除 NoisyNet）」名義 = 5/6 with C51。

加上 C51 的部分：
```
src/hw3/models.py
├── CategoricalRainbowDQN     # Dueling + softmax over n_atoms per action
└── project_distribution()    # Bellman 投影到固定 support，含 eq_mask 處理整數 bin

src/hw3/train_rainbow.py
├── _step_distributional()    # cross-entropy loss + Double DQN action 用 Q 期望值
└── --no-distributional flag  # 退回原 Huber/MSE 路徑
```

C51 設定（toy env 對應）：
- `n_atoms = 51`
- `v_min = -1.0, v_max = 1.0`（試過 ±2，效果差，改成更貼近實際 return range 的 ±1）
- target net 強制 `eval()` 移除 noise（沿用 NoisyNet 那邊的修法）
- PER priority 訊號 = cross-entropy loss per sample（取代 |TD-error|）

### Phase 4：C51 結果不如預期 → 跑完整 ablation

5/6 with C51（V=±1）跑出來: **0.313 win rate** — 比 4/6 Rainbow-Lite 的 0.653 還差！

進一步跑 6/6 全 Rainbow（C51 + NoisyNet）：**0.133 win rate** — 最差！

完整 ablation table（5 個配置）：

| 配置 | Rainbow 元件 | episodes / 25k steps | last-300 win rate |
|---|---:|---:|---:|
| HW3-3 · Dueling Double DQN（baseline）| 2/6 | 1074 | 0.643 |
| **Rainbow 4/6**（+ PER + n=3）✨ best | 4/6 | **1224 (+14%)** | **0.653** |
| Rainbow 5/6（+ C51）| 5/6 | 950 | 0.313 |
| Rainbow 5/6（+ NoisyNet）| 5/6 | 812 | 0.367 |
| Rainbow 6/6（full）| 6/6 | 706 | 0.133 |

**最終定調**：實作了完整 6/6 Rainbow，但 ablation 顯示 4/6 才是甜蜜點。
這與 Rainbow paper 本身的觀察一致 — 在較簡單環境上，distributional 與 NoisyNet 的好處比較有限。

### Phase 5：失敗原因分析（寫進 README）

**C51 為什麼變差？**
1. **Reward 退化**：reward ∈ {−1, 0, +1}、γ=0.9、n=3，return 只在 [−1, 1]。51 個 atom 中只有 ~5–10 個有非零機率，其他 40+ 個是浪費的參數
2. **Cross-entropy gradient scale 不同**：CE 的梯度尺度與 lr cosine schedule 不太搭
3. **PER priority 變奇怪**：CE 訊號數值範圍與訓練階段強相關，讓 PER 在中段過度集中

**NoisyNet 為什麼變差？**
1. **狀態空間太小 → 共享權重 noise 過度相關**：4×4 random gridworld 的有效 feature 模式有限，NoisyLinear 對整個 batch 共用權重擾動 → 所有狀態同方向偏移
2. **σ 學成後回不去**：σ 是 learnable，可能不收斂；ε-greedy 衰減是時間決定論式的
3. **PER 放大這個問題**：被 noise 隨機誤判而吃 pit 的 transition 拿到極高 priority

兩者疊加（6/6）→ 雪上加霜，勝率掉到 0.13。

### Phase 6：rollout GIF 二度踩雷

第一次用 seed=777 → 玩家剛好在 goal 隔壁，1 步結束。掃 seed 1-999，挑選 manhattan(player, goal) ≥ 3 的 → seed=99，7 步抵達 goal。寫死進 `train_rainbow.py`。

### Phase 7：artifacts + README + ai_record 整理

最終 artifacts：

```
artifacts/
├── rainbow_lite_rewards.{npy,png}        # 4/6 — 最佳實測配置
├── rainbow_lite_winrate.png
├── rainbow_c51_rewards.{npy,png}         # 5/6 with C51
├── rainbow_c51_winrate.png
├── rainbow_lite_noisy_rewards.npy        # 5/6 with NoisyNet
├── rainbow_full_rewards.{npy,png}        # 6/6 完整 Rainbow
├── rainbow_full_winrate.png
├── rainbow_ablation_rewards.png          # 5 條曲線 comparison
├── rainbow_ablation_winrate.png
├── rollout_rainbow_lite_random.gif       # 4/6 7-step win @ seed=99
├── rollout_rainbow_c51_random.gif        # 5/6 C51 失敗 rollout
├── rollout_rainbow_full_random.gif       # 6/6 失敗 rollout
└── checkpoints/{rainbow_lite,rainbow_c51,rainbow_full}_random.pt
```

README §五 重寫成：
- 「六個改進都實作了」表（標明每個改進的程式碼位置）
- 設定對照表（HW3-3 vs HW3-4 主要差異）
- Ablation 主結果（五行配置 + 5-curve 對照圖）
- 「最佳配置（4/6）的展示」單獨一段（學習曲線 + rollout GIF）
- 「為什麼 C51 在這個 env 變差」+「為什麼 NoisyNet 在這個 env 變差」各一段
- Short understanding report（PER / n-step / IS-weight / C51 projection / 「全部都疊不一定最好」）

### 過程中的小插曲

| # | 事件 | 處理 |
|---|---|---|
| 1 | Lightning CPU 模式 `devices=None` 報錯（HW3-3 舊 bug 沿襲） | 改 `devices=1`（CPU/GPU 通用） |
| 2 | 第一版 Rainbow（含 NoisyNet）勝率從 0.46 跌到 0.21 | 三輪修正後仍輸 baseline → 改設計成「ε-greedy 主路線、NoisyNet 走 ablation」 |
| 3 | C51 加入後勝率 0.31，比 Rainbow-Lite 還差 | 接受這個 ablation 結果，誠實寫進 README，把 4/6 作為展示配置 |
| 4 | rollout seed=777 玩家剛在 goal 隔壁 → 1 步結束 | 改 seed=99，掃過 manhattan 距離 ≥ 3 的局面 |
| 5 | 三條曲線對照圖原本截短到 NoisyNet 的 episode 數 → 不公平 | 改成不截短，各畫各的真實長度 |
| 6 | ai_record 變雙 session 過長 | 拆成 `airecords/airecord1.md` + `airecord2.md` 兩檔 |

### Rainbow DQN 名義 vs 實質的最後決定

使用者選 B 想要「Rainbow DQN」名義，但 ablation 證實 C51 與 NoisyNet 在此 toy env 是負貢獻。
最終定位：

- **專案名稱保留為「Rainbow DQN」**：因為程式碼裡 6 個組件都有實作（`CategoricalRainbowDQN` + `NoisyLinear` 均存在）
- **README 主標題：「HW3-4 ─ Rainbow DQN」**：名實相符
- **Headline rollout：4/6 配置**（empirically best），但 README 清楚標示其他配置在 ablation 中的表現
- **ablation 結論直接寫進 README**：「在 4×4 toy random gridworld 上，只有 PER + n-step 真的有效」，呼應老師「random 能解就好」的提示

這比強行用 5/6 當 headline（明知它更差）誠實，也比叫「Rainbow-Lite」（聽起來不夠完整）更有份量。

## 交付檢核（HW3-4）

- [x] `src/hw3/prioritized_replay.py`（PER + sum-tree）
- [x] `src/hw3/nstep.py`（n-step 累積器）
- [x] `src/hw3/models.py` 新增 `NoisyLinear`、`RainbowDQN`、`CategoricalRainbowDQN`、`project_distribution`
- [x] `src/hw3/train_rainbow.py`（Lightning + `--no-distributional` / `--noisy` flag）
- [x] 五種配置完整 ablation（HW3-3 baseline、4/6、5/6 C51、5/6 NoisyNet、6/6）
- [x] README.md §五 完整重寫（6/6 實作 + ablation 表 + 失敗原因分析 + Short understanding report）
- [x] artifacts：5 條學習曲線 / 勝率對照圖、最佳配置的 rollout GIF、4 個 checkpoint
- [x] last-300 win rate 0.653 > baseline 0.643（同時 +14% 回合數 → 策略更有效率）
- [x] ai_record 拆成 airecord1 + airecord2 分檔
- [x] Commit 仍為 `oomao <csm088220@gmail.com>`，**無 Claude trailer**
