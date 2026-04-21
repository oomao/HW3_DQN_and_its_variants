# AI 對話紀錄 — DRL HW3 DQN and its variants

**日期:** 2026-04-21
**Model:** Claude Opus 4.7 (1M context) via Claude Code
**專案:** https://github.com/oomao/HW3_DQN_and_its_variants
**Live demo:** https://oomao.github.io/HW3_DQN_and_its_variants/

---

## 使用者初始需求

延續 HW2 的工作風格，本次作業要求：
- 以 *Deep Reinforcement Learning in Action* GitHub repo 的 Chapter 3 starter code 為基底
- HW3-1 (30%)：Naive DQN + Experience Replay（static 模式）
- HW3-2 (40%)：Double DQN 與 Dueling DQN（player 模式）
- HW3-3 (30%)：PyTorch Lightning 改寫 + 訓練技巧（random 模式）
- 上傳至 `https://github.com/oomao/HW3_DQN_and_its_variants.git`，commit 作者 `oomao <csm088220@gmail.com>`，無 Claude trailer
- 所有文件使用繁體中文
- Live demo 也要
- 可用 RTX 4090 CUDA 加速 HW3-3

## 關鍵決策

| 問題 | 使用者選擇 |
|---|---|
| HW3-3 框架 | PyTorch Lightning |
| 其他流程 | 照舊（openspec + GitHub Pages + 中文 README） |
| HW3-3 硬體 | CUDA（RTX 4090） |
| 文件語言 | 繁體中文 |

## 執行步驟總結

### Phase 0：探索
- Clone 教科書 repo 到 `_upstream/`（納入 `.gitignore`，不上 git）
- 使用 Explore subagent 分析 Chapter 3 / Environments
  - `Gridworld(size=4, mode={'static','player','random'})`
  - 狀態 `board.render_np()` → `(4,4,4)` → reshape `(1,64)` + 雜訊
  - 動作 `{0:u,1:d,2:l,3:r}`，reward ±1/0
  - Listing 3.2 網路：`Linear(64→150→100→4)`
  - Listing 3.7 有 target network 但無 Dueling；Lightning 從零實作

### Phase 1：openspec
- `openspec init --tools claude`
- 三個 change：`01-naive-dqn-static`、`02-double-dueling-player`、`03-lightning-random-tips`
- 每個 change 含 `proposal.md`、`tasks.md`、`specs/dqn/spec.md`
- `openspec validate --strict` 全部通過後 archive

### Phase 2：實作
```
src/hw3/
├── gridworld_vendored.py   # 教科書 Gridworld/GridBoard（MIT，附 attribution header）
├── env.py                  # reset/step 包裝、狀態前處理
├── models.py               # DQN / DuelingDQN / soft_update / hard_update
├── replay.py               # ReplayBuffer (deque + torch sample)
├── viz.py                  # 學習曲線 + 勝率 + rollout GIF（CJK 字型修復）
├── train_naive.py          # HW3-1：有/無 replay 對比
├── train_variants.py       # HW3-2：Vanilla / Double / Dueling 三路對比
└── train_lightning.py      # HW3-3：LightningModule，支援 --accelerator gpu
```

### Phase 3：訓練結果

| 題目 | 模式 | 硬體 | 規模 | 最終指標 |
|---|---|---|---|---|
| HW3-1 | static | CPU | 3000 episodes | 最後 200 回合勝率 ≈ **99.5%**（w/ replay） |
| HW3-2 | player | CPU | 4000 episodes × 3 variants | 三者皆 **100%** 勝率 |
| HW3-3 | random | **CUDA (RTX 4090)** | 25000 steps | 最後 300 回合勝率 ≈ **64.3%** |

HW3-3 特別記事：
- 最初 12000 步 CPU 版本因 Lightning `IterableDataset` 在 buffer 未 warmup 時
  一直 yield `None` 而造成 dataloader 空轉，~30 分鐘仍未完成。
- 改成：先在 `Trainer.fit()` 之前 `prefill()` 預熱 500 transition；
  `IterableDataset` 每次都 yield 真 batch；移除 `None` 的情況。
- 改寫後 6000 步 CPU = 18 秒（完全被 Lightning 包裝的 overhead 主導，每步 ~3ms）
- 應使用者要求切到 CUDA + 25000 步 = 3 分鐘，勝率衝到 64%。

### Phase 4：視覺化
- `artifacts/*.png`：學習曲線 + 勝率（用 Microsoft JhengHei 支援中文）
- `artifacts/*.gif`：每題最後一支「學成後的 greedy rollout」
- Pit 原本用 Unicode MINUS 字符顯示，字型缺字 → 改為 "X"

### Phase 5：Live demo + README
- `docs/index.html`：深色主題中文頁面，分 HW3-1 / HW3-2 / HW3-3 三個區塊
- `docs/.nojekyll`：避免 GitHub Pages 自動渲染 README
- README.md：環境說明 + 三題各自的參數表、結果、分析、Short understanding report

### Phase 6：Git + Push
- 設定 `user.email=csm088220@gmail.com`、`user.name=oomao`
- 初次 push 被 reject（GitHub 自動建了 README placeholder）
- `git pull --allow-unrelated-histories` → 手動解衝突（保留中文 README） → push 成功
- 最終兩個 commit 皆為 `oomao <csm088220@gmail.com>`，**無 Claude trailer**

## 過程中的小插曲

| # | 事件 | 處理 |
|---|---|---|
| 1 | Lightning CPU 訓練 30+ 分鐘未結束 | 砍掉重寫：預熱 buffer + 修掉 dataloader None yield；變成 18 秒 |
| 2 | matplotlib 中文顯示成豆腐塊 | `rcParams["font.sans-serif"]` 改成 `Microsoft JhengHei` fallback |
| 3 | Pit 的 `−` 字符顯示成豆腐塊 | 改成 `X` |
| 4 | `git push` 被 remote 擋 | `pull --allow-unrelated-histories` → 解衝突 → 重 push |
| 5 | Rollout GIF 隨機種子踩空 | 掃 seed 0-99 找出第一個贏且步數 ≥ 3 的關卡（seed=3，6 步） |

## 交付檢核

- [x] openspec `01-naive-dqn-static`（archived）
- [x] openspec `02-double-dueling-player`（archived）
- [x] openspec `03-lightning-random-tips`（archived）
- [x] HW3-1：static mode 勝率 99.5%，rollout GIF 成功 7 步抵達
- [x] HW3-2：player mode 三個變體全 100%
- [x] HW3-3：PyTorch Lightning + 5 項訓練 tips + CUDA + 64.3% 勝率（random 本身上限約 70%）
- [x] 中文 README（含 Short understanding report）
- [x] GitHub Pages 中文 live demo
- [x] Commit 僅 `oomao <csm088220@gmail.com>`，無 Claude trailer
- [x] 推送到 `https://github.com/oomao/HW3_DQN_and_its_variants.git`
- [x] `.claude/`、`_upstream/` 不在 GitHub 上
- [x] 老師原始素材收納在 `reference/`（本作業無提供原始圖片，但資料夾已預留）

## 使用者後續需要做的

1. 到 https://github.com/oomao/HW3_DQN_and_its_variants/settings/pages 啟用 GitHub Pages：
   - Source: `Deploy from a branch`
   - Branch: `main`、資料夾選 `/docs`
2. 等 1-2 分鐘後 `https://oomao.github.io/HW3_DQN_and_its_variants/` 就會上線
