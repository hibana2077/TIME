## 實驗目標（用一句話對審稿人說清楚）

在不同 DP 預算 ((\epsilon,\delta)) 下，量化 **data attribution 的 (1) 穩定性 stability、(2) 可信度 credibility** 如何隨 (\epsilon) 變化，並和 **utility** 一起畫出 trade-off 曲線/曲面，最後把三者彙整成可計算的 **Trustworthiness Index**。

---

## 實驗要怎麼做（Step-by-step，直接可寫進論文）

### 1) 任務、資料、模型（先選「簡單但標準」）

* **資料集（建議 1 個就好，先把 story 做完整）**

  * CIFAR-10（或 MNIST / Adult-Income 也可，但 CIFAR-10 比較像「真實深度學習」）
* **模型**

  * 小型 CNN / ResNet-18（別太大，因為你要重複訓練很多次）
* **訓練設定固定**（學習率、batch size、epoch、optimizer）

### 2) DP 訓練：DP-SGD + budget sweep

* 用 DP-SGD（per-sample gradient clipping + Gaussian noise）
* 固定 (\delta)（例如 (10^{-5})），掃一組 (\epsilon)：

  * (\epsilon \in {\infty, 10, 5, 2, 1, 0.5, 0.2})（(\infty)=non-private baseline）
* 每個 (\epsilon) **至少跑 5 個 random seeds**（這是你 stability 指標的根基）

> 你稿子自己就說要看 stability：跨 seed / resampling / repeated runs 的一致性。

### 3) Data Attribution 方法（挑一個能落地、算得動的）

選「**training-data attribution**」而不是 feature attribution，符合你文意（data attribution）。
建議二選一：

* **TracIn**（實作相對友善：checkpoint 梯度內積累加）
* **Influence Functions (近似版)**（較難、但更經典）

> 只做一種 attribution method 就夠把主張驗證起來；若要加分，補一個「第二種 attribution」當 robustness check。

### 4) 你要量化的指標（把「討論」變成「可算」）

你文中把 attribution trust 拆成 stability + credibility。

#### A. Stability（跨重跑的一致性）

對每個 test query（例如抽 100 個測試樣本），你會得到一個「對所有訓練資料的 attribution 分數向量」。

對同一個 (\epsilon)，跨不同 seeds 的結果做：

* **Spearman rank correlation**（分數排名一致性）
* **Top-k 重疊率**：Jaccard((Top_k))（例如 k=50）
  -（可選）Kendall-(\tau)

最後把 stability 報成：各 (\epsilon) 的平均 ± 95% CI（跨 seeds, queries）。

#### B. Credibility（「像不像真的」）

你文中定義 credibility：和 non-private baseline 一致 + counterfactual influence tests。
所以做兩個量化：

1. **Agreement with non-private baseline**

* 對同一批 queries，算 attribution 向量在 DP vs non-DP 的相關（Spearman / Pearson）
* 這給你「DP 把 attribution 扭曲到什麼程度」

2. **Counterfactual influence test（最重要，審稿人會買單）**

* 對每個 query，取 DP attribution 的 Top-k 訓練點
* 做兩種移除（或下權重）比較：

  * 移除 Top-k
  * 移除 Random-k（重複多次取平均）
* 觀察 **模型輸出改變量**（擇一即可）：

  * 該 query 的 logit / probability 變化
  * 該 query 是否 label flip
  * 或整體測試集 accuracy drop（較粗但好懂）

Credibility 可以用「效果差」表示：
[
\Delta = \text{Effect(remove Top-k)} - \text{Effect(remove Random-k)}
]
(\Delta) 越大代表 attribution 越「真的抓到有影響力的資料」。

#### C. Utility（任務表現）

* Test accuracy / F1（跟你任務一致）

---

## 結果要長怎麼樣（你該交出的圖表長相）

你要讓審稿人一眼看到：**越嚴格的隱私（(\epsilon) 越小），attribution 會先崩，utility 可能晚一點才掉**——這就是你文中 “budget-dependent fragile” 的實證版。

### 必備 4 張圖（最少但完整）

1. **Utility vs (\epsilon)**：accuracy（y）對 (\epsilon)（x，建議 log scale）
2. **Stability vs (\epsilon)**：Spearman / Jaccard@k（含 95% CI）
3. **Credibility vs (\epsilon)**：

   * baseline agreement（相關） vs (\epsilon)
   * counterfactual (\Delta) vs (\epsilon)（最好加顯著性標記）
4. **Trade-off 圖**（二選一）

   * 2D：x=Utility、y=Attribution Trust（可用 stability+credibility 加權平均）
   * 3D / heatmap：((\epsilon), Utility, Attribution Trust) 做 surface（knee point 超好講）

### 必備 2 張表（補足「統計/比較」）

* **Table A：每個 (\epsilon)** 的 Utility / Stability / Credibility（mean±CI）
* **Table B：counterfactual test 的統計檢定**

  * remove Top-k vs random-k 的差異，用 paired t-test 或 Wilcoxon（附 p-value、effect size）

---

## Trustworthiness Index 要怎麼「落地」（把指標真的算出來）

你文中說要把 privacy strength、attribution trust、utility 聚合成單一可稽核分數。
建議用「**加權幾何平均**」：任何一項很爛就會拉低總分（符合直覺、也好 defend）

1. 先把三個量都 normalize 到 ([0,1])：

* (U(\epsilon))：accuracy normalize（相對於 non-DP）
* (T(\epsilon))：attribution trust（例如 (0.5\cdot Stability + 0.5\cdot Credibility)）
* (P(\epsilon))：privacy strength（可用 (P=1/(1+\epsilon)) 再 normalize）

2. 定義：
   [
   \text{TWI}(\epsilon)=\big(P(\epsilon)^{w_p}\cdot T(\epsilon)^{w_t}\cdot U(\epsilon)^{w_u}\big)^{1/(w_p+w_t+w_u)}
   ]
   權重先用均等 (w_p=w_t=w_u=1)，再在附錄做敏感度分析（換權重不改趨勢就很加分）。

---

## 論文要怎麼報告（直接給你章節骨架）

新增一個「Experiments」章節，把你現在被批評的空缺補齊：

### 4 Experiments

**4.1 Setup**

* Dataset / model / training details
* DP accounting（(\epsilon,\delta)、clipping norm、noise multiplier、seeds 數）

**4.2 Attribution Method**

* 選 TracIn/Influence 的定義與實作細節（checkpoint、計算成本）

**4.3 Metrics**

* Stability（Spearman、Jaccard@k）
* Credibility（baseline agreement、counterfactual (\Delta) + 統計檢定）
* Utility（accuracy）

**4.4 Results**

* 4 張圖 + 2 張表
* 明確指出 knee point：例如「(\epsilon\le 1) 時 stability/credibility 急遽崩潰，但 utility 只小幅下降」這類結論（要用圖支撐）

**4.5 Trustworthiness Index instantiation**

* 給 TWI 公式、normalize 方法、曲線（TWI vs (\epsilon)）
* 給「建議 DP budget 選擇」：取 TWI 最大或在某 threshold 上的最小 (\epsilon)

**4.6 Limitations & Open Problems**

* 直接承認「在嚴格 DP budget 下做 attribution 並能公開釋出」仍是 open（你 reviewer 也講這點），但你已經量化了 degrade 曲線，並提供可操作的 reporting protocol。