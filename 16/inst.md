## 1）把「核心想法挖深」：用一個句子統一全篇

你現在的內容其實已經有很好的骨架：你把問題拆成 data/model/deployment 的生命週期，並討論 explainability、robustness、fairness、evaluation。 
但 reviewer 會覺得「太泛」的原因通常是：**缺少一個可操作的框架把這些關聯“鎖”起來**。

### 你可以把核心主張改成這種「可落地」版本（放在 Introduction 結尾 + Abstract + Conclusion 呼應）

> *透明與可信不是一組方法，而是一條「證據鏈（evidence chain）」：從資料可追溯 → 模型可解釋/可檢驗 → 對 domain shift 與偏差有壓力測試 → 以臨床可用的指標與治理流程形成閉環。*

然後你在文中做兩件事，就會“挖深”：

1. **每個維度都回答同一個問題**：它在 evidence chain 上提供什麼證據？證據如何失效？如何評估？
2. **把 Table 1/2 的資訊升級成「決策指南」**：不是列框架，而是告訴讀者「何時用哪種評估、缺口在哪」。你文末其實也寫到基準分裂、缺乏統一標準，但要更像「結論」。

---

# (Added) Implementation deliverables (English)

This folder now includes a runnable, paper-oriented sanity study implementation.

## What you can run

- Script: `experiment_sanity_medmnist.py`
- Dependencies: `requirements.txt`
- Paper-ready text: `paper_ready_sanity_section.md`

## What it produces

For each run, outputs are written to `16/outputs/sanity_<timestamp>/`:

- `tables/per_seed_metrics.csv`: metrics per dataset per seed
- `tables/agg_mean_std.csv`: mean ± std across seeds (per dataset)
- `tables/paired_ttests_gradcam_vs_ig.csv`: paired t-tests (Grad-CAM vs IG)
- `tables/deletion_curves_raw.json`: raw deletion curves for regeneration
- `plots/*.png`: deletion curves, stability boxplots, robustness curves, reliability diagrams, qualitative examples
- `repro/`: config + environment + package list (+ optional git info)

## One-command run

```bash
pip install -r 16/requirements.txt
python 16/experiment_sanity_medmnist.py --datasets all --seeds 0,1,2,3,4 --pretrained
```


## 3）補「最簡單實驗」：做一個 3 小時能跑完的 sanity study

你要的不是 SOTA，而是**用最小成本證明你的論點**：

> 「單靠 post-hoc saliency 不足以提供可靠證據；需要同時檢查 faithfulness、stability、robustness/calibration。」

這正好呼應你 paper 已經在講的結論：post-hoc 不夠、要 lifecycle + evaluation + governance。

### 最小可行實驗設計（只做一個任務）

**Task**：二分類或多分類（任意一個公開小 dataset）

* 優先：MedMNIST（輕量、下載快、跑得動）
* 模型：ResNet{18,34,50}（timm 直接用）
* 解釋法：Grad-CAM + Integrated Gradients（兩個就夠）
* 你要報告的指標只要 4 個（全部都能不用 ROI 標註就算）：

1. **Faithfulness（刪除測試 Deletion AUC）**：把 saliency top-k 區域遮掉，觀察信心/正確率下降曲線
2. **Stability（擾動一致性）**：對同一張圖加很小的高斯噪聲/亮度變化，算 heatmap 的 SSIM 或 rank correlation
3. **Robustness（簡單 corruption）**：Gaussian noise / blur 下 accuracy drop
4. **Calibration（ECE）**：報 ECE 或 reliability diagram（可選，但很加分）

你最後放：

* **一張圖**：原圖 + 兩種 heatmap + deletion curve（或 stability boxplot）
* **一個小表**：Acc / Deletion-AUC / Stability-SSIM / Acc@Noise

這會直接回應 reviewer：「太概念」「沒有實驗/圖」以及「evaluation framework 不清楚」——因為你用實驗示範**“怎麼評估”**。

### 最短可以貼進 paper 的實驗段落（英文，直接用）

你可以新增一節：*“Lightweight Empirical Sanity Check: Evaluating Saliency Evidence”*
（下面這段可直接貼、再把 dataset 名稱補上）

> We conduct a lightweight sanity study to illustrate why post-hoc explanations alone are insufficient as evidence for clinical trust. Using a standard CNN classifier (ResNet-18) on a publicly available medical imaging benchmark, we compare Grad-CAM and Integrated Gradients under four criteria: (i) **faithfulness** measured by deletion-based perturbation curves (AUC), (ii) **stability** measured by similarity of attribution maps under small input perturbations, (iii) **robustness** measured by performance under simple corruptions (noise/blur), and (iv) **calibration** measured by expected calibration error (ECE). The results show that attribution maps can be visually plausible yet unstable, and faithfulness scores vary substantially across methods, motivating the need for standardized, multi-dimensional evaluation protocols aligned with deployment risks.

---