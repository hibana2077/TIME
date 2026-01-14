## 小而完整的 POC / toy experiment 設計（一天內可跑完）

### 核心問題（你要在 paper 裡回答）

> 「當模型學到 spurious shortcut 時，常見 XAI 熱圖會不會看起來很合理但其實不 faithful？我們能不能用 deletion metric 把這件事量化？」

### Dataset：自己合成，避免下載大型醫療資料（但仍“像”醫療）

生成 64×64 灰階圖：

* 背景：高斯雜訊（模擬 imaging noise）
* **真因果特徵（lesion）**：一個亮圓點/亮斑（有 ground-truth mask）
* **spurious 特徵（hospital logo/watermark）**：右下角小方塊，訓練時與 label 高度相關（例如 95%）

建立三個 split：

1. Train(ID-spurious)：logo 幾乎決定 label
2. Test(ID)：同分佈（logo 還在）
3. Test(OOD)：logo 打散/移除（讓捷徑失效）

### Model：兩個版本（快速對照）

* CNN baseline（3–4 層 conv）
* 同一 CNN 但訓練時做「logo randomization」或「cutout 抹掉角落」當作 mitigation（讓模型回到 lesion）

### Explainability：至少一個即可（建議 Grad-CAM）

* 對每張圖產生 saliency map（Grad-CAM / IG 擇一即可，Grad-CAM最快）

### Metrics（重點：要“量化”）

1. **Accuracy drop（ID→OOD）**：證明模型真的依賴 spurious shortcut
2. **Faithfulness（Deletion curve / AUC）**：

   * 刪掉 top-k% 最顯著像素（依 saliency 排序）
   * 觀察預測信心下降幅度
   * 再加一個 random deletion baseline 對照（避免騙自己）
3. **Localization IoU（saliency vs lesion mask）**：

   * 把 saliency threshold 成 binary mask
   * 與真 lesion mask 算 IoU
   * 直觀呈現「熱圖到底有沒有在看 lesion」

### 你最後要放進 paper 的輸出（建議最小集）

* Figure：同一張圖（原圖、Grad-CAM、lesion mask）各放 2–3 個例子：

  * spurious model：熱圖聚焦在 logo（IoU 低）
  * mitigated model：熱圖回到 lesion（IoU 高）
* 一張 deletion curve（或用 deletion AUC 的 bar chart）
* 一張小表：Accuracy(ID/OOD)、IoU、Deletion AUC

> 這一段會直接把「evaluation frameworks unclear」變成「我們給出一個可操作的最小評估 protocol」，也補你說的深度。

---

## 10–12 小時：把 POC 結果「嵌回去」解決 chairs 的批評

你不用大改全文，只要新增一節（例如 6.3 或 7 前）：

**“A Minimal, Reproducible Protocol: Faithfulness under Shortcut Learning”**

* 先用 1 段把 protocol 寫成 step-by-step（像 checklist）
* 接著放圖表＋2–3 段討論：

  * 視覺上 plausible ≠ faithful
  * deletion/Iou 能把差異量化
  * 連回你 survey 的主張（呼應你在 Table 1 對 faithfulness 指標的整理）

最後補強 conclusion：用 6–8 句把「你做了什麼整理＋你主張什麼評估方式＋臨床落地意味什麼」寫完整（你現在結論太短）。

---

如果你願意，我也可以直接幫你把「POC 那一節」寫成可貼進論文的段落結構（含小節標題、圖表 caption 模板、以及 deletion metric 的數學定義寫法），你只要把你用的框架（PyTorch/TensorFlow）跟你想用的 explanation 方法（Grad-CAM/IG/SHAP）回我一個就行。
