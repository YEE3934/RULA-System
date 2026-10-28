# RULA v11.3｜可選品質提醒雙照片複核（結案前介面修正）

部署版本：`RULA_V11_3_USER_NOTICE_ACTIONS_2026-10-28`

本版以 **v11.2 的完整 `222.py`** 修改，保留原本的五步驟設定／分析後分頁、三視角時間同步、Hand 2D 優先、Pose 2D 備援、骨架遮擋 QC、頸部比較、RULA 逐影格修正及作業基本資料與自動命名。**本次不修改角度計算公式或 RULA Table A/B/C。**

## 更新功能

分析完成後，進入「🧩 剔除樣本」分頁。頁面有「需要處理／一般品質提醒／全部」篩選，選擇每筆紀錄後，可切換三個鏡頭，並排看**未渲染原圖**與**骨架／可疑關節照片**。

每筆資料可以做以下選擇：

- **維持／確認使用自動評分**：保留原自動分數，計入複核後統計。一般品質提醒預設採用這一項，不必逐筆處理。
- **改用人工重新評分**：在同一分頁輸入上臂、前臂、手腕、腕扭轉、頸部、軀幹、腿部、肌肉、負重九項 RULA 分數，勾選已人工核對並儲存後才納入可信統計；未完成時保持待處理。
- **剔除此側樣本**：從複核後可信統計排除，但保留原始資料、原圖和候選角度。

**辨識失敗而沒有原始自動分數的影格，不能選擇「確認採用原自動評分」**，可以手動輸入 RULA 各項分數或剔除。左右側分開操作：若只有其中一側可信，該側仍保留，但不會產生誤導性的「完整雙側最高分」。

## 匯出

下載 ZIP 內仍包含全部原始分析、原始影格、骨架影格、品質提醒、失敗診斷；另外新增：

- `Final_Reviewed/Side_Reviewed_Results.csv`：每筆側別的原始分數、複核後分數、狀態與是否可納入統計。
- `Final_Reviewed/Combined_Reviewed_Results.csv`：完整雙側有效的最高分；只有一側有效則明確標為資料不完整。
- `Final_Reviewed/Review_Decisions.csv`：每筆品質紀錄的使用者決定及原因。
- `Final_Reviewed/Manual_RULA_Scores.csv`：人工重新評分所填的各項分數與計算結果。
- `Final_Reviewed/README.txt`：欄位語意與注意事項。
- `Diagnostics/Rejected_Manual_Reviews.csv`：沿用相容檔名，含使用者最新決定。

原始 RULA CSV **從不被覆寫**，提供稽核及對照。ZIP 命名仍根據第一頁的日期、單位、員工及作業名稱。

## 部署

將本資料夾內 `222.py` 更新至 GitHub 專案根目錄。`requirements.txt`、`packages.txt` 及 `scripts/` 使用已測試的版本即可，不必新裝套件。Streamlit Cloud 使用 **Python 3.11**。若已放入 `models/hand_landmarker.task` 請保留；部署包不含模型二進位檔。

測試指令：

```bash
python -m py_compile 222.py
python -m unittest discover -s tests -p 'test*.py' -v
```

已完成離線單元測試、使用此前分析 ZIP 的實際資料回放與 ZIP 完整性測試；尚未在使用者的 Streamlit Cloud 進行真實影片全流程驗收。

**學術注意：**品質檢查與人工確認流程不能證明實際關節角度精度；RULA Table A/B/C 尚需獨立核對標準工作表。請在結案報告中區分「辨識有效比例」與「角度準確率」。
