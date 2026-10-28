# RULA v11.2 — 結案穩定版

版本：`RULA_V11_2_FINAL_REVIEW_CLARITY_2026-10-28`

本版以 v11.1 完整 `222.py` 修改，不是重新製作。保留 Hand 2D 優先、Pose 2D 備援、三鏡頭同步、五步設定／分析後分頁、人工修正、頸部比較、異常及剔除照片、RULA 報表、ZIP 自動命名，以及日期／公司單位／員工／作業名稱欄位。

## 這次修正

1. **不再混淆品質提醒與人工複核**：分析結果的「剔除／骨架品質待複核」只顯示需真正處理的紀錄；可採信的「留意」在可收合區域，無須逐筆勾選。
2. **匯出清單拆開**：`Diagnostics/Rejected_Samples_Detail.csv` 僅為辨識失敗及需人工複核；`Diagnostics/Quality_Notices.csv` 僅為仍可採信的品質提示。完整原始品質 CSV 和原有原圖／骨架照片仍保留，不遺失證據。
3. **同步時間友善化**：`Synchronization/Frame_Time_Map.csv` 及有樣本的計分報表新增 `分析經過時間(秒)`；風險圖以分析開始 0 秒呈現。舊的「共同時間(秒)」仍以同步事件為 0（所以事件前可能是負數），每台相機的實際時間戳也保留。
4. **完全不改 RULA 或姿勢分析核心**：保持 v11.1 的 MediaPipe 初始化、Hand／Pose 幾何與品質門檻、RULA Table A/B/C、人工補正和同步偏移演算法。

## 部署方式

- 將本資料夾內的 `222.py`、`requirements.txt`、`packages.txt` 與必要附屬檔案放在 GitHub 儲存庫**根目錄**；Streamlit 主程式指定 `222.py`。
- Streamlit Community Cloud：Python **3.11**。保留既有 `models/hand_landmarker.task`；部署包不包含 Google 二進位模型，必要時以 `scripts/download_hand_model.py` 下載。
- 更新後確認畫面頂部出現 `RULA_V11_2_FINAL_REVIEW_CLARITY_2026-10-28`。
- 用先前同一組影片測試：三鏡頭預覽、分析、剔除／品質待複核、ZIP 下載與報表欄位。
- `requirements.txt`、`packages.txt` 原則上與 v11.1 相同，不必為這版另加套件。

## 離線測試

```bash
python -m py_compile 222.py
python -m unittest discover -s tests -p 'test*.py' -v
```

截至交付，18 項單元測試和語法檢查通過；但仍**沒有**在你自己的 Streamlit Cloud 上以真實影片完成端到端驗收。

## 最後的學術注意事項

- 品質檢查通過 ≠ 人體解剖角度準確率已驗證。
- Hand 與 Pose 的差異 ≠ 與人工真值之差異。
- RULA Table A/B/C 尚需對照正式工作表逐格驗證，不應在未驗證前聲稱所有評分與標準完全一致。
- 若要結案，請固定一個成功跑完的版本，並備份三視角影片、ZIP 報告、照片與網站截圖。
