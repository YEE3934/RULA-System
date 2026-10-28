# RULA v10｜Hand 優先／Pose 2D 備援

以 v9.1 原始完整程式為基礎。設定流程由六步縮成五步，移除「Hand 實驗設定」頁。手腕正式計分優先使用幾何品質通過的 Hand 2D；Hand 無法匹配或品質不合格時使用 Pose 2D；兩者都無效則略過樣本。Pose 3D 與舊版 0.6×Pose2D+0.4×Pose3D 僅在研究欄位保留，絕不用於正式手腕分數。

## Hand 幾何品質
- 偵測到目標手且 21 個點有效；Hand 腕點須對應目標 Pose 腕點，排除靠近對側手的匹配。
- 手掌像素長度、掌長與前臂比例、腕點位置偏差等門檻。
- Hand 與 Pose 2D 差異 **不再是拒絕 Hand 的條件**，只在資料中留下警示。
- 為避免資料被靜默改寫，時序孤立峰值保留作為診斷警示；此版本尚未使用未來影格做事後回算 RULA，品質過關但偶發尖峰仍可能影響分數，需用實際影片驗證。
- 單一鏡頭 2D 手腕角仍是投影代理值，非量角器真值；Hand 優先是根據使用者目前人工比對觀察採取的工作假設。

## 部署
將本資料夾 222.py、requirements.txt、packages.txt 放到 GitHub 儲存庫根目錄，保留 models/hand_landmarker.task（若已有）；Streamlit Cloud 使用 Python 3.11。requirements.txt、packages.txt 沿用 v9 系列已成功安裝的設定。

## 測試限制
已通過語法與靜態規則檢查，未在本環境以真實影片／MediaPipe 執行完整 Cloud 端到端測試。請先用同組影片比較 Hand 來源佔比、Pose 備援樣本、RULA 分數和極端影格。
