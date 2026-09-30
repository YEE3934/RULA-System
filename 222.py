import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import os
import zipfile
import io
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ==================== 【正統 RULA 查表矩陣】 ====================
TABLE_A = {
    1: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:2, 2:3}, 4: {1:3, 2:3}},
         2: {1: {1:2, 2:2}, 2: {1:2, 2:3}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         3: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}} },
    2: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         3: {1: {1:3, 2:4}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}} },
    3: { 1: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         2: {1: {1:3, 2:3}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}},
         3: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}} },
    4: { 1: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}},
         2: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         3: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}} },
    5: { 1: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         2: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         3: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}} },
    6: { 1: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         2: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}},
         3: {1: {1:6, 2:7}, 2: {1:7, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}} }
}

TABLE_B = {
    1: {1: {1:1, 2:3}, 2: {1:2, 2:3}, 3: {1:3, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:6}, 6: {1:7, 2:7}},
    2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:4, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:7}, 6: {1:7, 2:8}},
    3: {1: {1:3, 2:3}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:6, 2:6}, 5: {1:7, 2:7}, 6: {1:7, 2:8}},
    4: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    5: {1: {1:6, 2:6}, 2: {1:6, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    6: {1: {1:7, 2:7}, 2: {1:7, 2:8}, 3: {1:8, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}}
}

TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 5, 6], [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7], [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7]
]

# ==================== 【核心判定與幾何計算】 ====================
def get_upper_arm_score(angle): return 1 if angle <= 20 else (2 if angle <= 45 else (3 if angle <= 90 else 4))
def get_lower_arm_score(angle): return 1 if 60 <= angle <= 100 else 2
def get_neck_score(angle): return 1 if angle <= 10 else (2 if angle <= 20 else 3)
def get_trunk_score(angle): return 1 if angle <= 5 else (2 if angle <= 20 else (3 if angle <= 60 else 4))

def compute_rula_full_score(upper_arm, lower_arm, neck, trunk, muscle_val, force_val):
    ua, la, w, wt = min(max(upper_arm, 1), 6), min(max(lower_arm, 1), 3), 1, 1
    nk, tk, lg = min(max(neck, 1), 6), min(max(trunk, 1), 6), 1
    score_a, score_b = TABLE_A[ua][la][w][wt], TABLE_B[nk][tk][lg]
    score_c, score_d = score_a + muscle_val + force_val, score_b + muscle_val + force_val
    grand_score = TABLE_C[min(score_c, 8) - 1][min(score_d, 7) - 1]
    al_num = 1 if grand_score <= 2 else (2 if grand_score <= 4 else (3 if grand_score <= 6 else 4))
    return score_a, score_b, score_c, score_d, grand_score, f"AL{al_num}"

def calculate_3d_angle(a, b, c):
    ba, bc = np.array(a) - np.array(b), np.array(c) - np.array(b)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)))

def put_chinese_text(img, text, position, color, size=15):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("msjh.ttc", size)
    except: font = ImageFont.load_default()
    draw.text(position, text, fill=(color[2], color[1], color[0]), font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 【單張影像完整 AI 分析模組 (含人因零度校正 & 幾何防禦)】 ====================
def analyze_image(frame, pose_model, m_val, f_val):
    orig_h, orig_w = frame.shape[:2]
    scale = min(800 / orig_w, 600 / orig_h)
    frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if not results.pose_world_landmarks:
        return False, None, image, 0.0
        
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    w_lm = results.pose_world_landmarks.landmark
    n_lm = results.pose_landmarks.landmark
    
    try:
        # 計算原始各部位信賴度
        conf_neck = round((n_lm[7].visibility + n_lm[8].visibility + n_lm[11].visibility + n_lm[12].visibility) / 4 * 100, 1)
        conf_r_arm = round((n_lm[24].visibility + n_lm[12].visibility + n_lm[14].visibility + n_lm[16].visibility) / 4 * 100, 1)
        conf_l_arm = round((n_lm[23].visibility + n_lm[11].visibility + n_lm[13].visibility + n_lm[15].visibility) / 4 * 100, 1)
        overall_conf = (conf_neck + conf_r_arm + conf_l_arm) / 3 

        # 幾何合理性審查 (防禦正面彎腰透視變形)
        shoulder_width_2d = abs(n_lm[11].x - n_lm[12].x)
        trunk_length_2d = math.hypot(((n_lm[11].x + n_lm[12].x) / 2) - ((n_lm[23].x + n_lm[24].x) / 2),
                                     ((n_lm[11].y + n_lm[12].y) / 2) - ((n_lm[23].y + n_lm[24].y) / 2))
        if trunk_length_2d > 0 and shoulder_width_2d > trunk_length_2d * 0.8: 
            overall_conf = overall_conf * 0.5 

        # 抓取 3D 座標點
        ls, rs = [w_lm[11].x, w_lm[11].y, w_lm[11].z], [w_lm[12].x, w_lm[12].y, w_lm[12].z]
        le, re = [w_lm[13].x, w_lm[13].y, w_lm[13].z], [w_lm[14].x, w_lm[14].y, w_lm[14].z]
        lw, rw = [w_lm[15].x, w_lm[15].y, w_lm[15].z], [w_lm[16].x, w_lm[16].y, w_lm[16].z]
        lh, rh = [w_lm[23].x, w_lm[23].y, w_lm[23].z], [w_lm[24].x, w_lm[24].y, w_lm[24].z]
        le_ear, ri_ear = [w_lm[7].x, w_lm[7].y, w_lm[7].z], [w_lm[8].x, w_lm[8].y, w_lm[8].z]

        # =================【RULA 人因工程標準絕對角度校正】=================
        # 1. 下臂彎曲 (Elbow): 180 - 數學內角 (完全打直=0度)
        ang_r_elb = abs(180 - calculate_3d_angle(rs, re, rw))
        ang_l_elb = abs(180 - calculate_3d_angle(ls, le, lw))

        # 2. 上臂屈曲 (Upper Arm): 相對於絕對垂直線 (Y軸向下 +1.0)
        r_sh_down = [rs[0], rs[1] + 1.0, rs[2]]
        l_sh_down = [ls[0], ls[1] + 1.0, ls[2]]
        ang_r_sh = calculate_3d_angle(r_sh_down, rs, re)
        ang_l_sh = calculate_3d_angle(l_sh_down, ls, le)

        # 3. 軀幹彎曲 (Trunk): 相對於骨盆絕對垂直線 (Y軸向上 -1.0)
        mid_sh = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
        mid_hip = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
        hip_up = [mid_hip[0], mid_hip[1] - 1.0, mid_hip[2]] 
        ang_trunk = calculate_3d_angle(hip_up, mid_hip, mid_sh)

        # 4. 頸部彎曲 (Neck): 相對於軀幹軸線
        mid_ear = [(le_ear[0]+ri_ear[0])/2, (le_ear[1]+ri_ear[1])/2, (le_ear[2]+ri_ear[2])/2]
        ang_neck = abs(180 - calculate_3d_angle(mid_hip, mid_sh, mid_ear))
        # ===============================================================

        s_nk, s_tk = get_neck_score(ang_neck), get_trunk_score(ang_trunk)
        r_ua, r_la = get_upper_arm_score(ang_r_sh), get_lower_arm_score(ang_r_elb)
        l_ua, l_la = get_upper_arm_score(ang_l_sh), get_lower_arm_score(ang_l_elb)

        r_a, r_b, r_c, r_d, r_grand, r_al = compute_rula_full_score(r_ua, r_la, s_nk, s_tk, m_val, f_val)
        l_a, l_b, l_c, l_d, l_grand, l_al = compute_rula_full_score(l_ua, l_la, s_nk, s_tk, m_val, f_val)
        
        worst_side = "右手側" if r_grand >= l_grand else "左手側"
        final_grand = max(r_grand, l_grand)
        final_al = r_al if r_grand >= l_grand else l_al

        data = {
            "頸部角度": round(ang_neck, 1), "軀幹角度": round(ang_trunk, 1),
            "右肩角度": round(ang_r_sh, 1), "右手肘角": round(ang_r_elb, 1),
            "左肩角度": round(ang_l_sh, 1), "左手肘角": round(ang_l_elb, 1),
            "最危害側": worst_side,
            "Score A": r_a if worst_side=="右手側" else l_a,
            "Score B": r_b if worst_side=="右手側" else l_b,
            "Score C": r_c if worst_side=="右手側" else l_c,
            "Score D": r_d if worst_side=="右手側" else l_d,
            "Grand Score": final_grand,
            "AL": final_al,
            "Overall_Conf": overall_conf
        }
        return True, data, image, overall_conf
    except Exception as e:
        return False, None, image, 0.0  

# ==================== 【Tkinter 彈出式設定視窗】 ====================
class MultiCamSettingsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RULA 系統")
        self.root.geometry("500x450")
        
        self.vid1, self.vid2 = "", ""
        self.num_samples = tk.IntVar(value=50)
        self.is_start = False
        
        tk.Label(root, text="第一步：選擇拍攝視角影片", font=("微軟正黑體", 12, "bold")).pack(pady=(10,5))
        
        frame_cams = tk.Frame(root)
        frame_cams.pack()
        
        self.btn_v1 = tk.Button(frame_cams, text=" 選擇 [正面鏡頭] 影片", command=lambda: self.select_file(1), font=("微軟正黑體", 10))
        self.btn_v1.grid(row=0, column=0, padx=10, pady=5)
        self.lbl_v1 = tk.Label(frame_cams, text="未選擇", fg="blue", width=20, anchor="w")
        self.lbl_v1.grid(row=0, column=1)
        
        self.btn_v2 = tk.Button(frame_cams, text=" 選擇 [側面鏡頭] 影片", command=lambda: self.select_file(2), font=("微軟正黑體", 10))
        self.btn_v2.grid(row=1, column=0, padx=10, pady=5)
        self.lbl_v2 = tk.Label(frame_cams, text="未選擇 (可選)", fg="gray", width=20, anchor="w")
        self.lbl_v2.grid(row=1, column=1)
        
        tk.Label(root, text="第二步：等距抽樣與參數設定", font=("微軟正黑體", 12, "bold")).pack(pady=(15,5))
        
        frame_params = tk.Frame(root)
        frame_params.pack()
        tk.Label(frame_params, text="抽樣照片張數:").grid(row=0, column=0, sticky="e", pady=5)
        tk.Spinbox(frame_params, from_=5, to=200, textvariable=self.num_samples, width=10).grid(row=0, column=1, pady=5)
        
        tk.Label(frame_params, text="肌肉狀態 (Muscle):").grid(row=1, column=0, sticky="e", pady=5)
        self.combo_muscle = ttk.Combobox(frame_params, values=["0: 無維持超過1分鐘", "1: 姿勢維持>1分鐘/高頻率"], state="readonly", width=25)
        self.combo_muscle.current(0)
        self.combo_muscle.grid(row=1, column=1, pady=5)
        
        tk.Label(frame_params, text="荷重施力 (Force):").grid(row=2, column=0, sticky="e", pady=5)
        self.combo_force = ttk.Combobox(frame_params, values=["0: < 2kg (間歇)", "1: 2-10kg (間歇)", "2: 2-10kg (靜態/重複)", "2: 負載 > 10kg (間歇性)", "3: > 10kg"], state="readonly", width=25)
        self.combo_force.current(0)
        self.combo_force.grid(row=2, column=1, pady=5)
        
        self.btn_start = tk.Button(root, text="啟動分析", command=self.start_analysis, font=("微軟正黑體", 12, "bold"), bg="#4CAF50", fg="white", height=2)
        self.btn_start.pack(pady=20, fill="x", padx=20)
        
    def select_file(self, cam_id):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi")])
        if path:
            if cam_id == 1:
                self.vid1 = path
                self.lbl_v1.config(text=os.path.basename(path), fg="blue")
            else:
                self.vid2 = path
                self.lbl_v2.config(text=os.path.basename(path), fg="blue")
            
    def start_analysis(self):
        if not self.vid1:
            messagebox.showwarning("請至少選擇第一支影片！")
            return
        self.m_val = int(self.combo_muscle.get().split(":")[0])
        self.f_val = int(self.combo_force.get().split(":")[0])
        self.is_start = True
        self.root.destroy()

root = tk.Tk()
app = MultiCamSettingsApp(root)
root.mainloop()

if not app.is_start:
    print("使用者取消操作，程式結束。")
    exit()

# ==================== 【主程式：等距截圖與 Sensor Fusion】 ====================
print(f"\n 準備進行分析...")
cap1 = cv2.VideoCapture(app.vid1)
fps = cap1.get(cv2.CAP_PROP_FPS)
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

use_dual_cam = bool(app.vid2)
if use_dual_cam:
    cap2 = cv2.VideoCapture(app.vid2)
    total_frames = min(total_frames, int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

# 使用 numpy linspace 確保時間完美等分，包含首尾
target_samples = app.num_samples.get()
sample_indices = np.linspace(0, total_frames - 1, target_samples, dtype=int).tolist()

print(f"\n 正在擷取抽樣照片")
raw_frames_1, raw_frames_2 = [], []
for idx in sample_indices:
    current_time_sec = round(idx / fps, 2)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
    succ1, f1 = cap1.read()
    if succ1: raw_frames_1.append((current_time_sec, f1))
    
    if use_dual_cam:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        succ2, f2 = cap2.read()
        if succ2: raw_frames_2.append((current_time_sec, f2))

cap1.release()
if use_dual_cam: cap2.release()
print(f"照片擷取完成\n")

print(f"進行多視角信賴度分析")
records = []
zip_images = []

with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.7) as pose:
    for i in range(len(raw_frames_1)):
        sec, frame1 = raw_frames_1[i]
        
        succ_1, data_1, img_1, conf_1 = analyze_image(frame1, pose, app.m_val, app.f_val)
        best_data, best_img, best_cam_label, best_conf = data_1, img_1, "正面/鏡頭 A", conf_1
        
        if use_dual_cam and i < len(raw_frames_2):
            _, frame2 = raw_frames_2[i]
            succ_2, data_2, img_2, conf_2 = analyze_image(frame2, pose, app.m_val, app.f_val)
            
            # 若正面鏡頭發生透視壓縮，其 conf_1 會被打對折，讓側面 conf_2 勝出
            if succ_2 and conf_2 > conf_1:
                best_data, best_img, best_cam_label, best_conf = data_2, img_2, "側面/鏡頭 B", conf_2
        
        if best_data is not None:
            final_record = {"樣本編號": i + 1, "時間(秒)": sec, "最佳視角來源": best_cam_label, "最佳綜合信賴度(%)": round(best_conf, 1)}
            final_record.update(best_data)
            final_record.pop('Overall_Conf', None)
            records.append(final_record)
            
            # 畫上採用結果
            best_img = put_chinese_text(best_img, f"時間: {sec}秒 | {best_data['最危害側']}高風險", (15, 15), (0, 255, 255), 18)
            best_img = put_chinese_text(best_img, f"總分: {best_data['Grand Score']} 分 | {best_data['AL']}", (15, 45), (0, 0, 255) if best_data['Grand Score'] > 4 else (0,255,0), 16)
            best_img = put_chinese_text(best_img, f"🏆 採用畫面: {best_cam_label} (信賴度: {best_conf:.1f}%)", (15, 75), (255, 150, 0), 14)
            
            is_success, buffer = cv2.imencode(".jpg", best_img)
            if is_success:
                zip_images.append((f"Sample_{i+1:03d}_{sec}s_BestCam.jpg", buffer.tobytes()))
                
        print(f"進度: {i+1}/{len(raw_frames_1)} 筆比對完成...", end="\r")

print(f"\n✅ 分析完成！\n")

# ==================== 【主程式：階段 3 (建立 ZIP 壓縮檔匯出)】 ====================
print(f"打包照片庫與數據報表成 ZIP")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"MultiCam_RULA_Report_{current_time}.zip"

with zipfile.ZipFile(zip_filename, 'w') as zf:
    if records:
        df = pd.DataFrame(records)
        csv_bytes = df.to_csv(index=False).encode('utf-8-sig')
        zf.writestr("RULA統計報表.csv", csv_bytes)
    
    for filename, img_bytes in zip_images:
        zf.writestr(f"照片庫/{filename}", img_bytes)

print(f" 成果已成功產生")
print(f" 在左側檔案總管查看並解壓縮這個檔案：【 {zip_filename} 】\n")
