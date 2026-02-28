
def detect_and_crop_both_feet(file_path, padding_ratio=None, save_output=True):
    """
    偵測雙腳位置並裁切（padding 根據腳距佔圖片寬度的比例自動計算）
    
    Args:
        file_path: 圖片路徑
        padding_ratio: 如果為 None，則自動使用 feet_width/image_width 作為 padding_ratio
                      也可手動指定比例（例如 0.1 表示 padding = 圖片寬度的 10%）
        save_output: 是否儲存結果
    """
    
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return None
    
    img_cv = cv2.imread(file_path)
    if img_cv is None:
        print(f"❌ 無法讀取圖片: {file_path}")
        return None
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    detection_result = detector.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        print("❌ 未偵測到人體姿勢！")
        return None
    
    pose_landmarks = detection_result.pose_landmarks[0]
    
    foot_landmarks = [
        pose_landmarks[27],  # LEFT_ANKLE
        pose_landmarks[28],  # RIGHT_ANKLE
        pose_landmarks[29],  # LEFT_HEEL
        pose_landmarks[30],  # RIGHT_HEEL
        pose_landmarks[31],  # LEFT_FOOT_INDEX
        pose_landmarks[32],  # RIGHT_FOOT_INDEX
    ]
    
    foot_points = [(int(lm.x * w), int(lm.y * h)) for lm in foot_landmarks]
    
    x_coords = [pt[0] for pt in foot_points]
    y_coords = [pt[1] for pt in foot_points]
    
    feet_min_x = min(x_coords)
    feet_max_x = max(x_coords)
    feet_width = feet_max_x - feet_min_x
    
    feet_width_ratio = feet_width / w
    
    if padding_ratio is None:
        padding_ratio = feet_width_ratio + 0.03  # 預設在腳距比例基礎上增加 5% 的 padding
    
    padding = int(w * padding_ratio)
    
    print(f"📏 左右腳距離: {feet_width} 像素")
    print(f"📐 圖片寬度: {w} 像素")
    print(f"📊 腳距佔圖片寬度比: {feet_width_ratio*100:.2f}%")
    print(f"🔧 padding_ratio: {padding_ratio*100:.2f}%")
    print(f"✂️ 計算出的 padding: {padding} 像素 (圖片寬度的 {padding_ratio*100:.2f}%)")
    
    min_x = max(0, min(x_coords) - padding)
    max_x = min(w, max(x_coords) + padding)
    min_y = max(0, min(y_coords) - padding)
    max_y = min(h, max(y_coords) + padding)
    
    left_ankle_px = foot_points[0]
    right_ankle_px = foot_points[1]
    left_heel_px = foot_points[2]
    right_heel_px = foot_points[3]
    left_foot_px = foot_points[4]
    right_foot_px = foot_points[5]
    
    print(f"📦 雙腳裁切區域: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    print(f"📐 裁切尺寸: {max_x - min_x} x {max_y - min_y} 像素")
    
    img_bgr = img_cv.copy()
    
    cv2.circle(img_bgr, left_foot_px, 5, (255, 0, 255), -1)
    cv2.circle(img_bgr, right_foot_px, 5, (255, 0, 255), -1)
    cv2.circle(img_bgr, left_heel_px, 5, (0, 255, 255), -1)
    cv2.circle(img_bgr, right_heel_px, 5, (0, 255, 255), -1)
    cv2.circle(img_bgr, left_ankle_px, 5, (0, 255, 0), -1)
    cv2.circle(img_bgr, right_ankle_px, 5, (0, 255, 0), -1)
    
    feet_crop = img_bgr[min_y:max_y, min_x:max_x]
    
    if save_output:
        crop_path = "both_feet_crop.png"
        cv2.imwrite(crop_path, feet_crop)
        print(f"✂️ 雙腳裁切圖已儲存: {crop_path}")
    
    result = {
        'left_foot_toe': left_foot_px,
        'left_foot_heel': left_heel_px,
        'left_ankle': left_ankle_px,
        'right_foot_toe': right_foot_px,
        'right_foot_heel': right_heel_px,
        'right_ankle': right_ankle_px,
        'crop_region': (min_x, min_y, max_x, max_y),
        'crop_size': (max_x - min_x, max_y - min_y),
        'feet_crop': feet_crop,
        'feet_width': feet_width,
        'feet_width_ratio': feet_width_ratio,
        'padding_used': padding
    }
    
    return result
