
def mediapipe_detect(file_path):
    """MediaPipe 姿勢偵測（保持圖片直立方向）"""
    
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        print("請從以下網址下載: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
        return None, None
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到圖片檔案: {file_path}")
        return None, None
    
    img_cv = cv2.imread(file_path)
    if img_cv is None:
        print(f"❌ 無法讀取圖片: {file_path}")
        return None, None
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    print(f"📐 圖片尺寸: 高 {img_rgb.shape[0]} x 寬 {img_rgb.shape[1]}")
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    detection_result = detector.detect(mp_image)
    img_array = img_rgb

    height_pixels = 0
    if detection_result.segmentation_masks:
        mask = detection_result.segmentation_masks[0].numpy_view()

        if mask.ndim == 3 and mask.shape[2] == 1:
            mask_2d = mask[:, :, 0]
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        y_indices, x_indices = np.where(mask_2d > 0.5)
        if len(y_indices) > 0:
            top_y = np.min(y_indices)
            bottom_y = np.max(y_indices)
            height_pixels = bottom_y - top_y
            print(f"頭到腳的 pixel 高度: {height_pixels} 像素")
        else:
            top_y = bottom_y = 0
            print("未偵測到人體區域，請確認影像中有人物。")

        color_mask = np.zeros_like(img_array, dtype=np.uint8)
        color_mask[mask_2d > 0.5] = (0, 0, 255)
        alpha = 0.5
        overlay = cv2.addWeighted(img_array, 1.0, color_mask, alpha, 0)

        h, w, _ = overlay.shape
        center_x = w // 2
        cv2.circle(overlay, (center_x, top_y), 8, (0, 255, 255), -1)
        cv2.circle(overlay, (center_x, bottom_y), 8, (0, 255, 255), -1)

        text1 = f"Height: {height_pixels}px"
        cv2.putText(overlay, text1, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        out_path = os.path.splitext(file_path)[0] + "_overlay.png"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"✅ 覆蓋圖已儲存: {out_path}")

    annotated_image, all_landmark_pixels = draw_selected_landmarks(img_array, detection_result)
    
    landmarks_path = os.path.splitext(file_path)[0] + "_landmarks.png"
    cv2.imwrite(landmarks_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"✅ 關節點標記圖已儲存: {landmarks_path}")

    left_index = None
    right_index = None
    
    if all_landmark_pixels and len(all_landmark_pixels) > 0:
        for name, x, y in all_landmark_pixels[0]:
            if name == "LEFT_INDEX":
                left_index = (x, y)
            elif name == "RIGHT_INDEX":
                right_index = (x, y)

    hand_distance = 0
    if left_index and right_index:
        lx, ly = left_index
        rx, ry = right_index
        hand_distance = round(np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2), 2)
        print(f"左右食指的距離: {hand_distance} 像素")
    else:
        print("⚠️ 未偵測到左右食指")

    return hand_distance, height_pixels
