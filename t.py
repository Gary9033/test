def draw_selected_landmarks(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    selected_pixels = []

    for idx, pose_landmarks in enumerate(pose_landmarks_list):
        h, w, _ = annotated_image.shape
        pixel_coords = []

        # 遍歷每個關節
        for i, landmark in enumerate(pose_landmarks):
            name = POSE_LANDMARK_NAMES[i]
            if name in TARGET_LANDMARKS:
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                pixel_coords.append((name, x_px, y_px))
                # 畫節點
                cv2.circle(annotated_image, (x_px, y_px), 5, (0,255,255), -1)

        selected_pixels.append(pixel_coords)

        # 畫連線（可選：只連肩膀到手、肩膀到臀部）
        connections = [
            # ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            # ("LEFT_SHOULDER", "LEFT_ELBOW"),
            # ("LEFT_ELBOW", "LEFT_WRIST"),
            # ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            # ("RIGHT_ELBOW", "RIGHT_WRIST"),
            # ("LEFT_SHOULDER", "LEFT_HIP"),
            # ("RIGHT_SHOULDER", "RIGHT_HIP"),
            # ("LEFT_HIP", "RIGHT_HIP"),
            ("LEFT_INDEX", "RIGHT_INDEX") 
        ]
        for start_name, end_name in connections:
            start = next(((x,y) for n,x,y in pixel_coords if n==start_name), None)
            end = next(((x,y) for n,x,y in pixel_coords if n==end_name), None)
            if start and end:
                cv2.line(annotated_image, start, end, (255,255,0), 2)

    return annotated_image, selected_pixels
