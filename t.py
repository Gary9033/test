def draw_selected_landmarks(rgb_image, detection_result):
    """繪製選定的關節點"""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    selected_pixels = []

    for idx, pose_landmarks in enumerate(pose_landmarks_list):
        h, w, _ = annotated_image.shape
        pixel_coords = []

        for i, landmark in enumerate(pose_landmarks):
            name = POSE_LANDMARK_NAMES[i]
            if name in TARGET_LANDMARKS:
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                pixel_coords.append((name, x_px, y_px))
                cv2.circle(annotated_image, (x_px, y_px), 5, (0, 255, 255), -1)

        selected_pixels.append(pixel_coords)

        connections = [
            ("LEFT_INDEX", "RIGHT_INDEX")
        ]
        for start_name, end_name in connections:
            start = next(((x, y) for n, x, y in pixel_coords if n == start_name), None)
            end = next(((x, y) for n, x, y in pixel_coords if n == end_name), None)
            if start and end:
                cv2.line(annotated_image, start, end, (255, 255, 0), 2)

    return annotated_image, selected_pixels
