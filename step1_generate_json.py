import mediapipe as mp
import cv2
import json
import os
from pathlib import Path
# Hàm trích xuất keypoints từ video rồi đưa vào json
def process_video(video_path, save_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    frame_index = 1
    pose_keypoints = []
    save_data_json = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image with Mediapipe
            results = pose.process(image)

            # Draw the keypoints on the image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i <= 24:
                        pose_keypoints.append(landmark.x)
                        pose_keypoints.append(landmark.y)
            video_path_split = video_path.split("\\")
            save_data = {
                "uid": video_path,
                "label": video_path_split[-2],
                "frame_index": frame_index,
                "pose": pose_keypoints,
            }
            save_data_json.append(save_data)
            pose_keypoints = []
            frame_index +=  1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = Path(save_dir) / f"{video_path_split[-1].split('.')[0]}.json"
    with open(save_dir, "w") as f:
            json.dump(save_data_json, f)


# Hàm liệt kê các video ở trong train, valid, test ở trong file txt
def save_json(video_path, save_data_path, map_path, mode = ""):
    file_path = Path(map_path) / f'include_{mode}.txt'
    with open(file_path, 'r') as file:
        file_contents = file.read()
    file_contents = file_contents.split("\n")
    save_data_path = save_data_path + "_" + mode
    print(len(file_contents))
    for file in file_contents:
        file_name = Path(video_path) / f"{file}"
        print(file)
        if not file_name.exists():
            print("The path does not exist.")
        # print(file_name)
        # process_video(str(file_name), save_data_path)

if __name__ == "__main__":
    video_path = "Video"
    save_data_path = "save_data"
    map_path = "map"
    #tạo file json theo từng tập train, val, test
    save_json(video_path, save_data_path, map_path, mode = "train")
    # save_json(video_path, save_data_path, map_path, mode = "val")
    # save_json(video_path, save_data_path, map_path, mode = "test")