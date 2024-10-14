import os
import cv2
import numpy as np
from tqdm import tqdm

# 데이터 경로 설정
data_dir = r"C:\Users\SNUH\Desktop\echo\sample_10000_img"
output_dir = r"C:\Users\SNUH\Desktop\echo\sample_10000_mp4"
views = ['4CV', 'PLAX', 'PSAX(PM)']

def create_video_from_images(image_folder, output_file, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # 파일 이름으로 정렬

    if not images:
        print(f"No images found in {image_folder}")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in tqdm(images, desc=f"Creating video for {os.path.basename(image_folder)}"):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def process_view_folders(data_dir, output_dir, views):
    for view in views:
        view_input_dir = os.path.join(data_dir, view)
        view_output_dir = os.path.join(output_dir, view)
        
        if not os.path.exists(view_output_dir):
            os.makedirs(view_output_dir)

        for folder in tqdm(os.listdir(view_input_dir), desc=f"Processing {view} folders"):
            input_folder = os.path.join(view_input_dir, folder)
            output_file = os.path.join(view_output_dir, f"{folder}.mp4")

            if os.path.isdir(input_folder):
                create_video_from_images(input_folder, output_file)

if __name__ == "__main__":
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 각 view에 대해 처리
    process_view_folders(data_dir, output_dir, views)

    print("All videos have been created successfully.")