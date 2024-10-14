import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# 데이터 경로 설정
input_dir = r"C:\Users\SNUH\Desktop\echo\sample_10000_mp4"
output_dir = r"C:\Users\SNUH\Desktop\echo\sample_10000_gif"
views = ['4CV', 'PLAX', 'PSAX(PM)']

def convert_mp4_to_gif(input_file, output_file, fps=10):
    try:
        clip = VideoFileClip(input_file)
        clip.write_gif(output_file, fps=fps)
        clip.close()
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")

def process_view_folders(input_dir, output_dir, views):
    for view in views:
        view_input_dir = os.path.join(input_dir, view)
        view_output_dir = os.path.join(output_dir, view)
        
        if not os.path.exists(view_output_dir):
            os.makedirs(view_output_dir)

        mp4_files = [f for f in os.listdir(view_input_dir) if f.endswith('.mp4')]
        
        for mp4_file in tqdm(mp4_files, desc=f"Converting {view} MP4s to GIFs"):
            input_file = os.path.join(view_input_dir, mp4_file)
            output_file = os.path.join(view_output_dir, mp4_file.replace('.mp4', '.gif'))
            convert_mp4_to_gif(input_file, output_file)

if __name__ == "__main__":
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 각 view에 대해 처리
    process_view_folders(input_dir, output_dir, views)

    print("All MP4 files have been converted to GIFs successfully.")