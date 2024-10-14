import pandas as pd
import os
import shutil
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import numpy as np
from tqdm import tqdm

# Excel 파일 읽기
excel_file = r"C:\Users\SNUH\OneDrive\SNUH BMI Lab\RCDM_AI\echo\labeling_template_1013Song.xlsx"
df = pd.read_excel(excel_file)

# 각 view별 데이터 개수 계산
view_counts = {
    '4CV': df['4CV'].notna().sum(),
    'PLAX': df['PLAX'].notna().sum(),
    'PSAX(PM)': df['PSAX(PM)'].notna().sum()
}

print("각 view별 데이터 개수:")
for view, count in view_counts.items():
    print(f"{view}: {count}")

# 각 view에 해당하는 파일명 추출
view_files = {
    '4CV': df[df['4CV'].notna()]['FileName'].tolist(),
    'PLAX': df[df['PLAX'].notna()]['FileName'].tolist(),
    'PSAX(PM)': df[df['PSAX(PM)'].notna()]['FileName'].tolist()
}

def convert_dicom_to_jpg(dicom_file, output_folder):
    # DICOM 파일 읽기
    dicom = pydicom.dcmread(dicom_file)
    
    # 픽셀 데이터 추출
    if 'PixelData' not in dicom:
        raise ValueError(f"DICOM file {dicom_file} does not contain pixel data")
    
    # 픽셀 배열 가져오기
    pixel_array = dicom.pixel_array

    # 데이터 형태 처리
    if pixel_array.ndim == 4:
        if pixel_array.shape[0] == 1 and pixel_array.shape[1] == 1:
            pixel_array = pixel_array[0, 0]  # (1, 1, height, width) 또는 (1, 1, height, 3) 형태 처리
    
    # 여러 프레임 처리
    if pixel_array.ndim == 4:  # (frames, height, width, channels)
        for i in range(pixel_array.shape[0]):
            frame = pixel_array[i]
            if frame.shape[2] == 3:  # RGB 이미지
                image = Image.fromarray(frame.astype('uint8'))
            else:  # 흑백 이미지
                frame = apply_voi_lut(frame, dicom)
                frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                image = Image.fromarray(frame)
            
            image_path = os.path.join(output_folder, f"{i+1}.jpg")
            image.save(image_path)
    else:  # 단일 프레임
        if pixel_array.ndim == 3 and pixel_array.shape[2] == 3:  # RGB 이미지
            image = Image.fromarray(pixel_array.astype('uint8'))
        else:  # 흑백 이미지
            data = apply_voi_lut(pixel_array, dicom)
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            image = Image.fromarray(data)
        
        image_path = os.path.join(output_folder, "image.jpg")
        image.save(image_path)

def process_dicom_files(input_folder, output_base_folder, view_files):
    for view, files in view_files.items():
        view_folder = os.path.join(output_base_folder, view)
        if not os.path.exists(view_folder):
            os.makedirs(view_folder)
        
        for filename in tqdm(files, desc=f"Processing {view} files"):
            dicom_file = os.path.join(input_folder, f"{filename}.dcm")
            if not os.path.exists(dicom_file):
                tqdm.write(f"File not found: {dicom_file}")
                continue
            
            output_folder = os.path.join(view_folder, filename)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            try:
                convert_dicom_to_jpg(dicom_file, output_folder)
                tqdm.write(f"Converted {filename} successfully")
            except Exception as e:
                tqdm.write(f"Error converting {filename}: {str(e)}")

# 실행
input_folder = r"C:\Users\SNUH\Desktop\echo\sample_10000"
output_base_folder = r"C:\Users\SNUH\Desktop\echo\sample_10000_img"

process_dicom_files(input_folder, output_base_folder, view_files)

print("처리가 완료되었습니다.")