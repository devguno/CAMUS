import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import numpy as np
from tqdm import tqdm

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

def process_dicom_files(input_folder, output_base_folder):
    dicom_files = [f for f in os.listdir(input_folder) if f.endswith('.dcm')]
    
    for filename in tqdm(dicom_files, desc="Processing DICOM files"):
        dicom_file = os.path.join(input_folder, filename)
        output_folder = os.path.join(output_base_folder, os.path.splitext(filename)[0])
        
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

process_dicom_files(input_folder, output_base_folder)