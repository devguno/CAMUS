import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from tqdm import tqdm

# 데이터 경로 설정
data_dir = r"C:\Users\SNUH\Desktop\echo\sample_10000_img"
dicom_test_dir = r"C:\Users\SNUH\Desktop\echo\dicom_test_data"
processed_test_dir = r"C:\Users\SNUH\Desktop\echo\processed_test_data"

# 뷰 클래스 정의
views = ['4CV', 'PLAX', 'PSAX(PM)']

# 이미지 크기 및 배치 크기 설정
img_height, img_width = 224, 224
batch_size = 32

# DICOM to JPG 변환 함수
def convert_dicom_to_jpg(dicom_file, output_folder):
    dicom = pydicom.dcmread(dicom_file)
    
    if 'PixelData' not in dicom:
        raise ValueError(f"DICOM file {dicom_file} does not contain pixel data")
    
    pixel_array = dicom.pixel_array

    if pixel_array.ndim == 4:
        if pixel_array.shape[0] == 1 and pixel_array.shape[1] == 1:
            pixel_array = pixel_array[0, 0]
    
    if pixel_array.ndim == 4:  # Multiple frames
        for i in range(pixel_array.shape[0]):
            frame = pixel_array[i]
            if frame.shape[2] == 3:  # RGB image
                image = Image.fromarray(frame.astype('uint8'))
            else:  # Grayscale image
                frame = apply_voi_lut(frame, dicom)
                frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                image = Image.fromarray(frame)
            
            image_path = os.path.join(output_folder, f"{i+1}.jpg")
            image.save(image_path)
    else:  # Single frame
        if pixel_array.ndim == 3 and pixel_array.shape[2] == 3:  # RGB image
            image = Image.fromarray(pixel_array.astype('uint8'))
        else:  # Grayscale image
            data = apply_voi_lut(pixel_array, dicom)
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            image = Image.fromarray(data)
        
        image_path = os.path.join(output_folder, "image.jpg")
        image.save(image_path)

# DICOM 테스트 데이터 처리 함수
def process_dicom_test_data(dicom_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in tqdm(os.listdir(dicom_dir), desc="Processing DICOM files"):
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(dicom_dir, filename)
            output_folder = os.path.join(output_dir, os.path.splitext(filename)[0])
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            try:
                convert_dicom_to_jpg(dicom_file, output_folder)
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

# 데이터 증강 및 전처리
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 훈련 데이터 생성기
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=views
)

# 검증 데이터 생성기
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=views
)

# 모델 구축
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(views), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 모델 학습 함수
def train_model(model, epochs=15):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    return history

# 학습 결과 시각화 함수
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

# 모델 평가 함수 (DICOM 테스트 데이터용)
def evaluate_model_dicom(model, dicom_test_dir, processed_test_dir):
    # DICOM 파일을 JPG로 변환
    process_dicom_test_data(dicom_test_dir, processed_test_dir)

    # 변환된 이미지로 테스트 데이터 생성기 생성
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        processed_test_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    # 예측
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # 결과 출력
    for filename, predicted_class in zip(test_generator.filenames, predicted_classes):
        print(f"File: {filename}, Predicted class: {views[predicted_class]}")

    return predictions, predicted_classes

# 메인 실행 부분
if __name__ == "__main__":
    # 모델 생성 및 학습
    model = build_model()
    history = train_model(model)

    # 학습 결과 시각화
    plot_training_history(history)

    # 모델 저장
    model.save('echo_view_classification_model.h5')

    # DICOM 테스트 데이터에 대한 모델 평가
    predictions, predicted_classes = evaluate_model_dicom(model, dicom_test_dir, processed_test_dir)

    print("\nEvaluation complete. You can now analyze the predictions and predicted classes.")
    
    # 추가 분석을 위한 예시 코드 (필요에 따라 주석 해제 및 수정)
    # 예측 확률 분포 시각화
    # plt.figure(figsize=(10, 6))
    # plt.hist(np.max(predictions, axis=1), bins=20)
    # plt.title('Distribution of Maximum Prediction Probabilities')
    # plt.xlabel('Probability')
    # plt.ylabel('Frequency')
    # plt.show()

    # 특정 임계값을 적용한 분류 결과
    # threshold = 0.8
    # high_confidence_predictions = predictions[np.max(predictions, axis=1) > threshold]
    # high_confidence_classes = np.argmax(high_confidence_predictions, axis=1)
    # for filename, predicted_class in zip(test_generator.filenames, high_confidence_classes):
    #     print(f"High confidence prediction - File: {filename}, Predicted class: {views[predicted_class]}")