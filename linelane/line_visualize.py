import os
import numpy as np
import cv2

# ===========================
# 설정
# ===========================
# DATASET_DIR = 'lane_dataset'  # 이미지 및 라벨이 들어있는 상위 폴더
DATASET_DIR = 'drivable_dataset'  # 이미지 및 라벨이 들어있는 상위 폴더
IMG_DIR = os.path.join(DATASET_DIR, 'images')
LBL_DIR = os.path.join(DATASET_DIR, 'labels')
SAVE_DIR = os.path.join(DATASET_DIR, 'vis')  # 시각화 결과 저장 경로
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# ===========================
# 시각화 루프
# ===========================
file_list = sorted(os.listdir(LBL_DIR))

for label_file in file_list:
    label_path = os.path.join(LBL_DIR, label_file)
    img_file = label_file.replace('.txt', '.jpg')
    img_path = os.path.join(IMG_DIR, img_file)

    if not os.path.exists(img_path):
        print(f"❌ 이미지 누락: {img_path}")
        continue

    # 이미지 로드
    image = cv2.imread(img_path)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # polygon이 아님

        cls = int(parts[0])  # class id (0)
        coords = list(map(float, parts[1:]))

        # (x, y)로 좌표 쌍 묶기
        pts = [(int(x * IMAGE_WIDTH), int(y * IMAGE_HEIGHT)) for x, y in zip(coords[::2], coords[1::2])]

        if len(pts) >= 3:
            cv2.polylines(image, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # 저장
    out_path = os.path.join(SAVE_DIR, img_file)
    cv2.imwrite(out_path, image)
    print(f"✅ 시각화 저장됨: {out_path}")
