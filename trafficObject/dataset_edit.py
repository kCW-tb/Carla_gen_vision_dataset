import os
import cv2
import logging

# 종료 : q
# 이미지 다음 : d, 이전 : a
# BBox 다음 : s, 이전 : w
# BBox 삭제 : n, 이미지, 라벨 삭제 : m

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMAGE_DIR = 'yolo_dataset/images'
LABEL_DIR = 'yolo_dataset/labels'
IMAGE_EXT = '.jpg'
LABEL_EXT = '.txt'
WINDOW_NAME = 'BBox Editor'

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(IMAGE_EXT)])
current_image_idx = 0
current_bbox_idx = 0

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"이미지 로드 실패: {image_path}")
        return None
    return img

def load_labels(label_path):
    if not os.path.exists(label_path):
        logging.warning(f"레이블 파일이 존재하지 않습니다: {label_path}")
        return []
    try:
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        logging.error(f"레이블 파일 로드 실패: {label_path}, {e}")
        return []

def save_labels(label_path, labels):
    try:
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels) + '\n')
        logging.info(f"레이블 저장 완료: {label_path}")
    except Exception as e:
        logging.error(f"레이블 저장 실패: {label_path}, {e}")

def draw_bbox(image, label_line, bbox_index, total_bboxes):
    h, w = image.shape[:2]
    parts = label_line.split()
    if len(parts) != 5:
        logging.warning(f"잘못된 레이블 형식: {label_line}")
        return image
    try:
        cls, cx, cy, bw, bh = map(float, parts)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            logging.warning(f"유효하지 않은 BBox 좌표: {label_line} (이미지 크기: {w}x{h})")
            return image
        img = image.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"BBox {bbox_index + 1}/{total_bboxes}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img
    except ValueError as e:
        logging.warning(f"레이블 파싱 오류: {label_line}, {e}")
        return image

def delete_image_and_label(image_path, label_path, image_name):
    """
    이미지와 레이블 파일을 데이터셋에서 삭제
    """
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"이미지 삭제 완료: {image_path}")
        else:
            logging.warning(f"이미지 파일이 존재하지 않습니다: {image_path}")
        if os.path.exists(label_path):
            os.remove(label_path)
            logging.info(f"레이블 삭제 완료: {label_path}")
        else:
            logging.warning(f"레이블 파일이 존재하지 않습니다: {label_path}")
        if image_name in image_files:
            image_files.remove(image_name)
            logging.info(f"이미지 목록에서 제거: {image_name}")
        return True
    except Exception as e:
        logging.error(f"파일 삭제 실패: {image_path}, {label_path}, {e}")
        return False

while True:
    if not image_files:
        logging.info("데이터셋에 이미지가 없습니다. 프로그램 종료.")
        break

    if current_image_idx < 0:
        current_image_idx = 0
    elif current_image_idx >= len(image_files):
        current_image_idx = len(image_files) - 1

    image_name = image_files[current_image_idx]
    label_name = image_name.replace(IMAGE_EXT, LABEL_EXT)

    img_path = os.path.join(IMAGE_DIR, image_name)
    label_path = os.path.join(LABEL_DIR, label_name)

    image = load_image(img_path)
    if image is None:
        logging.warning(f"이미지 로드 실패로 다음 이미지로 이동: {img_path}")
        image_files.pop(current_image_idx)
        if current_image_idx >= len(image_files):
            current_image_idx = len(image_files) - 1
        continue

    labels = load_labels(label_path)

    if current_bbox_idx >= len(labels):
        current_bbox_idx = len(labels) - 1
    if current_bbox_idx < 0:
        current_bbox_idx = 0

    display_img = image.copy()
    if labels and current_bbox_idx < len(labels):
        display_img = draw_bbox(display_img, labels[current_bbox_idx], current_bbox_idx, len(labels))

    cv2.imshow(WINDOW_NAME, display_img)
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q') or key == 27:  # 'q' 또는 ESC: 종료
        break
    elif key == ord('d'):  # 다음 이미지
        current_image_idx += 1
        current_bbox_idx = 0
    elif key == ord('a'):  # 이전 이미지
        current_image_idx -= 1
        current_bbox_idx = 0
    elif key == ord('s'):  # 다음 BBox
        current_bbox_idx += 1
        if current_bbox_idx >= len(labels):
            current_bbox_idx = len(labels) - 1
    elif key == ord('w'):  # 이전 BBox
        current_bbox_idx -= 1
        if current_bbox_idx < 0:
            current_bbox_idx = 0
    elif key == ord('n') and labels:  # 현재 BBox 삭제
        logging.info(f"BBox {current_bbox_idx + 1} 삭제: {label_name}")
        labels.pop(current_bbox_idx)
        save_labels(label_path, labels)
        if current_bbox_idx >= len(labels):
            current_bbox_idx = len(labels) - 1
    elif key == ord('m'):  # 이미지와 레이블 삭제
        logging.info(f"데이터셋에서 삭제: {image_name}, {label_name}")
        if delete_image_and_label(img_path, label_path, image_name):
            if image_files:  # 리스트가 비어 있지 않으면 인덱스 조정
                if current_image_idx >= len(image_files):
                    current_image_idx = len(image_files) - 1
            else:
                break  # 이미지 목록이 비면 종료
            current_bbox_idx = 0

cv2.destroyAllWindows()
logging.info("프로그램 종료")