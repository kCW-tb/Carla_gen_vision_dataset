import os
import cv2
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 클래스 ID와 이름 매핑
CLASS_IDS = {
    0: 'vehicle',
    1: 'pedestrian'
}

# 데이터셋 디렉토리 설정
dataset_dir = "./yolo_dataset"
image_dir = os.path.join(dataset_dir, "images")
label_dir = os.path.join(dataset_dir, "labels")

def load_yolo_labels(label_path, img_width, img_height):
    """
    YOLO 레이블 파일을 읽어 바운딩 박스 정보를 반환
    """
    annotations = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                logging.warning(f"잘못된 레이블 형식: {line.strip()} in {label_path}")
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                # 픽셀 좌표로 변환
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                annotations.append({
                    'class_id': class_id,
                    'bbox': [x_min, y_min, x_max, y_max]
                })
            except ValueError as e:
                logging.warning(f"레이블 파싱 오류: {line.strip()} in {label_path}, {e}")
                continue
        logging.info(f"YOLO 레이블 로드 완료: {label_path}")
        return annotations
    except Exception as e:
        logging.error(f"레이블 파일 로드 실패: {label_path}, {e}")
        return []

def draw_bounding_boxes(image, annotations):
    """
    이미지에 바운딩 박스를 그림
    """
    for ann in annotations:
        class_id = ann['class_id']
        x_min, y_min, x_max, y_max = [int(v) for v in ann['bbox']]
        # 유효성 검사
        if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0]:
            logging.warning(f"유효하지 않은 바운딩 박스 좌표: {ann['bbox']} (이미지 크기: {image.shape[1]}x{image.shape[0]})")
            continue
        class_name = CLASS_IDS.get(class_id, 'unknown')
        # 바운딩 박스 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image

def visualize_yolo_dataset():
    """
    YOLO 데이터셋을 시각화
    """
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        logging.error(f"이미지 또는 레이블 디렉토리가 존재하지 않습니다: {image_dir}, {label_dir}")
        raise FileNotFoundError(f"이미지 또는 레이블 디렉토리가 존재하지 않습니다")

    # 이미지 파일 목록 가져오기
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    for image_filename in image_files:
        image_path = os.path.join(image_dir, image_filename)
        label_filename = image_filename.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_filename)

        # 이미지 로드
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"이미지 로드 실패: {image_path}")
                continue
        except Exception as e:
            logging.error(f"이미지 로드 실패: {image_path}, {e}")
            continue

        img_height, img_width = image.shape[:2]

        # 레이블 로드
        if not os.path.exists(label_path):
            logging.warning(f"레이블 파일이 존재하지 않습니다: {label_path}")
            image_with_bboxes = image.copy()
        else:
            annotations = load_yolo_labels(label_path, img_width, img_height)
            image_with_bboxes = draw_bounding_boxes(image.copy(), annotations)

        # 이미지 표시
        cv2.imshow("YOLO Dataset Visualization", image_with_bboxes)
        logging.info(f"이미지 표시: {image_filename}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):  # 'n' 키: 다음 이미지
                break
            elif key == ord('q'):  # 'q' 키: 종료
                cv2.destroyAllWindows()
                logging.info("시각화 종료")
                return

    cv2.destroyAllWindows()
    logging.info("모든 이미지를 표시 완료")

if __name__ == "__main__":
    try:
        visualize_yolo_dataset()
    except Exception as e:
        logging.error(f"시각화 중 오류 발생: {e}")
        cv2.destroyAllWindows()