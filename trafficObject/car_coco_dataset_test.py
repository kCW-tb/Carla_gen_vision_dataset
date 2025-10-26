import json
import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 생성된 데이터셋 확인하기.
dataset_dir = "./TrafficObject/coco_dataset"

image_dir = os.path.join(dataset_dir, "images")
annotation_file = os.path.join(dataset_dir, "annotations.json")

def load_coco_dataset():
    if not os.path.exists(annotation_file):
        logging.error(f"Annotation file not found: {annotation_file}")
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    try:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        logging.info("COCO 어노테이션 파일 로드 완료")
        return coco_data
    except Exception as e:
        logging.error(f"어노테이션 파일 로드 실패: {e}")
        raise

def draw_bounding_boxes(image, annotations, image_id):
    for ann in annotations:
        if ann["image_id"] == image_id:
            bbox = ann["bbox"]
            x, y, w, h = [int(v) for v in bbox]
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                logging.warning(f"유효하지 않은 바운딩 박스 좌표: {bbox} (이미지 크기: {image.shape[1]}x{image.shape[0]})")
                continue
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(image, "vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image

def visualize_coco_dataset():
    coco_data = load_coco_dataset()
    images = coco_data["images"]
    annotations = coco_data["annotations"]

    for img_info in images:
        image_id = img_info["id"]
        image_filename = img_info["file_name"]
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            logging.warning(f"이미지 파일이 존재하지 않습니다: {image_path}")
            continue
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"이미지 로드 실패: {image_path}")
                continue
        except Exception as e:
            logging.error(f"이미지 로드 실패: {e}")
            continue

        image_with_bboxes = draw_bounding_boxes(image, annotations, image_id)

        cv2.imshow("COCO Dataset Visualization", image_with_bboxes)
        logging.info(f"이미지 표시: {image_filename}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):  # 'n' key: next image
                break
            elif key == ord('q'):  # 'q' key: quit
                cv2.destroyAllWindows()
                logging.info("quit visualization")
                return

    cv2.destroyAllWindows()
    logging.info("모든 이미지를 표시 완료")

if __name__ == "__main__":
    try:
        visualize_coco_dataset()
    except Exception as e:
        logging.error(f"시각화 중 오류 발생: {e}")
        cv2.destroyAllWindows()