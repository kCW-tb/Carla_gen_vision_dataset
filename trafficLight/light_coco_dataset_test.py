import json
import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터셋 디렉토리 설정
dataset_dir = "./TrafficLight/dataset/"
image_dir = os.path.join("C:/CARLA-0.10.0/TrafficLight/dataset/images")
annotation_file = os.path.join("C:/CARLA-0.10.0/TrafficLight/dataset/annotations.json")

def load_coco_dataset():
    if not os.path.exists(annotation_file):
        logging.error(f"Annotation file not found: {annotation_file}")
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    try:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        logging.info("COCO 어노테이션 파일 로드 완료")
        
        # 카테고리 ID와 이름 매핑
        category_map = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
        return coco_data, category_map
    except Exception as e:
        logging.error(f"어노테이션 파일 로드 실패: {e}")
        raise

def draw_bounding_boxes(image, annotations, image_id, category_map):
    for ann in annotations:
        if ann["image_id"] == image_id:
            bbox = ann["bbox"]
            x, y, w, h = [int(v) for v in bbox]
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                logging.warning(f"잘못된 BBox 좌표: {bbox} (이미지 크기: {image.shape[1]}x{image.shape[0]})")
                continue
            
            # 카테고리 ID에 따른 색상 및 레이블 설정
            category_id = ann.get("category_id")
            category_name = category_map.get(category_id, "Unknown")
            
            # 카테고리에 따라 색상 지정 (BGR 형식)
            if category_name == "Red":
                color = (0, 0, 255)  # 빨간색
            elif category_name == "Green":
                color = (0, 255, 0)  # 초록색
            elif category_name == "Yellow":
                color = (0, 255, 255)  # 노란색
            else:
                color = (128, 128, 128)  # 회색 (기본값)
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            # 카테고리 이름 표시
            cv2.putText(image, category_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def visualize_coco_dataset():
    coco_data, category_map = load_coco_dataset()
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

        image_with_bboxes = draw_bounding_boxes(image, annotations, image_id, category_map)

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