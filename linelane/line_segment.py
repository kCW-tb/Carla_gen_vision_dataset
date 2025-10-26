# 클래스 확인하는 코드

import sys
import glob
import numpy as np
import cv2
from PIL import Image
import os
import time

# CARLA Python API 경로 설정
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# ===========================
# 설정
# ===========================
# carla 클래스
# 14번 차량
# 15번 트럭
# 18번 오토바이
# 20번 잡동사니 건물..?
# 21번 잡동사니
# LANE_CLASS_ID = 24  # 차선 경계선 class (Carla semantic segmentation 기준)
# 1번 차도
# 2번 인도
# 3번 배경
# 6번 가로등
# 7번 신호등 엑터
# 8번 TrafficSign
# 12번 사람.
LANE_CLASS_ID = 24 # 차선 경계선 class (Carla semantic segmentation 기준)
SAVE_DURATION = 120  # 저장할 시간(초)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# OUTPUT_DIR = 'lane_dataset'  # 저장 경로
OUTPUT_DIR = 'drivable_dataset'  # 저장 경로

# ===========================
# 센서 설정
# ===========================
def spawn_segmentation_camera(world, vehicle):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    cam_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", "0.2")  # 0.2초마다 프레임 생성 (5fps)

    transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
    return camera

def spawn_traffic(world, num_vehicles=20):
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')

    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < num_vehicles:
        num_vehicles = len(spawn_points)

    vehicles = []
    for i in range(num_vehicles):
        bp = np.random.choice(vehicle_blueprints)
        transform = spawn_points[i]
        npc = world.try_spawn_actor(bp, transform)
        if npc is not None:
            npc.set_autopilot(True)
            vehicles.append(npc)
    print(f"🚗 총 {len(vehicles)} 대의 AI 차량이 스폰되었습니다.")
    return vehicles
# ===========================
# YOLO Segmentation 저장
# ===========================
def process_and_save(image, frame_id):
    # Carla에서 받은 segmentation 이미지를 numpy 배열로 변환
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]
    segmentation_mask = array[:, :, 2]  # Red 채널이 class ID


    # 차선(RoadLines)만 추출
    # mask = (segmentation_mask == LANE_CLASS_ID).astype(np.uint8) * 255
    # 차선 + 도로 추출
    INCLUDED_CLASSES = [1, 24]
    mask = np.isin(segmentation_mask, INCLUDED_CLASSES).astype(np.uint8) * 255

    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []
    for contour in contours:
        if len(contour) < 6:
            continue  # 너무 작은 contour는 무시
        # 정규화 (YOLO 포맷: 0~1 좌표)
        normalized = [(pt[0][0] / IMAGE_WIDTH, pt[0][1] / IMAGE_HEIGHT) for pt in contour]
        flattened = ' '.join([f"{x:.6f} {y:.6f}" for x, y in normalized])
        label_lines.append(f"0 {flattened}")  # class 0 (lane)

    # 이미지 저장 (RGB 변환)
    rgb_image = Image.fromarray(array).convert("RGB")
    img_path = os.path.join(OUTPUT_DIR, 'images', f"{frame_id:06d}.jpg")
    label_path = os.path.join(OUTPUT_DIR, 'labels', f"{frame_id:06d}.txt")

    rgb_image.save(img_path)

    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))
    
    # ------------------------
    # 시각화: 원본 이미지 위에 차선 마스크 반투명 오버레이
    # ------------------------
    # OpenCV용 BGR 이미지로 변환
    bgr_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

    # 마스크 컬러 (녹색)
    color_mask = np.zeros_like(bgr_image)
    color_mask[mask == 255] = (0, 255, 0)  # 녹색

    # 반투명 합성 (원본 70%, 마스크 30%)
    overlayed = cv2.addWeighted(bgr_image, 0.7, color_mask, 0.3, 0)

    # OpenCV 윈도우에 출력
    cv2.imshow('Segmentation Overlay', overlayed)
    cv2.waitKey(1)  # 1ms 대기 (필수, 그래야 이미지가 출력됨)



# ===========================
# 메인 실행
# ===========================
def main():
    # Carla 클라이언트 연결
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # 차량 선택
    vehicles = world.get_actors().filter('vehicle.*')
    if not vehicles:
        print("🚗 차량이 없습니다. 하나 스폰해 주세요.")
        return
    ego_vehicle = vehicles[0]

    # 추가: 트래픽 생성
    traffic_vehicles = spawn_traffic(world, num_vehicles=30)

    # 저장 폴더 생성
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

    # 센서 설정
    camera = spawn_segmentation_camera(world, ego_vehicle)

    # 프레임 저장 루프
    frame = {'id': 0}
    def callback(image):
        process_and_save(image, frame['id'])
        print(f"📸 저장됨: frame {frame['id']}")
        frame['id'] += 1

    camera.listen(callback)

    try:
        print("🟢 차선 데이터 수집 시작...")
        time.sleep(SAVE_DURATION)
    finally:
        camera.stop()
        camera.destroy()
        # 추가: 트래픽 차량 정리
        for v in traffic_vehicles:
            v.destroy()
        print("🛑 수집 종료 및 모든 차량 제거 완료.")


if __name__ == '__main__':
    main()
