import random
import time
import queue
import os
import cv2
import numpy as np
import math
import glob
import sys
from pathlib import Path

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# 클래스 ID와 이름 매핑
CLASS_IDS = {
    'Red': 0,
    'Green': 1,
    'Yellow': 2
}

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    if point_img[2] <= 0:  # 카메라 뒤의 점 제외
        return None
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def load_last_image_id(labels_dir):
    """labels.txt에서 마지막 image_id를 로드"""
    labels_path = labels_dir / "labels.txt"
    max_image_id = -1
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    image_id = int(line.split()[0])
                    max_image_id = max(max_image_id, image_id)
    return max_image_id + 1

def append_labels(labels_path, image_id, yolo_labels):
    """labels.txt에 레이블 데이터를 추가"""
    with open(labels_path, 'a') as f:
        for label in yolo_labels:
            f.write(f"{image_id} {label}\n")

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # 데이터셋 저장 경로 설정
    dataset_dir = Path("./dataset")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 마지막 image_id 로드
    image_id = load_last_image_id(labels_dir)

    try:
        # 동기 모드 설정
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.0333  # 30 FPS
        world.apply_settings(settings)
        print("동기 모드 설정 완료 (30 FPS)")

        # ego 차량 탐색
        vehicle = None
        for actor in world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                vehicle = actor
                break
        if vehicle is None:
            raise RuntimeError("Ego 차량 (role_name='hero')를 찾을 수 없습니다.")

        # 카메라 설정
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(z=2))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # 카메라 파라미터
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        K = build_projection_matrix(image_w, image_h, fov)

        # 마지막 저장 시간
        last_save_time = time.time()

        world.tick()
        try:
            image = image_queue.get(timeout=1.0)
        except queue.Empty:
            raise RuntimeError("초기 카메라 데이터를 수신하지 못했습니다.")

        while True:
            world.tick()
            try:
                image = image_queue.get(timeout=1.0)
            except queue.Empty:
                print("카메라 데이터 수신 실패")
                continue

            # 원본 이미지 처리 (BGRA로 가정)
            img_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)  # BGRA에서 BGR로 변환
            img_display = img_bgr.copy()  # 시각화용 이미지 복사

            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 신호등 바운딩 박스 + 상태 액터 가져오기
            bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            traffic_lights = world.get_actors().filter("traffic.traffic_light*")

            # 프레임별 YOLO 레이블
            yolo_labels = []
            has_valid_class = False

            for bb in bounding_box_set:
                # 보행자 신호등 필터링 (높이로 차량용 신호등만 인식)
                if bb.location.z < 3.0:
                    continue
                # 60m 이상 멀리 있는 신호등 제외
                if bb.location.distance(vehicle.get_transform().location) > 60:
                    continue

                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 1:
                    continue

                # 가장 가까운 신호등 액터 찾기
                closest_tl = None
                min_dist = float('inf')
                for tl in traffic_lights:
                    dist = tl.get_location().distance(bb.location)
                    if dist < min_dist:
                        closest_tl = tl
                        min_dist = dist

                if closest_tl is None:
                    continue

                # XY 평면 각도 계산
                cam_transform = camera.get_transform()
                camera_forward = cam_transform.get_forward_vector()
                tl_y_plus = closest_tl.get_transform().rotation.get_right_vector()

                # XY 평면 투영
                cam_forward_xy = carla.Vector3D(camera_forward.x, camera_forward.y, 0)
                tl_y_plus_xy = carla.Vector3D(tl_y_plus.x, tl_y_plus.y, 0)

                # 단위 벡터 정규화
                cam_forward_xy = cam_forward_xy.make_unit_vector()
                tl_y_plus_xy = tl_y_plus_xy.make_unit_vector()

                # 내적 및 각도 계산
                cos_angle = cam_forward_xy.dot(tl_y_plus_xy)
                angle_deg = math.degrees(math.acos(max(min(cos_angle, 1.0), -1.0)))
                cross_z = cam_forward_xy.x * tl_y_plus_xy.y - cam_forward_xy.y * tl_y_plus_xy.x
                if cross_z < 0:
                    angle_deg = 360 - angle_deg

                # 클래스 및 BBox 설정
                class_id = None
                class_name = None
                if 110 <= angle_deg <= 250:
                    state = closest_tl.get_state()
                    if state == carla.TrafficLightState.Red:
                        class_id = CLASS_IDS['Red']
                        class_name = 'Red'
                        has_valid_class = True
                    elif state == carla.TrafficLightState.Green:
                        class_id = CLASS_IDS['Green']
                        class_name = 'Green'
                        has_valid_class = True
                    elif state == carla.TrafficLightState.Yellow:
                        class_id = CLASS_IDS['Yellow']
                        class_name = 'Yellow'
                        has_valid_class = True

                if class_id is None:
                    continue

                # BBox計算
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                points = [p for p in [get_image_point(v, K, world_2_camera) for v in verts] if p is not None]
                if not points:
                    continue
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = max(0, min(x_coords)), min(image_w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(image_h, max(y_coords))
                width = x_max - x_min
                height = y_max - y_min

                # 유효한 BBox만 추가
                if width > 0 and height > 0:
                    # YOLO 형식: 정규화된 좌표
                    x_center = (x_min + x_max) / 2 / image_w
                    y_center = (y_min + y_max) / 2 / image_h
                    width_norm = width / image_w
                    height_norm = height / image_h
                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

                    # 시각화: 바운딩 박스 그리기
                    color = (0, 0, 255) if class_name == 'Red' else (0, 255, 0) if class_name == 'Green' else (0, 255, 255)
                    cv2.rectangle(img_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 1)
                    cv2.putText(img_display, class_name, (int(x_min), int(y_min) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 1초 간격 및 유효 클래스 확인 후 저장
            current_time = time.time()
            if has_valid_class and current_time - last_save_time >= 1.0:
                # 이미지 저장
                image_filename = f"image_{image_id:06d}.png"
                image_path = images_dir / image_filename
                try:
                    cv2.imwrite(str(image_path), img_bgr)
                except Exception as e:
                    print(f"이미지 저장 실패: {image_path}, {e}")
                    continue

                # 레이블 추가
                labels_path = labels_dir / "labels.txt"
                try:
                    append_labels(labels_path, image_id, yolo_labels)
                except Exception as e:
                    print(f"레이블 추가 실패: {labels_path}, {e}")
                    continue

                # 실시간 시각화
                cv2.imshow("Traffic Light Detection", img_display)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                print(f"데이터셋 저장: {image_filename}, labels.txt")
                image_id += 1
                last_save_time = current_time

    finally:
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if 'camera' in locals():
            camera.stop()
            camera.destroy()
        cv2.destroyAllWindows()
        print("리소스 정리 완료")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n사용자에 의해 종료됨')
    except Exception as e:
        print(f"에러 발생: {e}")