# # 차량 객체 인식 코드.
# import math
# import random
# import time
# import queue
# import numpy as np
# import cv2
# import json
# import os
# import logging
# from datetime import datetime
# import glob
# import sys

# # CARLA Python API 경로 설정
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# import carla

# # 로깅 설정
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # 클라이언트 및 월드 설정
# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)  # 연결 타임아웃 설정
# try:
#     world = client.get_world()
# except Exception as e:
#     logging.error(f"CARLA 서버 연결 실패: {e}")
#     raise
# bp_lib = world.get_blueprint_library()

# # 맵의 스폰 포인트 가져오기
# spawn_points = world.get_map().get_spawn_points()

# # manual_control.py 코드로 ego 차량을 미리 생성해 두어야 함.
# vehicle = None  # ego 차량 변수
# for actor in world.get_actors().filter('vehicle.*'):  # 모든 차량 액터 탐색
#     # 'hero' 역할 이름 확인 (차량이 해당 객체가 아니라면 차량 변경)
#     if actor.attributes.get('role_name') == 'hero':  
#         vehicle = actor  # ego 차량 설정
#         break
# if vehicle is None:
#     raise RuntimeError("Ego vehicle with role_name 'hero' not found.")  # ego 차량 없으면 오류

# # 카메라 스폰
# camera_bp = bp_lib.find('sensor.camera.rgb')
# camera_init_trans = carla.Transform(carla.Location(z=2))
# try:
#     camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
# except Exception as e:
#     logging.error(f"카메라 스폰 실패: {e}")
#     raise
# logging.info("카메라 설정 완료")

# # 동기 모드 설정
# settings = world.get_settings()
# settings.synchronous_mode = True  # 동기 모드 활성화
# settings.fixed_delta_seconds = 0.05
# world.apply_settings(settings)

# # 센서 데이터 저장을 위한 큐 생성
# image_queue = queue.Queue()
# camera.listen(image_queue.put)

# # COCO 데이터셋 저장 디렉토리 및 JSON 파일 설정
# output_dir = "coco_dataset"
# os.makedirs(output_dir, exist_ok=True)
# image_dir = os.path.join(output_dir, "images")
# os.makedirs(image_dir, exist_ok=True)
# coco_json = {
#     "info": {
#         "description": "CARLA Vehicle Detection Dataset",
#         "version": "1.0",
#         "year": 2025,
#         "date_created": datetime.now().strftime("%Y-%m-%d")
#     },
#     "images": [],
#     "annotations": [],
#     "categories": [
#         {"id": 1, "name": "vehicle", "supercategory": "vehicle"}
#     ]
# }
# annotation_id = 1
# image_id = 1
# last_save_time = time.time()  # 마지막 저장 시간 초기화


# # 카메라 투영 행렬 생성 함수
# def build_projection_matrix(w, h, fov, is_behind_camera=False):
#     focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
#     K = np.identity(3)
#     if is_behind_camera:
#         K[0, 0] = K[1, 1] = -focal
#     else:
#         K[0, 0] = K[1, 1] = focal
#     K[0, 2] = w / 2.0
#     K[1, 2] = h / 2.0
#     return K

# # 3D 좌표를 2D 이미지 좌표로 변환하는 함수
# def get_image_point(loc, K, w2c):
#     point = np.array([loc.x, loc.y, loc.z, 1])
#     point_camera = np.dot(w2c, point)
#     point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
#     point_img = np.dot(K, point_camera)
#     if point_img[2] <= 0:  # 카메라 뒤의 점 제외
#         return None
#     point_img[0] /= point_img[2]
#     point_img[1] /= point_img[2]
#     return point_img[0:2]

# # 캔버스 내 점인지 확인하는 함수
# def point_in_canvas(pos, img_h, img_w):
#     if pos is None:
#         return False
#     return (0 <= pos[0] < img_w) and (0 <= pos[1] < img_h)

# # 카메라 속성 가져오기
# image_w = camera_bp.get_attribute("image_size_x").as_int()
# image_h = camera_bp.get_attribute("image_size_y").as_int()
# fov = camera_bp.get_attribute("fov").as_float()

# # 투영 행렬 계산
# K = build_projection_matrix(image_w, image_h, fov)

# # NPC 차량 스폰
# for i in range(20):  # 성능을 위해 50에서 20으로 줄임
#     vehicle_bp = random.choice(bp_lib.filter('vehicle'))
#     npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
#     if npc:
#         npc.set_autopilot(True)
#         logging.info(f"NPC 차량 스폰 성공: ID {npc.id}")

# # 메인 루프
# try:
#     while True:
#         # 에고 차량 상태 확인
#         if not vehicle.is_alive:
#             logging.warning("에고 차량이 월드에서 사라짐!")
#             break

#         # 월드 틱 및 이미지 가져오기
#         world.tick()
#         image = image_queue.get()

#         # 이미지를 RGB 배열로 변환
#         img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
#         img_display = img.copy()  # 표시용 이미지 복사 (바운딩 박스 포함)

#         # 월드에서 카메라로의 변환 행렬 가져오기
#         world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

#         # COCO annotations 리스트
#         annotations = []

#         # 현재 시간 확인
#         current_time = time.time()

#         # 바운딩 박스 및 어노테이션 계산
#         for npc in world.get_actors().filter('*vehicle*'):
#             if npc.id != vehicle.id:  # 에고 차량 제외
#                 bb = npc.bounding_box
#                 dist = npc.get_transform().location.distance(vehicle.get_transform().location)

#                 # 70m 이내의 차량만 처리
#                 if dist < 70:
#                     # 카메라 앞에 있는지 확인 (전방 벡터와 내적 계산)
#                     forward_vec = vehicle.get_transform().get_forward_vector()
#                     ray = npc.get_transform().location - vehicle.get_transform().location
#                     if forward_vec.dot(ray) > 0:
#                         verts = [v for v in bb.get_world_vertices(npc.get_transform())]
#                         x_max = -float('inf')
#                         x_min = float('inf')
#                         y_max = -float('inf')
#                         y_min = float('inf')
#                         valid_points = False

#                         for vert in verts:
#                             p = get_image_point(vert, K, world_2_camera)
#                             if p is not None and point_in_canvas(p, image_h, image_w):
#                                 valid_points = True
#                                 x_max = max(x_max, p[0])
#                                 x_min = min(x_min, p[0])
#                                 y_max = max(y_max, p[1])
#                                 y_min = min(y_min, p[1])

#                         # 유효한 바운딩 박스가 있는 경우
#                         if valid_points and x_max > x_min and y_max > y_min:
#                             # COCO 어노테이션 추가
#                             bbox = [
#                                 float(x_min),
#                                 float(y_min),
#                                 float(x_max - x_min),
#                                 float(y_max - y_min)
#                             ]
#                             annotations.append({
#                                 "id": annotation_id,
#                                 "image_id": image_id,
#                                 "category_id": 1,
#                                 "bbox": bbox,
#                                 "area": bbox[2] * bbox[3],
#                                 "iscrowd": 0
#                             })
#                             annotation_id += 1

#                             # 바운딩 박스 그리기 (표시용 이미지에만)
#                             cv2.line(img_display, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 1)
#                             cv2.line(img_display, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
#                             cv2.line(img_display, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 1)
#                             cv2.line(img_display, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)

#         # 1초마다 이미지 저장 및 어노테이션 추가
#         if current_time - last_save_time >= 1.0:
#             # 이미지 저장 (바운딩 박스 없이)
#             image_filename = f"image_{image_id:06d}.jpg"
#             image_path = os.path.join(image_dir, image_filename)
#             try:
#                 cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
#                 logging.info(f"이미지 저장: {image_path}")
#             except Exception as e:
#                 logging.error(f"이미지 저장 실패: {e}")

#             # COCO 이미지 정보 추가
#             coco_json["images"].append({
#                 "id": image_id,
#                 "file_name": image_filename,
#                 "width": image_w,
#                 "height": image_h
#             })
#             coco_json["annotations"].extend(annotations)
#             image_id += 1
#             last_save_time = current_time  # 마지막 저장 시간 업데이트

#         # 이미지 표시 (바운딩 박스 포함)
#         cv2.imshow('ImageWindowName', img_display)
#         if cv2.waitKey(1) == ord('q'):
#             break

# finally:
#     # COCO JSON 파일 저장
#     try:
#         with open(os.path.join(output_dir, "annotations.json"), 'w') as f:
#             json.dump(coco_json, f, indent=2)
#         logging.info("COCO 어노테이션 파일 저장 완료")
#     except Exception as e:
#         logging.error(f"COCO 어노테이션 파일 저장 실패: {e}")

#     # 리소스 정리 (에고 차량 제외)
#     logging.info("리소스 정리 시작")
#     if camera.is_alive:
#         camera.destroy()
#     for npc in world.get_actors().filter('*vehicle*'):
#         if npc.id != vehicle.id and npc.is_alive:
#             npc.destroy()
#     cv2.destroyAllWindows()
#     settings.synchronous_mode = False
#     world.apply_settings(settings)
#     logging.info("리소스 정리 완료, 에고 차량은 유지됨")

import math
import random
import time
import queue
import numpy as np
import cv2
import json
import os
import logging
from datetime import datetime
import glob
import sys

# CARLA Python API 경로 설정
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === 설정 ===
IM_WIDTH = 800
IM_HEIGHT = 600
SAVE_INTERVAL = 1.0  # 초 단위
VISIBLE_THRESHOLD = 0.05  # 5% 이상 보일 때만 저장
CLASS_IDS = {
    'vehicle': {'carla_ids': [10, 14, 15, 18], 'yolo_id': 0},  # 차량, 트럭, 오토바이 등
    'pedestrian': {'carla_ids': [12], 'yolo_id': 1}  # 보행자
}
OUTPUT_DIR = "yolo_dataset"
COCO_DIR = "coco_dataset"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
os.makedirs(os.path.join(COCO_DIR, "images"), exist_ok=True)

# === COCO JSON 초기화 ===
coco_json = {
    "info": {
        "description": "CARLA Vehicle and Pedestrian Detection Dataset",
        "version": "1.0",
        "year": 2025,
        "date_created": datetime.now().strftime("%Y-%m-%d")
    },
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "vehicle", "supercategory": "vehicle"},
        {"id": 1, "name": "pedestrian", "supercategory": "pedestrian"}
    ]
}
annotation_id = 1
image_id = 1

# === client 및 world 초기화 ===
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# === ego vehicle 찾기 ===
vehicle = None
for actor in world.get_actors().filter('vehicle.*'):
    if actor.attributes.get('role_name') == 'hero':
        vehicle = actor
        break
if not vehicle:
    raise RuntimeError("Ego vehicle with role_name 'hero' not found.")

# === NPC 차량 및 보행자 스폰 ===
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)

npc_list = []
# 차량 스폰
for sp in spawn_points[:20]:
    npc_bp = random.choice(bp_lib.filter('vehicle.*'))
    npc = world.try_spawn_actor(npc_bp, sp)
    if npc:
        npc.set_autopilot(True)
        npc_list.append(npc)
        logging.info(f"Spawned NPC vehicle {npc.type_id} at {sp.location}")

# 보행자 스폰
pedestrian_list = []
controller_list = []
for sp in spawn_points[20:30]:  # 10명의 보행자
    ped_bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
    try:
        ped = world.try_spawn_actor(ped_bp, sp)
        if ped:
            pedestrian_list.append(ped)
            controller_bp = bp_lib.find('controller.ai.walker')
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=ped)
            if controller:
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller_list.append(controller)
                logging.info(f"Spawned pedestrian {ped.type_id} at {sp.location}")
    except RuntimeError as e:
        logging.warning(f"Pedestrian spawn failed: {e}")

# === 센서 블루프린트 ===
rgb_bp = bp_lib.find('sensor.camera.rgb')
rgb_bp.set_attribute("image_size_x", str(IM_WIDTH))
rgb_bp.set_attribute("image_size_y", str(IM_HEIGHT))
rgb_bp.set_attribute("fov", "90")
rgb_bp.set_attribute("sensor_tick", "0.05")  # 20fps

seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
seg_bp.set_attribute("image_size_x", str(IM_WIDTH))
seg_bp.set_attribute("image_size_y", str(IM_HEIGHT))
seg_bp.set_attribute("fov", "90")
seg_bp.set_attribute("sensor_tick", "0.05")  # 20fps

# === 센서 스폰 ===
cam_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)
seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=vehicle)

# === 동기 모드 설정 ===
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# === 큐 ===
rgb_q = queue.Queue()
seg_q = queue.Queue()
rgb_cam.listen(rgb_q.put)
seg_cam.listen(seg_q.put)

# === Helper 함수들 ===
def build_K(w, h, fov):
    f = w / (2.0 * np.tan(fov * np.pi / 360))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    p = np.array([loc.x, loc.y, loc.z, 1.0])
    pc = np.dot(w2c, p)
    pc = [pc[1], -pc[2], pc[0]]
    if pc[2] <= 0:
        return None
    pp = np.dot(K, pc)
    pp[0] /= pp[2]
    pp[1] /= pp[2]
    return pp[0:2]

def point_in_image(p, h, w):
    return p is not None and 0 <= p[0] < w and 0 <= p[1] < h

# === 메인 루프 ===
image_id = 0
last_save = time.time()
K = build_K(IM_WIDTH, IM_HEIGHT, float(rgb_bp.get_attribute("fov")))

try:
    while True:
        world.tick()

        rgb_img = rgb_q.get()
        seg_img = seg_q.get()

        # 카메라 pose
        cam_tf = rgb_cam.get_transform()
        w2c = np.array(cam_tf.get_inverse_matrix())

        # RGB image
        rgb = np.frombuffer(rgb_img.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))
        rgb = rgb[:, :, :3][:, :, ::-1].copy()  # RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vis_img = rgb.copy()

        # segmentation class ID (R 채널만)
        seg_raw = np.frombuffer(seg_img.raw_data, dtype=np.uint8).reshape((IM_HEIGHT, IM_WIDTH, 4))
        class_map = seg_raw[:, :, 2]  # Red = class ID

        yolo_lines = []
        coco_annotations = []

        # 차량 및 보행자 처리
        for actor in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*")):
            if actor.id == vehicle.id:
                continue

            dist = actor.get_transform().location.distance(vehicle.get_transform().location)
            if dist > 70:
                continue

            fwd = vehicle.get_transform().get_forward_vector()
            rel = actor.get_transform().location - vehicle.get_transform().location
            if fwd.dot(rel) < 0:
                continue

            verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            x_min, y_min = IM_WIDTH, IM_HEIGHT
            x_max, y_max = 0, 0
            valid = False

            for v in verts:
                p = get_image_point(v, K, w2c)
                if point_in_image(p, IM_HEIGHT, IM_WIDTH):
                    x, y = int(p[0]), int(p[1])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    valid = True

            if not valid or x_max <= x_min or y_max <= y_min:
                continue

            crop = class_map[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            # 클래스 ID 확인
            class_name = 'pedestrian' if actor.type_id.startswith('walker.pedestrian') else 'vehicle'
            carla_ids = CLASS_IDS[class_name]['carla_ids']
            yolo_class_id = CLASS_IDS[class_name]['yolo_id']

            visible_mask = np.isin(crop, carla_ids)
            visible_ratio = np.sum(visible_mask) / (crop.shape[0] * crop.shape[1])
            if visible_ratio < VISIBLE_THRESHOLD:
                continue

            # YOLO format
            x_center = ((x_min + x_max) / 2) / IM_WIDTH
            y_center = ((y_min + y_max) / 2) / IM_HEIGHT
            w_norm = (x_max - x_min) / IM_WIDTH
            h_norm = (y_max - y_min) / IM_HEIGHT

            yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{class_name}: {visible_ratio*100:.1f}%", (x_min, y_min - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # COCO format
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": yolo_class_id,
                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                "area": float((x_max - x_min) * (y_max - y_min)),
                "iscrowd": 0
            })
            annotation_id += 1

        now = time.time()
        if now - last_save >= SAVE_INTERVAL:
            img_name = f"{image_id:06d}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", f"{img_name}.jpg"), rgb)
            with open(os.path.join(OUTPUT_DIR, "labels", f"{img_name}.txt"), "w") as f:
                f.writelines("\n".join(yolo_lines) + "\n")
            logging.info(f"Saved image & labels: {img_name}")

            # COCO JSON 업데이트
            coco_json["images"].append({
                "id": image_id,
                "file_name": f"{img_name}.jpg",
                "width": IM_WIDTH,
                "height": IM_HEIGHT
            })
            coco_json["annotations"].extend(coco_annotations)
            
            image_id += 1
            last_save = now

        cv2.imshow("View", vis_img)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # COCO JSON 저장
    try:
        with open(os.path.join(COCO_DIR, "annotations.json"), 'w') as f:
            json.dump(coco_json, f, indent=2)
        logging.info("COCO annotations saved")
    except Exception as e:
        logging.error(f"Failed to save COCO annotations: {e}")

    # 리소스 정리
    rgb_cam.stop()
    seg_cam.stop()
    rgb_cam.destroy()
    seg_cam.destroy()
    for npc in npc_list:
        npc.destroy()
    for ped in pedestrian_list:
        ped.destroy()
    for controller in controller_list:
        controller.stop()
        controller.destroy()
    world.apply_settings(carla.WorldSettings(synchronous_mode=False))
    cv2.destroyAllWindows()
    logging.info("Cleanup completed")