#주행가능영역 추출하는 코드.


import carla
import numpy as np
import cv2
import os
import argparse
import time
from queue import Queue

# ==============================================================================
# -- 상수 정의 (Constants)
# ==============================================================================

# 주행 가능 영역을 의미에 따라 다른 클래스로 정의
CLASS_VALUES = {
    "unlabeled": 0,       # 라벨링되지 않은 영역
    "current_lane": 1,    # 현재 주행 차선
    "lane_change": 2,     # 변경 가능한 인접 차선
    "intersection": 3     # 교차로 내 주행 가능 경로
}

# 각 클래스를 시각화하기 위한 색상 정의 (BGR 포맷)
COLOR_MAP = {
    0: [0, 0, 0],         # unlabeled: Black
    1: [255, 0, 0],       # current_lane: Blue
    2: [0, 255, 0],       # lane_change: Green
    3: [0, 0, 255]        # intersection: Red
}

# 기본 도로/차선으로 인식할 시맨틱 태그 ID
# DRIVABLE_SEG_TAGS = {7, 6} # CARLA 0.9.x 기준
DRIVABLE_SEG_TAGS = {1, 24}  # 사용자 제공 기준

# 이미지 해상도 및 카메라 설정
IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FOV = 1280, 720, 90
DATA_SAVE_INTERVAL = 1.0 # 데이터 저장 간격 (초)
OUTPUT_DIR = 'drivable_multiclass_dataset'

# ==============================================================================
# -- 도우미 함수 (Helper Functions)
# ==============================================================================

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2], K[1, 2] = w / 2.0, h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    if point_camera[2] < 0.01: return None
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]; point_img[1] /= point_img[2]
    return (int(point_img[0]), int(point_img[1]))

def create_colormap(mask):
    colormap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for value, color in COLOR_MAP.items():
        colormap[mask == value] = color
    return colormap

# ==============================================================================
# -- 메인 실행 함수
# ==============================================================================

def game_loop(args):
    client, world, actor_list = None, None, []
    
    try:
        # 1. CARLA 서버 연결
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        carla_map = world.get_map()
        print(f"현재 맵 '{carla_map.name.split('/')[-1]}'에 연결했습니다.")
        
        # 2. 데이터 저장 폴더 생성
        ensure_dir(os.path.join(OUTPUT_DIR, 'image'))
        ensure_dir(os.path.join(OUTPUT_DIR, 'label_mask'))
        ensure_dir(os.path.join(OUTPUT_DIR, 'label_txt'))
        ensure_dir(os.path.join(OUTPUT_DIR, 'colormap'))

        # 3. Ego 차량 찾기
        vehicle = None
        for _ in range(15):
            actors = world.get_actors().filter('vehicle.*')
            for actor in actors:
                if actor.attributes.get('role_name') == args.role_name:
                    vehicle = actor; break
            if vehicle: break
            print(f"'{args.role_name}' 차량을 찾는 중...")
            time.sleep(1)
        
        if not vehicle:
            print(f"오류: '{args.role_name}' 차량을 찾을 수 없습니다. manual_control.py를 먼저 실행해주세요.")
            return
        print(f"Ego 차량 (ID: {vehicle.id})을 찾았습니다. 데이터 수집을 시작합니다.")

        # 4. 동기화된 센서 설치
        sensor_queue = Queue()
        blueprint_library = world.get_blueprint_library()
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # RGB 카메라
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(IMAGE_WIDTH)); rgb_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)
        actor_list.append(rgb_cam)
        rgb_cam.listen(lambda data: sensor_queue.put(('rgb', data)))

        # 세그멘테이션 카메라
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(IMAGE_WIDTH)); seg_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=vehicle)
        actor_list.append(seg_cam)
        seg_cam.listen(lambda data: sensor_queue.put(('seg', data)))
        
        # 5. 동기 모드 설정 및 데이터 수집 루프
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True; settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        last_saved_time = 0

        while True:
            world.tick()
            try:
                s_data = [sensor_queue.get(True, 1.0) for _ in range(2)]
            except Exception: continue

            s_data_dict = {item[0]: item[1] for item in s_data}
            if not ('rgb' in s_data_dict and 'seg' in s_data_dict) or \
               s_data_dict['rgb'].frame != s_data_dict['seg'].frame:
                continue

            rgb_image = s_data_dict['rgb']
            if rgb_image.timestamp - last_saved_time < DATA_SAVE_INTERVAL:
                continue
            
            last_saved_time = rgb_image.timestamp
            frame_id = rgb_image.frame

            # --- Waypoint 기반 다중 클래스 라벨링 로직 ---
            waypoint = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True)
            drivable_lanes_waypoints = []
            
            if waypoint.is_junction:
                # 교차로: 가능한 모든 다음 경로를 "intersection" 클래스로 추가
                next_waypoints = waypoint.next(2.0)
                for w in next_waypoints:
                    # 각 경로를 15m 앞까지 추적
                    lane_wps = w.next_until_lane_end(15.0)
                    drivable_lanes_waypoints.append({"waypoints": lane_wps, "class": "intersection"})
            else:
                # 일반 도로: 현재 차선, 변경 가능 차선 탐색
                # 1. 현재 차선 ("current_lane")
                drivable_lanes_waypoints.append({"waypoints": waypoint.next_until_lane_end(30.0), "class": "current_lane"})
                # 2. 우측 변경 가능 차선 ("lane_change")
                if waypoint.lane_change & carla.LaneChange.Right:
                    right_wp = waypoint.get_right_lane()
                    if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                        drivable_lanes_waypoints.append({"waypoints": right_wp.next_until_lane_end(30.0), "class": "lane_change"})
                # 3. 좌측 변경 가능 차선 ("lane_change")
                if waypoint.lane_change & carla.LaneChange.Left:
                    left_wp = waypoint.get_left_lane()
                    if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                        drivable_lanes_waypoints.append({"waypoints": left_wp.next_until_lane_end(30.0), "class": "lane_change"})
            
            # --- 3D Waypoint를 2D 폴리곤으로 변환 ---
            K = build_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FOV)
            w2c = np.array(rgb_cam.get_transform().get_inverse_matrix())
            polygons_2d = []

            for lane in drivable_lanes_waypoints:
                lane_polygon_2d = []
                # Waypoint list가 비어있지 않은지 확인
                if not lane["waypoints"]: continue
                for wp in lane["waypoints"]:
                    half_width, r_vec = wp.lane_width / 2.0, wp.transform.get_right_vector()
                    p_left = wp.transform.location - r_vec * half_width
                    p_right = wp.transform.location + r_vec * half_width
                    
                    pt_left, pt_right = get_image_point(p_left, K, w2c), get_image_point(p_right, K, w2c)
                    if pt_left and pt_right:
                        lane_polygon_2d.insert(0, pt_left)
                        lane_polygon_2d.append(pt_right)
                
                if len(lane_polygon_2d) > 2:
                    polygons_2d.append({"polygon": np.array(lane_polygon_2d, dtype=np.int32), "class": lane["class"]})
            
            # --- 라벨 마스크 생성 및 데이터 저장 ---
            label_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
            # 우선순위가 낮은 순서부터 폴리곤을 그림 (교차로가 일반 차선을 덮어쓸 수 있도록)
            sorted_polygons = sorted(polygons_2d, key=lambda p: CLASS_VALUES[p['class']])
            
            for poly_info in sorted_polygons:
                class_value = CLASS_VALUES[poly_info["class"]]
                cv2.fillPoly(label_mask, [poly_info["polygon"]], class_value)

            filename = f"{frame_id:06d}"
            # 1. 원본 이미지 저장
            rgb_image.save_to_disk(os.path.join(OUTPUT_DIR, 'image', f"{filename}.png"))
            # 2. 라벨 마스크 저장 (각 픽셀값이 클래스 ID)
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'label_mask', f"{filename}.png"), label_mask)
            # 3. 텍스트 라벨 저장
            with open(os.path.join(OUTPUT_DIR, 'label_txt', f"{filename}.txt"), "w") as f:
                for poly_info in polygons_2d:
                    coords = " ".join([f"{p[0]},{p[1]}" for p in poly_info["polygon"]])
                    f.write(f"{poly_info['class']}:{coords}\n")
            # 4. 컬러맵 저장 (시각화용)
            colormap_image = create_colormap(label_mask)
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'colormap', f"{filename}.png"), colormap_image)

            print(f"📸 프레임 {frame_id} 데이터 저장 완료")
            cv2.imshow('Multi-Class Colormap', colormap_image)
            cv2.waitKey(1)

    finally:
        print("\n🛑 데이터 수집을 종료하고 환경을 정리합니다...")
        if world and 'original_settings' in locals():
            world.apply_settings(original_settings)
        if client and actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        cv2.destroyAllWindows()
        print("✅ 정리 완료.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CARLA Multi-Class Drivable Area Dataset Generator")
    parser.add_argument('--host', default='127.0.0.1', help='CARLA 서버 호스트 IP')
    parser.add_argument('-p', '--port', default=2000, type=int, help='CARLA 서버 TCP 포트')
    parser.add_argument('--role-name', default='hero', help="Ego 차량을 식별하기 위한 role_name")
    args = parser.parse_args()
    try:
        game_loop(args)
    except KeyboardInterrupt:
        pass
