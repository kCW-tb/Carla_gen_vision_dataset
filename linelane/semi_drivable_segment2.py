#주행가능영역 추출하는 코드.
import carla
import numpy as np
import cv2
import os
import argparse
import time
from queue import Queue
import math

# ==============================================================================
# -- 상수 및 설정 (Constants & Settings)
# ==============================================================================
CLASS_VALUES = {"unlabeled": 0, "primary_path": 1, "lane_change": 2}
COLOR_MAP = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}
DRIVABLE_SEG_TAGS = {1, 24}
IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FOV = 1280, 720, 90
WAYPOINT_DISTANCE_GENERAL = 50.0
WAYPOINT_DISTANCE_JUNCTION = 25.0
OUTPUT_DIR = 'drivable_twoclass_dataset_robust'

# ==============================================================================
# -- 도우미 함수 (Helper Functions)
# ==============================================================================
def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal; K[0, 2], K[1, 2] = w / 2.0, h / 2.0
    return K

def get_image_point(loc, K, w2c):
    p = np.array([loc.x, loc.y, loc.z, 1])
    p_cam = np.dot(w2c, p)
    p_cam = np.array([p_cam[1], -p_cam[2], p_cam[0]])
    if p_cam[2] < 0.01: return None
    p_img = np.dot(K, p_cam)
    p_img[0] /= p_img[2]; p_img[1] /= p_img[2]
    return (int(p_img[0]), int(p_img[1]))

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
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        world = client.get_world()
        carla_map = world.get_map()
        print(f"현재 맵 '{carla_map.name.split('/')[-1]}'에 연결했습니다.")

        vehicle = None
        for i in range(15):
            actors = world.get_actors().filter('vehicle.*')
            for actor in actors:
                if actor.attributes.get('role_name') == args.role_name:
                    vehicle = actor; break
            if vehicle: break
            print(f"'{args.role_name}' 차량을 찾는 중... ({i+1}/15)")
            time.sleep(1)

        if not vehicle:
            print(f"\n오류: '{args.role_name}' 차량을 찾을 수 없습니다.")
            return

        print(f"Ego 차량 (ID: {vehicle.id})을 찾았습니다. 데이터 수집을 시작합니다.")

        ensure_dir(os.path.join(OUTPUT_DIR, 'image'))
        ensure_dir(os.path.join(OUTPUT_DIR, 'label_mask'))
        ensure_dir(os.path.join(OUTPUT_DIR, 'label_txt'))
        ensure_dir(os.path.join(OUTPUT_DIR, 'colormap'))
        
        sensor_queue = Queue()
        bp_lib = world.get_blueprint_library()
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(IMAGE_WIDTH)); rgb_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)
        actor_list.append(rgb_cam)
        rgb_cam.listen(lambda data: sensor_queue.put(('rgb', data)))
        seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(IMAGE_WIDTH)); seg_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=vehicle)
        actor_list.append(seg_cam)
        seg_cam.listen(lambda data: sensor_queue.put(('seg', data)))
        
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True; settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        while True:
            world.tick()
            try:
                data_dict = {}
                for _ in range(2):
                    sensor_type, data = sensor_queue.get(True, 1.0)
                    data_dict[sensor_type] = data
            except Exception:
                print("⚠️ 센서 데이터 대기 시간 초과."); continue

            frame_id = data_dict['rgb'].frame
            
            waypoint = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            drivable_lanes_waypoints = []
            
            if waypoint.is_junction:
                next_wps = waypoint.next(2.0)
                for w in next_wps:
                    lane_wps = w.next_until_lane_end(WAYPOINT_DISTANCE_JUNCTION)
                    drivable_lanes_waypoints.append({"waypoints": [w] + lane_wps, "class": "primary_path"})
            else:
                current_lane_wps = waypoint.next_until_lane_end(WAYPOINT_DISTANCE_GENERAL)
                drivable_lanes_waypoints.append({"waypoints": [waypoint] + current_lane_wps, "class": "primary_path"})
                
                if waypoint.lane_change & carla.LaneChange.Right:
                    right_wp = waypoint.get_right_lane()
                    if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                        right_lane_wps = right_wp.next_until_lane_end(WAYPOINT_DISTANCE_GENERAL)
                        drivable_lanes_waypoints.append({"waypoints": [right_wp] + right_lane_wps, "class": "lane_change"})
                
                if waypoint.lane_change & carla.LaneChange.Left:
                    left_wp = waypoint.get_left_lane()
                    if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                        left_lane_wps = left_wp.next_until_lane_end(WAYPOINT_DISTANCE_GENERAL)
                        drivable_lanes_waypoints.append({"waypoints": [left_wp] + left_lane_wps, "class": "lane_change"})
            
            K = build_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FOV)
            w2c = np.array(rgb_cam.get_transform().get_inverse_matrix())
            polygons_2d = []

            # --- ### 수정된 폴리곤 생성 로직 ### ---
            for lane in drivable_lanes_waypoints:
                if not lane["waypoints"]: continue
                
                left_points = []
                right_points = []

                # 1. 경로를 따라 좌/우측 2D 경계점을 각각 수집합니다.
                for wp in lane["waypoints"]:
                    half_width, r_vec = wp.lane_width / 2.0, wp.transform.get_right_vector()
                    pt_left = get_image_point(wp.transform.location - r_vec * half_width, K, w2c)
                    pt_right = get_image_point(wp.transform.location + r_vec * half_width, K, w2c)
                    
                    if pt_left and pt_right:
                        left_points.append(pt_left)
                        right_points.append(pt_right)
                
                # 2. 수집된 점들을 합쳐 하나의 폐곡선 폴리곤을 만듭니다.
                # (오른쪽으로 갔다가 -> 왼쪽으로 돌아오는 순서)
                if len(left_points) > 1 and len(right_points) > 1:
                    # right_points는 순서대로, left_points는 역순으로 합쳐야 닫힌 폴리곤이 됨
                    full_polygon = right_points + left_points[::-1]
                    polygons_2d.append({"polygon": np.array(full_polygon, dtype=np.int32), "class": lane["class"]})
            
            label_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
            sorted_polygons = sorted(polygons_2d, key=lambda p: CLASS_VALUES[p['class']])
            for poly_info in sorted_polygons: cv2.fillPoly(label_mask, [poly_info["polygon"]], CLASS_VALUES[poly_info["class"]])
            
            colormap_image = create_colormap(label_mask)
            cv2.imshow('Robust Two-Class Drivable Area', colormap_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            filename = f"{frame_id:06d}"
            data_dict['rgb'].save_to_disk(os.path.join(OUTPUT_DIR, 'image', f"{filename}.png"))
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'label_mask', f"{filename}.png"), label_mask)
            with open(os.path.join(OUTPUT_DIR, 'label_txt', f"{filename}.txt"), "w") as f:
                for poly_info in polygons_2d:
                    coords = " ".join([f"{p[0]},{p[1]}" for p in poly_info["polygon"]])
                    f.write(f"{poly_info['class']}:{coords}\n")
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'colormap', f"{filename}.png"), colormap_image)
            print(f"📸 프레임 {frame_id} 데이터 저장 완료")

    finally:
        print("\n🛑 환경을 정리합니다...")
        if world and 'original_settings' in locals(): world.apply_settings(original_settings)
        if client and actor_list: client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        cv2.destroyAllWindows()
        print("✅ 정리 완료.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Robust CARLA Two-Class Drivable Area Generator")
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('-p', '--port', default=2000, type=int)
    parser.add_argument('--role-name', default='hero', help="Ego 차량 식별용 role_name")
    args = parser.parse_args()
    try: game_loop(args)
    except KeyboardInterrupt: pass
