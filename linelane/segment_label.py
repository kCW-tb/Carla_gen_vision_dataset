
import numpy as np
import weakref
import sys
import glob
import os
import cv2
import random
import time
import pygame
# ===========================
# 설정
# ===========================
# carla 클래스
# LANE_CLASS_ID = 24  # 차선 경계선 class (Carla semantic segmentation 기준)
# 14번 차량
# 15번 트럭
# 18번 오토바이
# 20번 잡동사니 건물..?
# 21번 잡동사니
# 1번 차도
# 2번 인도
# 3번 배경
# 6번 가로등
# 7번 신호등 엑터
# 8번 TrafficSign

# Carla Python API 경로 등록
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# ========== 클래스 ID → 이름 매핑 ==========
CLASS_LABELS = {
    0: 'None',
    1: 'Buildings',
    2: 'Fences',
    3: 'Other',
    4: 'Pedestrians',
    5: 'Poles',
    6: 'Road',
    7: 'RoadLines',
    8: 'Sidewalks',
    9: 'Vegetation',
    10: 'Vehicles',
    11: 'Walls',
    12: 'TrafficSigns'
}

# ========== Pygame Viewer ==========
def pygame_display_init(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA Semantic Segmentation Viewer")
    return display

# ========== 보행자 생성 함수 ==========
def spawn_pedestrians(world, blueprint_library, num_peds=10):
    spawn_points = world.get_map().get_spawn_points()
    walkers = []
    controllers = []
    batch = []
    
    walker_blueprints = blueprint_library.filter("walker.pedestrian.*")
    controller_bp = blueprint_library.find('controller.ai.walker')

    for _ in range(num_peds):
        spawn_point = random.choice(spawn_points)
        walker_bp = random.choice(walker_blueprints)
        walker_bp.set_attribute('is_invincible', 'false')
        walker = world.spawn_actor(walker_bp, spawn_point)
        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

        walkers.append(walker)
        controllers.append(controller)

    time.sleep(1.0)  # 컨트롤러가 제대로 연결될 시간 확보

    for controller in controllers:
        controller.start()
        controller.go_to_location(random.choice(spawn_points).location)
        controller.set_max_speed(1 + random.random())  # 1~2 m/s

    return walkers + controllers

# ========== 메인 ==========
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Ego 차량 생성 or 선택
    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(bp, spawn_point)
    else:
        vehicle = vehicles[0]

    # Semantic Segmentation 카메라 생성
    cam_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "90")
    transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    seg_camera = world.spawn_actor(cam_bp, transform, attach_to=vehicle)

    # 보행자 생성
    pedestrians = spawn_pedestrians(world, blueprint_library, num_peds=15)

    display = pygame_display_init(1280, 720)

    # 키보드 컨트롤러
    clock = pygame.time.Clock()
    control = carla.VehicleControl()

    shared_data = {'frame': None}

    # === 이미지 처리 콜백 ===
    def seg_callback(image, data_dict):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        class_ids = array[:, :, 2]

        vis = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

        for label_id in np.unique(class_ids):
            if label_id == 0:
                continue
            mask = (class_ids == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 300:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.putText(vis, str(label_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        vis = cv2.flip(vis, 1)
        data_dict['frame'] = vis

    seg_camera.listen(lambda image: seg_callback(image, shared_data))

    print("🕹️ W/A/S/D 키로 차량을 조종하세요. ESC로 종료")

    try:
        while True:
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            keys = pygame.key.get_pressed()
            control.throttle = 0.5 if keys[pygame.K_w] else 0.0
            control.brake = 1.0 if keys[pygame.K_s] else 0.0
            control.steer = (-1.0 if keys[pygame.K_a] else (1.0 if keys[pygame.K_d] else 0.0))
            vehicle.apply_control(control)

            if shared_data['frame'] is not None:
                surface = pygame.surfarray.make_surface(np.flip(shared_data['frame'], axis=1).swapaxes(0, 1))
                display.blit(surface, (0, 0))
                pygame.display.flip()

    finally:
        seg_camera.stop()
        seg_camera.destroy()
        for actor in pedestrians:
            actor.destroy()
        pygame.quit()
        print("🛑 종료되었습니다.")

if __name__ == '__main__':
    main()