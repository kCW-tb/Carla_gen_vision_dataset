import carla # CARLA 시뮬레이터 API를 사용하기 위한 라이브러리
import random # 무작위 값 생성을 위한 라이브러리
import time # 시간 관련 기능을 위한 라이브러리
import pygame # Pygame을 사용해 시뮬레이션 화면을 렌더링
import numpy as np # 넘파이를 사용해 이미지 데이터를 배열로 처리

# Pygame 윈도우 크기 설정
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

def pygame_init():
    """
    Pygame을 초기화하고 지정된 크기의 디스플레이 윈도우를 생성합니다.
    """
    pygame.init()
    # HWSURFACE: 하드웨어 가속 표면 사용
    # DOUBLEBUF: 더블 버퍼링 사용 (화면 깜빡임 방지)
    return pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

def process_image(image, display):
    """
    CARLA 센서(카메라)에서 받은 이미지 데이터를 Pygame 표면으로 변환하고
    화면에 그립니다.

    Args:
        image: CARLA의 carla.Image 객체
        display: Pygame의 디스플레이 표면 객체
    """
    # 이미지 데이터를 넘파이 배열로 변환 (uint8 타입)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # (높이, 너비, 4채널) 형태로 배열 재구성 (RGBA)
    array = np.reshape(array, (image.height, image.width, 4))
    # RGB 채널만 추출 (알파 채널 제거)
    array = array[:, :, :3]
    # RGB 순서로 채널 순서 변경
    array = array[:, :, ::-1]
    
    # 넘파이 배열을 Pygame 표면으로 변환 (축 교환 필요)
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    # 생성된 표면을 디스플레이에 그립니다
    display.blit(surface, (0, 0))

def main():
    """
    CARLA 시뮬레이션의 메인 로직을 실행하는 함수입니다.
    
    - CARLA 서버에 연결합니다.
    - 에고(ego) 차량과 NPC 차량을 생성합니다.
    - 카메라 센서를 부착하고 Pygame에 화면을 렌더링합니다.
    - 에고 차량의 자율 주행 경로를 설정합니다.
    - 시뮬레이션 루프를 실행하며, 차량들의 바운딩 박스를 그립니다.
    - 시뮬레이션 종료 시 모든 액터(actor)를 제거하고 리소스를 정리합니다.
    """
    # CARLA 클라이언트 생성 및 연결
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    ego_vehicle = None
    camera_sensor = None
    original_settings = None
    npc_vehicles = [] # NPC 차량 객체를 저장할 리스트

    try:
        display = pygame_init()
        original_settings = world.get_settings()
        
        # 동기화 모드 설정
        settings = world.get_settings()
        settings.synchronous_mode = True # 동기 모드 활성화
        settings.fixed_delta_seconds = 0.05 # 고정된 시간 간격 (20 FPS)
        world.apply_settings(settings)

        # 트래픽 매니저 설정
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True) # 트래픽 매니저도 동기 모드 사용

        # 블루프린트 라이브러리에서 차량 및 센서 설계도 가져오기
        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.find('vehicle.lincoln.mkz')
        ego_bp.set_attribute('role_name', 'ego_vehicle')

        # 맵의 스폰 포인트 가져오기
        spawn_points = world.get_map().get_spawn_points()
        
        # 미리 정의된 경로의 스폰 포인트 인덱스
        route_indices = [121, 11, 85, 93, 57, 43, 118, 79, 104, 95]
        
        # 에고 차량 스폰
        start_transform = spawn_points[route_indices[0]]
        ego_vehicle = world.try_spawn_actor(ego_bp, start_transform)
        
        if ego_vehicle is None:
            print(f"Failed to spawn ego vehicle. Exiting.")
            return

        print(f"Ego vehicle spawned at: {ego_vehicle.get_transform().location}")
        
        # 카메라 센서 블루프린트 설정 및 부착
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '100')
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        camera_sensor.listen(lambda image: process_image(image, display))
        
        # 경로 생성
        route = []
        for ind in route_indices:
            if ind < len(spawn_points):
                route.append(spawn_points[ind].location)
        
        # 에고 차량에 경로 설정
        traffic_manager.set_path(ego_vehicle, route)
        
        # 차량 속도 설정 (40km/h) 및 오토파일럿 활성화
        target_speed = 40.0
        # 속도 차이를 퍼센트로 설정. -100%는 최대 속도, 0%는 기본 속도
        traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, - (target_speed / 100.0))
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        
        # ------------------- NPC 차량 소환 -------------------
        print("\nSpawning 10 NPC vehicles...")
        for i in range(10):
            npc_bp = random.choice(blueprint_library.filter('vehicle')) # 무작위 차량 모델 선택
            npc_bp.set_attribute('role_name', f'npc_vehicle_{i}')
            
            spawn_point = random.choice(spawn_points) # 무작위 스폰 포인트 선택
            npc_vehicle = world.try_spawn_actor(npc_bp, spawn_point)
            
            if npc_vehicle is not None:
                npc_vehicles.append(npc_vehicle)
                print(f"NPC vehicle {i+1} spawned.")
        
        # 모든 NPC 차량에 오토파일럿 설정
        for npc in npc_vehicles:
            npc.set_autopilot(True, traffic_manager.get_port())
            
        print("Ego vehicle and NPC vehicles spawned. Running...")
        
        last_waypoint_location = route[-1]
        prev_location = ego_vehicle.get_location()
        # ------------------- 메인 시뮬레이션 루프 -------------------
        while True:
            # 시뮬레이션의 다음 프레임으로 이동
            world.tick()
            
            # ego 차량 경로 그리기
            current_location = ego_vehicle.get_location()
            if prev_location:
                world.debug.draw_line(
                    prev_location,
                    current_location,
                    thickness=1.0,
                    color=carla.Color(r=0, g=0, b=255, a=255), # 파란색
                    life_time=9999999.0
                )
            prev_location = current_location

            # 모든 NPC 차량의 3D 바운딩 박스를 그림
            for npc in npc_vehicles:
                # 차량의 바운딩 박스 정보(로컬 좌표) 가져오기
                bbox = npc.bounding_box
                # 차량의 현재 위치 및 회전 정보 가져오기
                transform = npc.get_transform()
                
                # 바운딩 박스 그리기
                # 1. BoundingBox 객체 생성: 월드 좌표계의 위치와 크기
                #    - 위치: 차량의 월드 위치 + 바운딩 박스 중심의 로컬 위치
                #    - 크기: 바운딩 박스의 크기
                # 2. 회전: 차량의 회전 정보
                # 3. 두께: 선의 두께
                # 4. 색상: RGBa 색상
                # 5. 지속 시간: 다음 틱까지 유지 (갱신을 위해)
                world.debug.draw_box(
                    carla.BoundingBox(transform.location + bbox.location, bbox.extent),
                    transform.rotation,
                    0.1, # 두께
                    carla.Color(255, 0, 0, 255), # 빨간색
                    life_time=world.get_settings().fixed_delta_seconds * 1.2 
                )
            
            # Pygame 화면 업데이트
            pygame.display.flip()
            
            # 에고 차량이 경로의 끝에 도착했는지 확인
            distance_to_end = ego_vehicle.get_location().distance(last_waypoint_location)
            if distance_to_end < 2.0:
                # 경로가 끝나면 다시 처음부터 경로를 반복
                traffic_manager.set_path(ego_vehicle, route)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

    finally:
        # 시뮬레이션 종료 시 리소스 정리
        print("\nCleaning up actors...")
        if original_settings is not None:
            world.apply_settings(original_settings)
        if camera_sensor is not None:
            camera_sensor.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        
        # 모든 NPC 차량을 일괄 제거
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicles])
        
        pygame.quit()
        print("Simulation finished.")

if __name__ == '__main__':
    # 스크립트가 직접 실행될 때 main 함수 호출
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

# import carla
# import random
# import time
# import pygame
# import numpy as np

# # Pygame 윈도우 크기
# WINDOW_WIDTH = 800
# WINDOW_HEIGHT = 600

# def pygame_init():
#     pygame.init()
#     return pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

# def process_image(image, display):
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]
    
#     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
#     display.blit(surface, (0, 0))
#     pygame.display.flip()

# def main():
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(10.0)
#     world = client.get_world()

#     ego_vehicle = None
#     camera_sensor = None
#     original_settings = None

#     try:
#         display = pygame_init()
#         original_settings = world.get_settings()
        
#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.05
#         world.apply_settings(settings)

#         traffic_manager = client.get_trafficmanager(8000)
#         traffic_manager.set_synchronous_mode(True)

#         blueprint_library = world.get_blueprint_library()
#         ego_bp = blueprint_library.find('vehicle.lincoln.mkz')
#         ego_bp.set_attribute('role_name', 'ego_vehicle')

#         spawn_points = world.get_map().get_spawn_points()
        
#         #경로
#         route_indices = [121, 11, 85, 93, 57, 43, 118, 79, 104, 95]
#         route_indices_0 = [121, 11, 85, 91, 111, 141]
        
#         if not route_indices or route_indices[0] >= len(spawn_points):
#             print("Invalid route indices. Exiting.")
#             return

#         start_transform = spawn_points[route_indices[0]]
#         ego_vehicle = world.try_spawn_actor(ego_bp, start_transform)
        
#         if ego_vehicle is None:
#             print(f"Failed to spawn ego vehicle at spawn point {route_indices[0]}. Exiting.")
#             return

#         print(f"Ego vehicle spawned at spawn point {route_indices[0]}: {ego_vehicle.get_transform().location}")
        
#         camera_bp = blueprint_library.find('sensor.camera.rgb')
#         camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
#         camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
#         camera_bp.set_attribute('fov', '100')
#         camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
#         camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
#         camera_sensor.listen(lambda image: process_image(image, display))
        
#         route = []
#         print("\nGenerated Route Waypoints:")
#         for i, ind in enumerate(route_indices):
#             if ind < len(spawn_points):
#                 waypoint_transform = spawn_points[ind]
#                 route.append(waypoint_transform.location)
#                 print(f"Waypoint {i+1}: Location(x={waypoint_transform.location.x}, y={waypoint_transform.location.y}, z={waypoint_transform.location.z})")
#             else:
#                 print(f"Invalid spawn point index {ind}. Skipping.")
        
#         traffic_manager.set_path(ego_vehicle, route)
        
#         # 차량 속도 설정 (40km/h)
#         target_speed = 40.0
#         traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, - (target_speed / 100.0))
#         ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        
#         # 경로 시각화
#         for i, loc in enumerate(route):
#             world.debug.draw_string(loc, str(i + 1), life_time=999999.0, persistent_lines=True)
            
#         print("Ego vehicle spawned and route set. Running...")
        
#         # 경로의 마지막 웨이포인트
#         last_waypoint_location = route[-1]
        
#         while True:
#             world.tick()
            
#             # --- 경로가 끝나면 새로운 목표를 설정하는 코드 위치 ---
#             # 차량이 마지막 웨이포인트에 근접했는지 확인
#             distance_to_end = ego_vehicle.get_location().distance(last_waypoint_location)
#             if distance_to_end < 2.0: # 2m 이내에 도착하면 새로운 목표 설정                
#                 # Traffic Manager에 새로운 경로 설정
#                 traffic_manager.set_path(ego_vehicle, route)

#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     return

#     finally:
#         if original_settings is not None:
#             world.apply_settings(original_settings)
#         if camera_sensor is not None:
#             camera_sensor.destroy()
#         if ego_vehicle is not None:
#             ego_vehicle.destroy()
        
#         pygame.quit()
#         print("Simulation finished.")

# if __name__ == '__main__':
#     try:
#         main()
#     except Exception as e:
#         print(f"An error occurred: {e}")