import random
import time
import keyboard  # pip install keyboard
import carla

def set_all_traffic_lights_state(world, state):
    """모든 신호등의 상태를 지정된 상태로 설정"""
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    count = 0
    for tl in traffic_lights:
        tl.set_state(state)  # 신호등 상태만 변경
        count += 1
    print(f"🔧 {count}개의 신호등 상태를 {state}로 설정했습니다.")

def teleport_vehicle(world):
    """차량을 가장 가까운 도로 웨이포인트로 순간이동"""
    # 플레이어 차량 가져오기 (첫 번째 차량을 대상으로 함)
    vehicles = world.get_actors().filter('vehicle.*')
    if not vehicles:
        print("⚠️ 차량이 없습니다!")
        return
    
    vehicle = vehicles[0]  # 첫 번째 차량 (플레이어 차량)
    location = vehicle.get_location()
    
    # 가장 가까운 웨이포인트 찾기
    waypoint = world.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if not waypoint:
        print("⚠️ 적합한 웨이포인트를 찾을 수 없습니다!")
        return
    
    # 랜덤으로 근처 웨이포인트 선택 (2미터 이내)
    next_waypoints = waypoint.next(2.0)
    if not next_waypoints:
        print("⚠️ 근처 웨이포인트를 찾을 수 없습니다!")
        return
    
    target_waypoint = random.choice(next_waypoints)
    
    # 물리 비활성화 (충돌 방지)
    vehicle.set_simulate_physics(False)
    
    # 차량을 웨이포인트로 순간이동
    vehicle.set_transform(target_waypoint.transform)
    
    # 물리 재활성화
    vehicle.set_simulate_physics(True)
    print(f"🚗 차량을 위치 ({target_waypoint.transform.location.x:.2f}, {target_waypoint.transform.location.y:.2f}, {target_waypoint.transform.location.z:.2f})로 순간이동했습니다.")

def main():
    try:
        client = carla.Client('localhost', 2000)  # manual_control.py와 다른 포트
        client.set_timeout(5.0)
        print(f"CARLA Client API 버전: {client.get_client_version()}")
        print(f"CARLA Server 버전: {client.get_server_version()}")

        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False  # 비동기 모드
        world.apply_settings(settings)
        print("비동기 모드 설정 완료")

        print("키보드 입력 대기: 'r' (빨간색), 'y' (노란색), 'g' (녹색), 'e' (차량 순간이동), 'q' (종료)")
        while True:
            if keyboard.is_pressed('r'):
                set_all_traffic_lights_state(world, carla.TrafficLightState.Red)
                time.sleep(0.2)  # 키 입력 반복 방지
            elif keyboard.is_pressed('y'):
                set_all_traffic_lights_state(world, carla.TrafficLightState.Yellow)
                time.sleep(0.2)
            elif keyboard.is_pressed('g'):
                set_all_traffic_lights_state(world, carla.TrafficLightState.Green)
                time.sleep(0.2)
            elif keyboard.is_pressed('e'):
                teleport_vehicle(world)
                time.sleep(0.2)  # 키 입력 반복 방지
            elif keyboard.is_pressed('q'):
                print("종료 키 입력됨")
                break
            time.sleep(0.01)  # CPU 부하 감소

    finally:
        # 리소스 정리
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("리소스 정리 완료")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("사용자에 의해 종료됨.")
    except Exception as e:
        print(f"에러 발생: {e}")