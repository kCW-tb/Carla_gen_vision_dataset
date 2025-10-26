import carla
import time
import math

def main():
    # CARLA 클라이언트 초기화
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        # 디버그 도구
        debug = world.debug

        # 모든 신호등 액터 가져오기
        traffic_lights = world.get_actors().filter('traffic.traffic_light')

        # 각 신호등에 대해 기저 벡터 표시
        for traffic_light in traffic_lights:
            # 신호등의 변환 정보(위치와 회전) 가져오기
            transform = traffic_light.get_transform()
            location = transform.location
            rotation = transform.rotation

            # 시작점 높이 조정 (z축 +2미터)
            location.z += 2.0

            # 기저 벡터 계산
            forward_vector = rotation.get_forward_vector()  # 로컬 X축 (정면)
            right_vector = rotation.get_right_vector()      # 로컬 Y축 (오른쪽)
            up_vector = rotation.get_up_vector()           # 로컬 Z축 (위쪽)

            # 벡터 길이 설정
            vector_length = 3.0

            # 정면 벡터 (초록색)
            forward_end = carla.Location(
                x=location.x + forward_vector.x * vector_length,
                y=location.y + forward_vector.y * vector_length,
                z=location.z + forward_vector.z * vector_length
            )
            debug.draw_arrow(
                location,
                forward_end,
                thickness=0.1,
                arrow_size=0.3,
                color=carla.Color(r=0, g=255, b=0),  # 초록색
                life_time=0.0  # 영구 표시
            )
            # 정면 벡터 값 및 기준 표시
            forward_text = f"Forward (Local X): [{forward_vector.x:.2f}, {forward_vector.y:.2f}, {forward_vector.z:.2f}]"
            debug.draw_string(
                carla.Location(x=forward_end.x, y=forward_end.y, z=forward_end.z + 0.5),
                forward_text,
                draw_shadow=True,
                color=carla.Color(r=0, g=255, b=0),
                life_time=0.0,
                persistent_lines=True
            )

            # 오른쪽 벡터 (빨강색)
            right_end = carla.Location(
                x=location.x + right_vector.x * vector_length,
                y=location.y + right_vector.y * vector_length,
                z=location.z + right_vector.z * vector_length
            )
            debug.draw_arrow(
                location,
                right_end,
                thickness=0.1,
                arrow_size=0.3,
                color=carla.Color(r=255, g=0, b=0),  # 빨강색
                life_time=0.0
            )
            # 오른쪽 벡터 값 및 기준 표시
            right_text = f"Right (Local Y): [{right_vector.x:.2f}, {right_vector.y:.2f}, {right_vector.z:.2f}]"
            debug.draw_string(
                carla.Location(x=right_end.x, y=right_end.y, z=right_end.z + 0.5),
                right_text,
                draw_shadow=True,
                color=carla.Color(r=255, g=0, b=0),
                life_time=0.0,
                persistent_lines=True
            )

            # 위쪽 벡터 (파랑색)
            up_end = carla.Location(
                x=location.x + up_vector.x * vector_length,
                y=location.y + up_vector.y * vector_length,
                z=location.z + up_vector.z * vector_length
            )
            debug.draw_arrow(
                location,
                up_end,
                thickness=0.1,
                arrow_size=0.3,
                color=carla.Color(r=0, g=0, b=255),  # 파랑색
                life_time=0.0
            )
            # 위쪽 벡터 값 및 기준 표시
            up_text = f"Up (Local Z): [{up_vector.x:.2f}, {up_vector.y:.2f}, {up_vector.z:.2f}]"
            debug.draw_string(
                carla.Location(x=up_end.x, y=up_end.y, z=up_end.z + 0.5),
                up_text,
                draw_shadow=True,
                color=carla.Color(r=0, g=0, b=255),
                life_time=0.0,
                persistent_lines=True
            )

        # 시뮬레이션 유지
        print("신호등 기저 벡터 표시 완료. 시뮬레이터를 확인하세요.")
        while True:
            world.wait_for_tick()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n프로그램 종료.")
    finally:
        print("완료.")

if __name__ == '__main__':
    main()