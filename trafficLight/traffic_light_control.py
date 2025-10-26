import random
import time
import queue
import os
import cv2
import numpy as np
import math
import glob
import sys
import json
from pathlib import Path

import carla

# CARLA 파이썬 API 경로 추가
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


def get_ego_vehicle(world):
    """Ego 차량을 찾습니다 (role_name == 'hero')"""
    for vehicle in world.get_actors().filter('vehicle.*'):
        if vehicle.attributes.get('role_name') == 'hero':
            return vehicle
    raise RuntimeError("Ego 차량 (role_name=hero)을 찾을 수 없습니다.")


def find_forward_traffic_light(world, ego_vehicle, max_distance=30.0):
    """
    ego 차량 전방에 있는 차량용 신호등 액터를 탐색합니다.
    max_distance 미터 이내에 위치한 정면 신호등만 반환합니다.
    """
    ego_location = ego_vehicle.get_location()
    ego_forward = ego_vehicle.get_transform().get_forward_vector()

    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for tl in traffic_lights:
        tl_location = tl.get_location()
        direction_vec = tl_location - ego_location

        # 거리 체크
        if direction_vec.length() > max_distance:
            continue

        # 시야 내 (정면) 체크 - dot product
        direction_vec = direction_vec.make_unit_vector()
        dot = ego_forward.x * direction_vec.x + ego_forward.y * direction_vec.y

        if dot > 0.7:  # 약 45도 이내 정면
            return tl

    return None


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()

    ego_vehicle = get_ego_vehicle(world)
    print("✅ Ego 차량 탐색 완료.")

    target_tl = find_forward_traffic_light(world, ego_vehicle)

    if target_tl is None:
        print("❌ 전방 신호등을 찾지 못했습니다.")
        return

    print("✅ 전방 차량용 신호등을 발견했습니다.")

    # 자동 제어 비활성화
    target_tl.set_simulate_physics(False)
    target_tl.set_autopilot(False)
    target_tl.freeze(True)  # 시간 흐름 고정

    desired_state = TrafficLightState.Red  # 또는 .Green, .Yellow
    target_tl.set_state(desired_state)
    print(f"🔧 신호등 상태를 {desired_state} 로 설정했습니다.")

    # 유지 시간 설정 (옵션)
    target_tl.set_green_time(10.0)
    target_tl.set_red_time(10.0)
    target_tl.set_yellow_time(4.0)

    print("🔁 신호등 수동 제어 완료")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("사용자에 의해 종료됨.")
