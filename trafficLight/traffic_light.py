import random
import time
import queue
import os
import cv2
import numpy as np
import math
import glob
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

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
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    try:
        # 동기 모드 설정
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.0333
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

        # 박스 엣지 정의
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        world.tick()
        try:
            image = image_queue.get(timeout=1.0)
        except queue.Empty:
            raise RuntimeError("초기 카메라 데이터를 수신하지 못했습니다.")
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('ImageWindowName', img)
        cv2.waitKey(1)

        while True:
            world.tick()
            try:
                image = image_queue.get(timeout=1.0)
            except queue.Empty:
                print("카메라 데이터 수신 실패")
                continue

            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # 신호등 바운딩 박스 + 상태 액터 가져옴
            bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            traffic_lights = world.get_actors().filter("traffic.traffic_light*")

            box_counter = 1
            for bb in bounding_box_set:
                # 보행자 신호등 필터링
                if bb.location.z < 3.0:
                    continue

                if bb.location.distance(vehicle.get_transform().location) > 60:
                    continue

                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 1:
                    continue

                # 해당 BBox와 가장 가까운 traffic_light 액터 찾기
                closest_tl = None
                min_dist = float('inf')
                for tl in traffic_lights:
                    dist = tl.get_location().distance(bb.location)
                    if dist < min_dist:
                        closest_tl = tl
                        min_dist = dist

                # 기본 색상: BLACK (보이지 않음)
                color = (0, 0, 0)

                angle_deg = None
                if closest_tl is not None:
                    cam_transform = camera.get_transform()
                    camera_forward = cam_transform.get_forward_vector()
                    # 신호등의 로컬 Y+ 방향 벡터
                    tl_y_plus = closest_tl.get_transform().rotation.get_right_vector()

                    # XY 평면에 투영 (Z 성분 제거)
                    cam_forward_xy = carla.Vector3D(camera_forward.x, camera_forward.y, 0)
                    tl_y_plus_xy = carla.Vector3D(tl_y_plus.x, tl_y_plus.y, 0)

                    # 단위 벡터로 정규화
                    cam_forward_xy = cam_forward_xy.make_unit_vector()
                    tl_y_plus_xy = tl_y_plus_xy.make_unit_vector()

                    # 내적을 사용해 코사인 각도 계산
                    cos_angle = cam_forward_xy.dot(tl_y_plus_xy)
                    # 각도를 도 단위로 변환 (0~180도)
                    angle_deg = math.degrees(math.acos(max(min(cos_angle, 1.0), -1.0)))
                    # 180도 이상인 경우 보정 (0~360도 범위)
                    cross_z = cam_forward_xy.x * tl_y_plus_xy.y - cam_forward_xy.y * tl_y_plus_xy.x
                    if cross_z < 0:
                        angle_deg = 360 - angle_deg

                    # XY 평면 사이각이 110도~250도 사이인지 확인
                    if 90 <= angle_deg <= 270:
                        # 신호등 상태에 따른 색상
                        state = closest_tl.get_state()
                        if state == carla.TrafficLightState.Red:
                            color = (0, 0, 255)
                        elif state == carla.TrafficLightState.Green:
                            color = (0, 255, 0)
                        elif state == carla.TrafficLightState.Yellow:
                            color = (0, 255, 255)
                        else:
                            color = (0, 0, 0)
                    else:
                        color = (0, 0, 0)  # BLACK

                # 바운딩 박스 그리기
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                top_vertex = min(verts, key=lambda v: get_image_point(v, K, world_2_camera)[1])
                top_point = get_image_point(top_vertex, K, world_2_camera)
                cv2.putText(img, str(box_counter), (int(top_point[0]), int(top_point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # 바운딩 박스 하단에 XY 평면 사이각 표시
                bottom_vertex = max(verts, key=lambda v: get_image_point(v, K, world_2_camera)[1])
                bottom_point = get_image_point(bottom_vertex, K, world_2_camera)
                if angle_deg is not None:
                    angle_text = f"angle: {angle_deg:.1f}°"
                    cv2.putText(img, angle_text, (int(bottom_point[0]), int(bottom_point[1]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                for edge in edges:
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
                box_counter += 1

            cv2.imshow('ImageWindowName', img)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if 'camera' in locals():
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