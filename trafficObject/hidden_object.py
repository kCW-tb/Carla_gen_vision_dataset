import carla
import numpy as np
import cv2
import time

IM_WIDTH = 1280
IM_HEIGHT = 720
VISIBLE_THRESHOLD = 0.05  # 5% 이하이면 "가려짐"

def get_camera_transform():
    return carla.Transform(carla.Location(x=1.6, z=1.7))  # 운전자 시점

def semantic_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))  # BGRA
    data_dict['semantic_image'] = array[:, :, 2]  # R 채널 = class ID

def project_to_image(world, cam, bbox, vehicle_tf, intrinsic_matrix):
    corners = [carla.Location(x=bbox.extent.x * sx,
                               y=bbox.extent.y * sy,
                               z=bbox.extent.z * sz)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]

    points_2d = []
    for corner in corners:
        world_loc = vehicle_tf.transform(corner + bbox.location)
        cam_loc = cam.get_transform().inverse().transform(world_loc)
        if cam_loc.z <= 0.1:
            continue
        point = intrinsic_matrix @ np.array([[cam_loc.x], [cam_loc.y], [cam_loc.z]])
        point = point / point[2]
        points_2d.append((int(point[0]), int(point[1])))

    if not points_2d:
        return None

    xs, ys = zip(*points_2d)
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = min(max(xs), IM_WIDTH - 1), min(max(ys), IM_HEIGHT - 1)
    return (x1, y1, x2, y2)

def get_visibility_ratio(mask, box, class_id=10):
    x1, y1, x2, y2 = box
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    total_pixels = crop.size
    visible_pixels = np.count_nonzero(crop == class_id)
    return visible_pixels / total_pixels

def draw_bbox(image, box, ratio):
    color = (0, 255, 0) if ratio >= VISIBLE_THRESHOLD else (0, 0, 255)
    label = f"{ratio * 100:.1f}%"
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Ego vehicle 탐색
    vehicles = world.get_actors().filter("vehicle.*")
    ego_vehicle = next((v for v in vehicles if v.attributes.get("role_name") == "hero"), None)
    if ego_vehicle is None:
        print("Ego vehicle with role_name='hero' not found.")
        return

    # 1. 세그멘테이션 카메라 blueprint 생성
    sem_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(IM_WIDTH))
    sem_bp.set_attribute("image_size_y", str(IM_HEIGHT))
    sem_bp.set_attribute("sensor_tick", "0.1")

    # 2. 카메라 transform (운전자 시점 위치)
    cam_transform = get_camera_transform()

    # 3. 카메라 actor 생성 및 ego 차량에 부착
    cam = world.spawn_actor(sem_bp, cam_transform, attach_to=ego_vehicle)

    # 4. 카메라로부터 이미지 데이터를 수신하도록 설정
    data = {"semantic_image": None}
    cam.listen(lambda image: semantic_callback(image, data))

    print("Running occlusion detector... Press ESC to exit.")

    while True:
        if data["semantic_image"] is None:
            time.sleep(0.1)
            continue

        seg_img = data["semantic_image"]
        vis_img = np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)

        for vehicle in vehicles:
            if vehicle.id == ego_vehicle.id:
                continue

            box = vehicle.bounding_box
            vehicle_tf = vehicle.get_transform()
            box_2d = project_to_image(world, cam, box, vehicle_tf, K)
            if not box_2d:
                continue

            vis_ratio = get_visibility_ratio(seg_img, box_2d, class_id=10)
            draw_bbox(vis_img, box_2d, vis_ratio)

        cv2.imshow("Occlusion Viewer", vis_img)
        if cv2.waitKey(1) == 27:
            break

    cam.stop()
    cam.destroy()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
