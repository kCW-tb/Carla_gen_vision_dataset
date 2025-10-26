import time
import queue
import cv2
import numpy as np
import math
import json
from pathlib import Path

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
    point_camera = (point_camera[1], -point_camera[2], point_camera[0])
    point_img = np.dot(K, point_camera)
    if point_img[2] <= 0:  # Exclude points behind the camera
        return None
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def load_existing_dataset(annotations_dir):
    """Load existing annotations.json and return dataset with last image_id and annotation_id."""
    json_path = annotations_dir / "annotations.json"
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Red"},
            {"id": 2, "name": "Green"},
            {"id": 3, "name": "Yellow"}
        ]
    }
    max_image_id = -1
    max_annotation_id = -1

    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
            # Validate required fields
            if not isinstance(existing_data, dict) or "images" not in existing_data or "annotations" not in existing_data:
                print("Invalid annotations.json format. Starting with a new dataset.")
                return coco_dataset, 0, 0
            
            coco_dataset["images"] = existing_data.get("images", [])
            coco_dataset["annotations"] = existing_data.get("annotations", [])
            coco_dataset["categories"] = existing_data.get("categories", coco_dataset["categories"])
            
            if coco_dataset["images"]:
                max_image_id = max(img.get("id", -1) for img in coco_dataset["images"])
            if coco_dataset["annotations"]:
                max_annotation_id = max(ann.get("id", -1) for ann in coco_dataset["annotations"])
            
            print(f"Loaded existing dataset: {len(coco_dataset['images'])} images, {len(coco_dataset['annotations'])} annotations")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to load annotations.json: {e}. Starting with a new dataset.")
    else:
        print("No existing annotations.json found. Starting with a new dataset.")
    
    return coco_dataset, max_image_id + 1, max_annotation_id + 1

def save_dataset(coco_dataset, annotations_dir):
    """Save coco_dataset to annotations.json."""
    json_path = annotations_dir / "annotations.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(coco_dataset, f, indent=4)
        print(f"Saved dataset to {json_path}")
    except Exception as e:
        print(f"Failed to save annotations.json: {e}")

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    print(f"CARLA Client API version: {client.get_client_version()}")
    print(f"CARLA Server version: {client.get_server_version()}")
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    dataset_dir = Path("./dataset")
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Load existing dataset
    coco_dataset, image_id, annotation_id = load_existing_dataset(annotations_dir)
    print(f"Starting with image_id: {image_id}, annotation_id: {annotation_id}")

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.03  # 20 FPS
        world.apply_settings(settings)
        print("Synchronous mode enabled (20 FPS)")

        vehicle = None
        for actor in world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                vehicle = actor
                break
        if vehicle is None:
            raise RuntimeError("Ego vehicle (role_name='hero') not found.")

        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_init_trans = carla.Transform(carla.Location(z=2))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        K = build_projection_matrix(image_w, image_h, fov)

        last_save_time = time.time()

        print("Calling initial world.tick()")
        world.tick()
        try:
            image = image_queue.get(timeout=1.0)
            print("Initial camera data received successfully")
        except queue.Empty:
            raise RuntimeError("Failed to receive initial camera data.")

        while True:
            print("Calling world.tick()")
            world.tick()
            try:
                image = image_queue.get(timeout=1.0)
                print("Camera data received successfully")
            except queue.Empty:
                print("Failed to receive camera data")
                continue

            img_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
            traffic_lights = world.get_actors().filter("traffic.traffic_light*")

            frame_annotations = []
            has_valid_class = False

            for bb in bounding_box_set:
                if bb.location.z < 3.0:
                    continue
                if bb.location.distance(vehicle.get_transform().location) > 60:
                    continue

                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 1:
                    continue

                closest_tl = None
                min_dist = float('inf')
                for tl in traffic_lights:
                    dist = tl.get_location().distance(bb.location)
                    if dist < min_dist:
                        closest_tl = tl
                        min_dist = dist

                if closest_tl is None:
                    continue

                cam_transform = camera.get_transform()
                camera_forward = cam_transform.get_forward_vector()
                tl_y_plus = closest_tl.get_transform().rotation.get_right_vector()

                cam_forward_xy = carla.Vector3D(camera_forward.x, camera_forward.y, 0)
                tl_y_plus_xy = carla.Vector3D(tl_y_plus.x, tl_y_plus.y, 0)

                cam_forward_xy = cam_forward_xy.make_unit_vector()
                tl_y_plus_xy = tl_y_plus_xy.make_unit_vector()

                cos_angle = cam_forward_xy.dot(tl_y_plus_xy)
                angle_deg = math.degrees(math.acos(max(min(cos_angle, 1.0), -1.0)))
                cross_z = cam_forward_xy.x * tl_y_plus_xy.y - cam_forward_xy.y * tl_y_plus_xy.x
                if cross_z < 0:
                    angle_deg = 360 - angle_deg

                category_id = None
                if 110 <= angle_deg <= 250:
                    state = closest_tl.get_state()
                    print(f"Traffic light state: {state}, Angle: {angle_deg:.2f}")
                    if state == carla.TrafficLightState.Red:
                        category_id = 1
                        has_valid_class = True
                    elif state == carla.TrafficLightState.Green:
                        category_id = 2
                        has_valid_class = True
                    elif state == carla.TrafficLightState.Yellow:
                        category_id = 3
                        has_valid_class = True

                if category_id is None:
                    continue

                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                points = [p for p in [get_image_point(v, K, world_2_camera) for v in verts] if p is not None]
                if not points:
                    continue
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = max(0, min(x_coords)), min(image_w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(image_h, max(y_coords))
                width = x_max - x_min
                height = y_max - y_min

                if width > 0 and height > 0:
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(x_min), float(y_min), float(width), float(height)],
                        "area": float(width * height),
                        "iscrowd": 0
                    }
                    frame_annotations.append(annotation)
                    annotation_id += 1

            current_time = time.time()
            if has_valid_class and current_time - last_save_time >= 1.0:
                image_filename = f"image_{image_id:06d}.png"
                image_path = images_dir / image_filename
                try:
                    cv2.imwrite(str(image_path), img_bgr)
                    print(f"Saved image: {image_path}")
                except Exception as e:
                    print(f"Failed to save image {image_path}: {e}")
                    continue

                coco_dataset["images"].append({
                    "id": image_id,
                    "file_name": image_filename,
                    "width": image_w,
                    "height": image_h
                })
                coco_dataset["annotations"].extend(frame_annotations)

                save_dataset(coco_dataset, annotations_dir)

                print(f"Dataset updated: {image_filename}, {len(frame_annotations)} new annotations")
                image_id += 1
                last_save_time = current_time

    finally:
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("Synchronous mode disabled")
        if 'camera' in locals():
            camera.stop()
            camera.destroy()
            print("Camera resources cleaned up")
        print("All resources cleaned up")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTerminated by user')
    except Exception as e:
        print(f"Error occurred: {e}")