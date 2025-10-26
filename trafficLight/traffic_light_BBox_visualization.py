import math
import random
import time
import queue
import numpy as np
import cv2
import glob
import sys
import os

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate
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
        # Ensure synchronous mode is enabled
        settings = world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            print("Synchronous mode enabled (20 FPS)")

        # Find ego vehicle with role_name='hero'
        vehicle = None
        for actor in world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                vehicle = actor
                break
        if vehicle is None:
            raise RuntimeError("Ego vehicle (role_name='hero') not found. Ensure manual_control.py is running.")

        # Spawn camera attached to ego vehicle
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(z=2))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

        # Create a queue to store sensor data
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # Get camera attributes
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        # Calculate camera projection matrix
        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        # Set up bounding boxes for traffic lights
        bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

        # Edge pairs for bounding box
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        # Retrieve and display first image
        world.tick()
        try:
            image = image_queue.get(timeout=1.0)
        except queue.Empty:
            raise RuntimeError("Failed to receive initial camera data.")
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('ImageWindowName', img)
        cv2.waitKey(1)

        while True:
            # Retrieve and reshape image
            world.tick()
            try:
                image = image_queue.get(timeout=1.0)
            except queue.Empty:
                print("Failed to receive camera data.")
                continue

            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            # Get camera matrix
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Get all traffic light actors
            traffic_lights = world.get_actors().filter('traffic.traffic_light')

            for bb in bounding_box_set:
                # Filter for distance from ego vehicle
                if bb.location.distance(vehicle.get_transform().location) < 50:
                    # Filter for objects in front of the camera
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = bb.location - vehicle.get_transform().location
                    if forward_vec.dot(ray) > 0:
                        # Find the closest traffic light actor to this bounding box
                        closest_tl = None
                        min_dist = float('inf')
                        for tl in traffic_lights:
                            dist = tl.get_location().distance(bb.location)
                            if dist < min_dist:
                                closest_tl = tl
                                min_dist = dist

                        if closest_tl is not None:
                            # Draw traffic light forward vector (+X) in 3D world
                            tl_forward = closest_tl.get_transform().get_forward_vector()
                            tl_forward_2d = carla.Vector3D(tl_forward.x, tl_forward.y, 0.0)  # X-Y plane projection
                            if tl_forward_2d.length() > 0:
                                tl_forward_2d = tl_forward_2d.make_unit_vector()  # Normalize to unit vector
                            else:
                                tl_forward_2d = carla.Vector3D(1.0, 0.0, 0.0)
                            tl_location = closest_tl.get_location()
                            tl_end = carla.Location(
                                x=tl_location.x + 5.0 * tl_forward_2d.x,
                                y=tl_location.y + 5.0 * tl_forward_2d.y,
                                z=tl_location.z
                            )
                            world.debug.draw_line(
                                tl_location,
                                tl_end,
                                thickness=0.1,
                                color=carla.Color(0, 0, 255, 255),  # Blue
                                life_time=0.05  # Match synchronous mode frame time
                            )

                        # Draw bounding box edges in camera image
                        verts = [v for v in bb.get_world_vertices(carla.Transform())]
                        for edge in edges:
                            p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                            p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255, 255), 1)

            # Display image
            cv2.imshow('ImageWindowName', img)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # Clean up
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if 'camera' in locals():
            camera.destroy()
        cv2.destroyAllWindows()
        print("Cleaning up...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Exiting...')
    except Exception as e:
        print(f"An error occurred: {e}")