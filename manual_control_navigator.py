import carla
import random
import time
import pygame
import numpy as np

# Pygame 윈도우 크기
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

def pygame_init():
    pygame.init()
    return pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

def process_image(image, display):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 변수를 try 블록 밖에서 미리 초기화
    ego_vehicle = None
    camera_sensor = None
    original_settings = None

    try:
        display = pygame_init()
        original_settings = world.get_settings()
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.find('vehicle.lincoln.mkz')
        ego_bp.set_attribute('role_name', 'ego_vehicle')

        spawn_points = world.get_map().get_spawn_points()
        
        print("\nVisualizing all spawn points for 10 minutes.")
        for i, transform in enumerate(spawn_points):
            world.debug.draw_string(
                transform.location,
                str(i),
                draw_shadow=True,
                color=carla.Color(r=0, g=255, b=0),
                life_time=600.0,
                persistent_lines=True,
                font_size=0.3
            )
            
        spawn_point = random.choice(spawn_points)
        ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point)
        
        if ego_vehicle is None:
            print("Ego vehicle spawning failed.")
            return

        print(f"Ego vehicle spawned at: {ego_vehicle.get_transform().location}")
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '100')
        
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        
        camera_sensor.listen(lambda image: process_image(image, display))

        route_indices = [86, 4, 126, 139, 102, 34, 35]
        route = []
        print("\nGenerated Route Waypoints:")
        for i, ind in enumerate(route_indices):
            waypoint = spawn_points[ind]
            route.append(waypoint.location)
            print(f"Waypoint {i+1}: Location(x={waypoint.location.x}, y={waypoint.location.y}, z={waypoint.location.z})")

        traffic_manager.set_path(ego_vehicle, route)
        target_speed = 40.0
        traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, - (target_speed / 100.0))
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        
        while True:
            world.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

    finally:
        if original_settings is not None:
            world.apply_settings(original_settings)
        if camera_sensor is not None:
            camera_sensor.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        
        pygame.quit()
        print("Simulation finished.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")