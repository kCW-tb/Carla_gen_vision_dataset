# import carla
# import matplotlib.pyplot as plt

# # ======================
# # 1. CARLA 서버 연결
# # ======================
# client = carla.Client('localhost', 2000)
# client.set_timeout(5.0)
# world = client.get_world()
# town_map = world.get_map()
# spawn_points = town_map.get_spawn_points()

# # ======================
# # 2. Spawn Point 좌표 추출
# # ======================
# x_vals = [sp.location.x for sp in spawn_points]
# y_vals = [sp.location.y for sp in spawn_points]

# # ======================
# # 3. 시각화
# # ======================
# plt.figure(figsize=(12, 8))
# plt.scatter(x_vals, y_vals, c='blue', s=30, label='Spawn Points')

# # 번호 붙이기
# for i, sp in enumerate(spawn_points):
#     plt.text(sp.location.x, sp.location.y, str(i), fontsize=9, color='red')

# plt.title(f"Spawn Points in {world.get_map().name}")
# plt.xlabel("X coordinate")
# plt.ylabel("Y coordinate")
# plt.legend()
# plt.grid(True)
# plt.show()


import carla
import time

def draw_spawn_points():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()

    for i, transform in enumerate(spawn_points):
        world.debug.draw_string(
            transform.location,
            str(i), 
            draw_shadow=False,
            color=carla.Color(r=0, g=255, b=0),
            life_time=1500.0,
            persistent_lines=True
        )
        print(f"Spawn Point {i}: Location(x={transform.location.x}, y={transform.location.y}, z={transform.location.z})")

if __name__ == '__main__':
    try:
        draw_spawn_points()
        print("Press Ctrl+C to exit.")
        time.sleep(1500)
    except KeyboardInterrupt:
        print("Cancelled by user.")