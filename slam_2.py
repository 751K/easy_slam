import numpy as np
from matplotlib import pyplot as plt

from particle_filter import ParticleFilter
from robot import Robot

# 静态地标数量与地图大小
num_static_landmarks = 50
map_size = 160
steps = 50

# 生成静态地标
landmark_xy = map_size * (np.random.rand(num_static_landmarks, 2) - 0.5)
landmark_id = np.linspace(0, num_static_landmarks - 1, num_static_landmarks, dtype='uint16').reshape(-1, 1)
static_landmarks = np.hstack((landmark_xy, landmark_id))

# 动态地标数量
num_dynamic_landmarks = 10
# 速度乘数
vm = 20
landmark_xy = map_size * (np.random.rand(num_dynamic_landmarks, 2) - 0.5)
landmark_v = np.random.rand(num_dynamic_landmarks, 2) - 0.5
landmark_id = np.linspace(num_static_landmarks, num_static_landmarks + num_dynamic_landmarks - 1,
                          num_dynamic_landmarks, dtype='uint16').reshape(-1, 1)
dynamic_landmarks = np.hstack((landmark_xy, landmark_id, landmark_v))

# 视野范围
fov = 90
# 运动噪声协方差矩阵：x、y和θ
Rt = 20 * np.array([[0.01, 0, 0],
                    [0, 0.01, 0],
                    [0, 0, 0.01]])
# 观测噪声协方差矩阵：距离和角度
Qt = 30 * np.array([[0.1, 0],
                    [0, 0.1]])

# 随机生成初始位置 (x, y) 在 [-10, 10] 范围内
initial_x = np.random.uniform(-10, 10)
initial_y = np.random.uniform(-10, 10)

# 随机生成初始方向在 [0, 2*pi] 范围内
initial_theta = np.random.uniform(0, 2 * np.pi)

initial_robot_state = [initial_x, initial_y, initial_theta]

r1 = Robot(initial_robot_state, fov, Rt, Qt)

# 生成初始真实状态
true_robot_state = [initial_robot_state]
sensor_observations = []

# 每一行代表一个step输入
motion_commands = np.zeros((steps, 3))
# 随机生成步长和曲率
step_size = np.random.uniform(low=0, high=6, size=steps).astype(int)
curvature = np.random.uniform(low=-1.0, high=1.0, size=steps)

motion_commands[:, 0] = step_size
motion_commands[:, 1] = curvature

# 状态转移矩阵
state_transition_matrix = np.array([[1, 0, 0, vm, 0],
                                    [0, 1, 0, 0, vm],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]])

# 初始化动态地标轨迹
dynamic_landmark_trajectory = dynamic_landmarks

num_particles = 10
pf = ParticleFilter(num_particles, initial_robot_state, Rt, Qt, fov)
# 生成动态地标轨迹
for _ in range(1, steps):
    # 更新每个地标的状态
    dynamic_landmarks = state_transition_matrix.dot(dynamic_landmarks.T).T
    # 将新的状态添加到轨迹中
    dynamic_landmark_trajectory = np.dstack((dynamic_landmark_trajectory, dynamic_landmarks))

# 生成机器人状态和观测
for movement, t in zip(motion_commands, range(steps)):
    # 生成了当前step的所有地标列表
    landmarks = np.append(static_landmarks, dynamic_landmark_trajectory[:, :3, t], axis=0)
    # 处理机器人运动与传感
    true_robot_state.append(r1.move(movement))
    sensor_observations.append(r1.sense(landmarks))

# 初始化状态矩阵
inf = 1e6
# 初始状态下对所有landmark未知，生成一个列向量
estimated_state = np.append(np.array([initial_robot_state]).T,
                            np.zeros((2 * (num_static_landmarks + num_dynamic_landmarks), 1)), axis=0)
new_estimated_state = estimated_state

# 一个对角线元素为inf，非对角线元素为0的方阵，代表对初始状态的协方差的估计
state_covariance = inf * np.eye(2 * (num_static_landmarks + num_dynamic_landmarks) + 3)
# 对初始状态加入一定不确定性
state_covariance[:3, :3] = np.eye(3)

# 元素全为0.5的列向量，代表对每个地标存在的初始概率的估计
landmark_existence_probability = 0.5 * np.ones((num_static_landmarks + num_dynamic_landmarks, 1))
covariance_sizes = []
predicted_positions = []
updated_positions = []

pf = ParticleFilter(num_particles, initial_robot_state, Rt, Qt, fov)

for movement, measurement in zip(motion_commands, sensor_observations):
    pf.predict(movement)

    # 使用粒子的平均状态作为预测状态
    predicted_state = np.mean(pf.particles, axis=0)
    predicted_positions.append(predicted_state[:2])

    pf.update(measurement, np.append(static_landmarks, dynamic_landmark_trajectory[:, :3, t], axis=0))

    # 使用粒子的平均状态作为更新状态
    updated_state = np.mean(pf.particles, axis=0)
    updated_positions.append(updated_state[:2])

    pf.resample()

# 绘制预测位置和更新位置的图
predicted_positions = np.array(predicted_positions)
updated_positions = np.array(updated_positions)

plt.figure()
plt.title('Predicted vs Updated Positions', fontsize=20)
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted')
# plt.plot(updated_positions[:, 0], updated_positions[:, 1], label='Updated')
plt.xlabel('X position', fontsize=16)
plt.ylabel('Y position', fontsize=16)
plt.legend(fontsize='large')
plt.show()
