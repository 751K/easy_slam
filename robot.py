import numpy as np


class Robot:
    def __init__(self, init_state, view_range, motion_noise_cov, observation_noise_cov):
        # 初始化机器人的状态
        init_state[2] = (init_state[2] + np.pi) % (2 * np.pi) - np.pi
        self.robot_state = np.array(init_state)  # 机器人的真实状态，包括位置和方向
        self.view_range = np.deg2rad(view_range)  # 视野范围，转换为弧度
        self.motion_noise_cov = motion_noise_cov  # 运动噪声的协方差矩阵
        self.observation_noise_cov = observation_noise_cov  # 观测噪声的协方差矩阵

    def move(self, control_signal):
        # 根据运动控制信号更新机器人的真实状态
        motion_noise = np.matmul(np.random.randn(1, 3), self.motion_noise_cov)[0]
        [delta_trans, delta_rot1, delta_rot2] = control_signal[:3] + motion_noise
        self.robot_state[0] += delta_trans * np.cos(self.robot_state[2] + delta_rot1)
        self.robot_state[1] += delta_trans * np.sin(self.robot_state[2] + delta_rot1)
        self.robot_state[2] = (self.robot_state[2] + delta_rot1 + delta_rot2 + np.pi) % (2 * np.pi) - np.pi
        return self.robot_state

    def sense(self, landmarks):
        # 对环境中的地标进行观测，并返回观测结果
        landmarks = np.array(landmarks)
        relative_angles = np.arctan2((landmarks[:, 1] - self.robot_state[1]), (landmarks[:, 0] - self.robot_state[0]))
        relative_angles_2pi = (relative_angles + 2 * np.pi) % (2 * np.pi)
        view_range_left = (self.robot_state[2] + self.view_range / 2 + 2 * np.pi) % (2 * np.pi)
        view_range_right = (self.robot_state[2] - self.view_range / 2 + 2 * np.pi) % (2 * np.pi)
        visible = ((view_range_left - relative_angles_2pi + np.pi) % (2 * np.pi) - np.pi > 0) & (
                (view_range_right - relative_angles_2pi + np.pi) % (2 * np.pi) - np.pi < 0)
        observations = np.empty((0, 3))
        if np.any(visible):
            ranges = (np.sqrt(np.square(landmarks[visible, 1] - self.robot_state[1]) + np.square(
                landmarks[visible, 0] - self.robot_state[0])) + self.observation_noise_cov[0][0]
                      * np.random.randn(sum(visible)))

            bearings = (relative_angles[visible] - self.robot_state[2] + self.observation_noise_cov[1][1]
                        * np.random.randn(sum(visible)) + np.pi) % (2 * np.pi) - np.pi

            observations = np.hstack((ranges.reshape(-1, 1), bearings.reshape(-1, 1),
                                      landmarks[visible, 2].reshape(-1, 1)))
        return observations
