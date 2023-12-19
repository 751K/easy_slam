import numpy as np


class Robot:
    def __init__(self, state, range, motion_cov, observation_cov):
        # 初始化机器人的状态，包括位置和方向
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array(state)

        # 将视野范围转换为弧度
        self.range = np.deg2rad(range)

        # 运动噪声的协方差矩阵
        self.motion_cov = motion_cov

        # 观测噪声的协方差矩阵
        self.observation_cov = observation_cov

    def move(self, control):
        # 根据运动控制信号更新机器人的真实状态
        noise = np.matmul(np.random.randn(1, 3), self.motion_cov)[0]
        [delta_trans, delta_rot1, delta_rot2] = control[:3] + noise
        self.state[0] += delta_trans * np.cos(self.state[2] + delta_rot1)
        self.state[1] += delta_trans * np.sin(self.state[2] + delta_rot1)
        self.state[2] = (self.state[2] + delta_rot1 + delta_rot2 + np.pi) % (2 * np.pi) - np.pi
        return self.state

    def sense(self, landmarks):
        # 对环境中的地标进行观测，并返回观测结果
        landmarks = np.array(landmarks)
        relative_angles = np.arctan2((landmarks[:, 1] - self.state[1]), (landmarks[:, 0] - self.state[0]))
        relative_angles_2pi = (relative_angles + 2 * np.pi) % (2 * np.pi)
        range_left = (self.state[2] + self.range / 2 + 2 * np.pi) % (2 * np.pi)
        range_right = (self.state[2] - self.range / 2 + 2 * np.pi) % (2 * np.pi)
        visible = ((range_left - relative_angles_2pi + np.pi) % (2 * np.pi) - np.pi > 0) & (
                (range_right - relative_angles_2pi + np.pi) % (2 * np.pi) - np.pi < 0)
        observations = np.empty((0, 3))
        if np.any(visible):
            ranges = (np.sqrt(np.square(landmarks[visible, 1] - self.state[1]) + np.square(
                landmarks[visible, 0] - self.state[0])) + self.observation_cov[0][0]
                      * np.random.randn(sum(visible)))

            bearings = (relative_angles[visible] - self.state[2] + self.observation_cov[1][1]
                        * np.random.randn(sum(visible)) + np.pi) % (2 * np.pi) - np.pi

            observations = np.hstack((ranges.reshape(-1, 1), bearings.reshape(-1, 1),
                                      landmarks[visible, 2].reshape(-1, 1)))
        return observations