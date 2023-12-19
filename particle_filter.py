import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, initial_state, motion_noise_cov, observation_noise_cov, view_range):
        self.num_particles = num_particles
        motion_noise = np.random.randn(num_particles, 3)
        self.particles = np.tile(np.array(initial_state), (num_particles, 1)) + np.matmul(motion_noise,
                                                                                          motion_noise_cov)
        self.weights = np.ones(num_particles) / num_particles
        self.motion_noise_cov = motion_noise_cov
        self.observation_noise_cov = observation_noise_cov
        self.view_range = view_range

    def predict(self, control_signal):
        # 根据运动模型和控制信号预测粒子的下一个状态
        for i in range(self.num_particles):
            motion_noise = np.matmul(np.random.randn(1, 3), self.motion_noise_cov)[0]  # 生成运动噪声
            [delta_trans, delta_rot1, delta_rot2] = control_signal[:3] + motion_noise  # 计算运动增量
            self.particles[i, 0] += delta_trans * np.cos(self.particles[i, 2] + delta_rot1)  # 更新粒子的x坐标
            self.particles[i, 1] += delta_trans * np.sin(self.particles[i, 2] + delta_rot1)  # 更新粒子的y坐标
            self.particles[i, 2] = (self.particles[i, 2] + delta_rot1 + delta_rot2 + np.pi) % (
                    2 * np.pi) - np.pi  # 更新粒子的方向

    def update(self, observations, landmarks):
        # 根据观测数据更新粒子的权重
        for i in range(self.num_particles):
            for observation in observations:
                expected_observation = self.expected_observation(self.particles[i], landmarks)  # 计算预期的观测值
                self.weights[i] *= np.prod(self.observation_probability(observation, expected_observation))  # 更新粒子的权重

        self.weights = (self.weights + 1e-10) / (np.sum(self.weights) + 1e-10 * len(self.weights))  # 归一化粒子的权重

    def resample(self):
        # 根据粒子的权重进行重采样
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)  # 生成新的粒子索引
        self.particles = self.particles[indices]  # 生成新的粒子集
        self.weights = np.ones(self.num_particles) / self.num_particles  # 重置粒子的权重

    @staticmethod
    def expected_observation(state, landmarks):
        # 计算预期的观测值
        relative_angles = np.arctan2((landmarks[:, 1] - state[1]), (landmarks[:, 0] - state[0]))  # 计算地标相对于粒子的角度
        ranges = np.sqrt(np.square(landmarks[:, 1] - state[1]) + np.square(landmarks[:, 0] - state[0]))  # 计算地标相对于粒子的距离
        bearings = (relative_angles - state[2] + np.pi) % (2 * np.pi) - np.pi  # 计算地标相对于粒子的角度
        expected_observations = np.hstack(
            (ranges.reshape(-1, 1), bearings.reshape(-1, 1), landmarks[:, 2].reshape(-1, 1)))  # 生成预期的观测值
        return expected_observations

    def observation_probability(self, observation, expected_observation):
        # 计算观测概率
        diff = observation - expected_observation  # 计算实际的观测值和预期的观测值之间的差异
        diff[:, 2] = (diff[:, 2] + np.pi) % (2 * np.pi) - np.pi  # 将角度差异转换到[-pi, pi]区间
        diff = diff[:, :2]
        # P(x) = exp(-0.5 * (x - μ)^T * Σ^-1 * (x - μ))
        inv_cov = np.linalg.inv(self.observation_noise_cov)  # 计算协方差矩阵的逆
        probabilities = np.array([np.exp(-0.5 * d.dot(inv_cov).dot(d.T)) for d in diff])  # 对每一行分别计算概率
        return probabilities
