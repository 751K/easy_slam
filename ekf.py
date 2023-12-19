import numpy as np


def predict(state_mean, state_cov, motion_control, motion_noise_cov):
    state_dim = len(state_mean)

    # 定义运动模型
    [delta_translation, delta_rotation1, delta_rotation2] = motion_control
    motion_vector = np.array([[delta_translation * np.cos(state_mean[2][0] + delta_rotation1)],
                              [delta_translation * np.sin(state_mean[2][0] + delta_rotation1)],
                              [delta_rotation1 + delta_rotation2]])
    motion_factor = np.append(np.eye(3), np.zeros((3, state_dim - 3)), axis=1)

    # 预测新的状态
    predicted_state_mean = state_mean + motion_factor.T.dot(motion_vector)

    # 定义运动模型的雅可比矩阵
    jacobian = np.array([[0, 0, -delta_translation * np.sin(state_mean[2][0] + delta_rotation1)],
                         [0, 0, delta_translation * np.cos(state_mean[2][0] + delta_rotation1)],
                         [0, 0, 0]])
    gain = np.eye(state_dim) + motion_factor.T.dot(jacobian).dot(motion_factor)

    # 预测新的协方差
    predicted_state_cov = gain.dot(state_cov).dot(gain.T) + motion_factor.T.dot(motion_noise_cov).dot(motion_factor)

    # 打印预测的位置
    print('predicted location\t x: {:.2f} \t y: {:.2f} \t theta: {:.2f}'.format(predicted_state_mean[0][0],
                                                                        predicted_state_mean[1][0],
                                                                        predicted_state_mean[2][0]))
    return predicted_state_mean, predicted_state_cov


def update(state_mean, state_cov, observations, static_prob, observation_noise_cov):
    state_dim = len(state_mean)

    for [range_obs, angle_obs, landmark_id] in observations:
        landmark_id = int(landmark_id)
        # 如果地标之前没有被观测到
        if state_cov[2 * landmark_id + 3][2 * landmark_id + 3] >= 1e6 and state_cov[2 * landmark_id + 4][
            2 * landmark_id + 4] >= 1e6:
            # 将地标的估计值设为当前的观测值
            state_mean[2 * landmark_id + 3][0] = state_mean[0][0] + range_obs * np.cos(angle_obs + state_mean[2][0])
            state_mean[2 * landmark_id + 4][0] = state_mean[1][0] + range_obs * np.sin(angle_obs + state_mean[2][0])

        # 如果地标是静态的
        if static_prob[landmark_id] >= 0.5:
            # 计算预期的观测值
            delta = np.array([state_mean[2 * landmark_id + 3][0] - state_mean[0][0],
                              state_mean[2 * landmark_id + 4][0] - state_mean[1][0]])
            q = delta.T.dot(delta)
            sqrt_q = np.sqrt(q)
            z_angle = np.arctan2(delta[1], delta[0])
            expected_obs = np.array([[sqrt_q], [z_angle - state_mean[2][0]]])

            # 计算雅可比矩阵
            F = np.zeros((5, state_dim))
            F[:3, :3] = np.eye(3)
            F[3, 2 * landmark_id + 3] = 1
            F[4, 2 * landmark_id + 4] = 1
            H_z = np.array([[-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
                            [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
            H = 1 / q * H_z.dot(F)

            # 计算卡尔曼增益
            K = state_cov.dot(H.T).dot(np.linalg.inv(H.dot(state_cov).dot(H.T) + observation_noise_cov))

            # 计算预期观测值和实际观测值之间的差异
            obs_diff = np.array([[range_obs], [angle_obs]]) - expected_obs
            obs_diff = (obs_diff + np.pi) % (2 * np.pi) - np.pi

            # 更新状态向量和协方差矩阵
            state_mean = state_mean + K.dot(obs_diff)
            state_cov = (np.eye(state_dim) - K.dot(H)).dot(state_cov)

    # 打印更新后的位置
    print('update location\t x: {:.2f} \t y: {:.2f} \t theta: {:.2f}'.format(state_mean[0][0], state_mean[1][0],
                                                                             state_mean[2][0]))
    return state_mean, state_cov, static_prob
