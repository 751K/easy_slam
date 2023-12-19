# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def stateToArrow(state):
    x = state[0]
    y = state[1]
    dx = 0.5 * np.cos(state[2])
    dy = 0.5 * np.sin(state[2])
    return x, y, dx, dy


def plotMap(ls, ldt, hist, robot, map_size):
    # 清空当前的图形
    plt.clf()

    # 获取机器人的真实状态
    x = robot.robot_state
    # 获取机器人的视场范围
    fov = robot.view_range

    # 绘制fov，使用红线表示
    plt.plot([x[0], x[0] + 50 * np.cos(x[2] + fov / 2)], [x[1], x[1] + 50 * np.sin(x[2] + fov / 2)], color="r")
    plt.plot([x[0], x[0] + 50 * np.cos(x[2] - fov / 2)], [x[1], x[1] + 50 * np.sin(x[2] - fov / 2)], color="r")

    # 对历史状态进行遍历，使用箭头绘制出历史路径
    for state in hist:
        plt.arrow(*stateToArrow(state), head_width=0.5)
    # 绘制静态landmark
    plt.scatter(ls[:, 0], ls[:, 1], s=10, marker="s", color=(0, 0, 1))

    # 画动态landmark的轨迹
    for i in range(ldt.shape[2]):
        plt.scatter(ldt[:, 0, i], ldt[:, 1, i], s=10, marker="s", color=(0, 1, 0))

    # 设置x轴和y轴的范围
    plt.xlim([-map_size / 2, map_size / 2])
    plt.ylim([-map_size / 2, map_size / 2])
    # 设置标题
    plt.title('True environment', fontsize=20)
    # plt.legend(fontsize='large')
    plt.show()


# Plot:
# Robot state estimates (red/green)
# Current robot state covariances
# Field of view
# Currently observed landmarks with covariances and lines
# Previously observed landmarks


# 可视化机器人的状态估计值和观测到的地标
def plotEstimate(mu, cov, robot, map_size):
    plt.figure(figsize=(20, 20))

    plt.cla()

    # 绘制历史
    for i in range(mu.shape[1]):
        if i == 0 or i % 2 == 1:
            plt.arrow(*stateToArrow(mu[:3, i]), head_width=0.5, color=(1, 0, 0))
        else:
            plt.arrow(*stateToArrow(mu[:3, i]), head_width=0.5, color=(0, 1, 0))

    # 绘制fov视野
    fov = robot.view_range
    plt.plot([mu[0, -1], mu[0, -1] + 50 * np.cos(mu[2, -1] + fov / 2)],
             [mu[1, -1], mu[1, -1] + 50 * np.sin(mu[2, -1] + fov / 2)], color="r")
    plt.plot([mu[0, -1], mu[0, -1] + 50 * np.cos(mu[2, -1] - fov / 2)],
             [mu[1, -1], mu[1, -1] + 50 * np.sin(mu[2, -1] - fov / 2)], color="r")

    # 绘制协方差椭圆
    # print("cov: "+str(cov))
    robot_cov = Ellipse(xy=mu[:2, -1], width=cov[0, 0], height=cov[1, 1], angle=0)
    robot_cov.set_edgecolor((0, 0, 0))
    robot_cov.set_fill(0)
    plt.gca().add_artist(robot_cov)

    # 绘制观测到的landmark
    n = int((len(mu) - 3) / 2)
    for i in range(n):
        if cov[2 * i + 3, 2 * i + 3] < 1e6 and cov[2 * i + 3, 2 * i + 3] < 1e6:
            zx = mu[2 * i + 3, -1]
            zy = mu[2 * i + 4, -1]
            plt.scatter(zx, zy, marker='s', s=10, color=(0, 0, 1))

    plt.xlim([-map_size / 2, map_size / 2])
    plt.ylim([-map_size / 2, map_size / 2])
    plt.title('Observations and trajectory estimate')
    plt.pause(0.1)
    plt.show()


# 绘制机器人的位置和它观察到的地标点
def plotMeasurement(mu, cov, obs, n):
    a = plt.subplot(132, aspect='equal')

    for z in obs:
        j = int(z[2])
        zx = mu[2 * j + 3]
        zy = mu[2 * j + 4]
        if j < n:
            plt.plot([mu[0], zx], [mu[1], zy], color=(0, 0, 1))
        else:
            plt.plot([mu[0], zx], [mu[1], zy], color=(0, 1, 0))

        landmark_cov = Ellipse(xy=[zx, zy], width=cov[2 * j + 3][2 * j + 3], height=cov[2 * j + 4][2 * j + 4], angle=0)
        landmark_cov.set_edgecolor((0, 0, 0))
        landmark_cov.set_fill(0)
        a.add_artist(landmark_cov)
        plt.pause(0.0001)

    plt.pause(0.01)


# 绘制估计误差的平方图
def plotError(mu, x_true):
    b = plt.subplot(133)
    mu = mu[:3, 0::2]  # keep only x,y,theta
    x_true = np.asarray(x_true).T[:, :mu.shape[1]]
    dif = np.power(np.abs(mu - x_true), 2)
    err = dif[0, :] + dif[1, :]
    b.plot(err, color="r")
    plt.title('Squared estimation error')
    plt.xlabel('Steps')
    plt.ylabel('Squared error')


#   b.plot(dif[2,:])

def plot_covariance_sizes(covariance_sizes):
    plt.figure()
    plt.title('Covariance Size Over Time', fontsize=20)
    plt.plot(covariance_sizes)
    plt.xlabel('Time step', fontsize=16)
    plt.ylabel('Covariance', fontsize=16)
    plt.show()


def plot_positions(predicted_positions, updated_positions):
    plt.figure()
    plt.title('Predicted vs Updated Positions', fontsize=20)
    plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted')
    plt.plot(updated_positions[:, 0], updated_positions[:, 1], label='Updated')
    plt.xlabel('X position', fontsize=16)
    plt.ylabel('Y position', fontsize=16)
    plt.legend(fontsize='large')
    plt.show()
