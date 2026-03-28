from pykalman import KalmanFilter
import numpy as np

def kalman_estimate(y, x):
    """
    使用卡尔曼滤波器估计对冲比率（一阶回归：y = beta * x + c）
    :param y: 因变量（股票Y）
    :param x: 自变量（股票X）
    :return: 估计的对冲比率序列 beta
    """
    y_vals = y.values.reshape(-1, 1)
    x_vals = x.values

    # observation_matrices 对应每个时间步的 [x_t, 1]，维度 (n_timesteps, n_dim_obs, n_dim_state)
    observation_matrices = np.vstack([x_vals, np.ones_like(x_vals)]).T[:, np.newaxis, :]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrices,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2) * 1.0,
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 0.0001
    )

    state_means, _ = kf.filter(y_vals)

    # state_means[:, 0] 为 beta, state_means[:, 1] 为截距
    beta = state_means[:, 0]
    intercept = state_means[:, 1]
    return beta, intercept