import numpy as np
import plotly
import plotly.graph_objects as go


def pflat(x):
    return x/x[-1]


def rgb2grey(img):
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def decompose_camera_matrix(P, K):
    """
    Decompose camera matrices to R and t s.t P[i] = K*R^T[I -t]
    :param P: ndarray of shape [n_cam, 3, 4], the cameras
    :param K: ndarray of shape [n_cam, 3, 3], the calibration matrices
    :return: R: ndarray of shape [n_cam, 3, 3]
            t: ndarray of shape [n_cam, 3]
    """
    Rt = np.linalg.inv(K) @ P
    Rs = np.transpose(Rt[:, :, :3],(0,2,1))
    ts = np.squeeze(-Rs @ np.expand_dims(Rt[:, 0:3, 3], axis=-1))
    return Rs, ts


def plot_cameras(P, K, X, title='reconstruction', point_colors='#3366CC'):
    """
    Plot a 3D image of the points and cameras
    :param P: ndarray of shape [n_cam, 3, 4], the cameras
    :param K: ndarray of shape [n_cam, 3, 3], the calibration matrices
    :param X: ndarray of shape [4, n_points], the predicted 3D points
    :param title: the name of the plot
    :param plotly color. Can by an ndarray of shapes [N_points, 3]
    """
    R,t = decompose_camera_matrix(P, K)
    data = []
    data.append(get_3D_quiver_trace(t, R[:, :3, 2], color='#86CE00', name='cam_learn'))
    data.append(get_3D_scater_trace(t.T, color='#86CE00', name='cam_learn', size=1))
    data.append(get_3D_scater_trace(X[:3,:], point_colors, '3D points', size=0.5))

    fig = go.Figure(data=data)
    path = title+'.html'
    plotly.offline.plot(fig, filename=path, auto_open=False)


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=directions[:, 0],
        v=directions[:, 1],
        w=directions[:, 2],
        sizemode='absolute',
        sizeref=0.5,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_3D_scater_trace(points, color, name='3d_points',size=0.5):
    assert points.shape[0] == 3, "3d plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d plot input points are not correctely shaped "

    trace = go.Scatter3d(
        name=name,
        x=points[0, :],
        y=points[1, :],
        z=points[2, :],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
        )
    )
    return trace
