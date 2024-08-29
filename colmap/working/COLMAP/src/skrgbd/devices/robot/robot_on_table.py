import numpy as np
import networkx as nx
import torch

from skrgbd.devices.robot.robot import Robot


class RobotOnTable(Robot):
    r"""Robot moving above the tabletop scene #2."""
    trajectory_name = 'table'

    def __init__(self, simulation=False):
        Robot.__init__(self, tcp_pos=(0, 0, .14, 0, 0, 0), simulation=simulation)
        self._set_parameters()
        self._cur_pos_id = 'unknown'

    def _set_parameters(self):
        self._up = torch.tensor([0, 0, 1.])
        right_movements = [dict(pos=[.13, -.41, .282], lookat=[1., -1., .2]),
                           dict(pos=[.13, -.41, .45], lookat=[1., -1., .2]),
                           dict(pos=[.13, -.41, .45], lookat=[.55, -.7, .2]),
                           dict(pos=[.13, -.41, .282], lookat=[.55, -.7, .2]),
                           dict(pos=[.12, -.36, .282], lookat=[.55, -.7, .2]),
                           dict(pos=[.11, -.36, .45], lookat=[.55, -.7, .2]),
                           dict(pos=[.11, -.36, .45], lookat=[1., -1., .2]),
                           dict(pos=[.12, -.36, .282], lookat=[1., -1., .2]),
                           dict(pos=[.29, -.33, .42], lookat=[1., -1., .2]),]
        middle_circle = make_middle_circle()
        small_circle = make_small_circle()
        cylinder = make_cylinder()
        trajectory = right_movements + middle_circle + small_circle + cylinder
        trajectory = trajectory[:-2]  # drop two points to make the whole trajectory 100 pts long
        self._trajectory = trajectory
        self._home_pos = (len(right_movements) + 6,)  # Pt 6 on middle_circle

        # Make pose graph
        poses_n = len(trajectory)
        pos_graph = nx.Graph()
        pos_graph.add_nodes_from(range(poses_n))
        pos_graph.add_nodes_from(['unknown', 'rest'])
        for pos_i in range(1, poses_n):
            pos_graph.add_edge(pos_i - 1, pos_i, time=edge_times[pos_i - 1])
        for pos_i in range(poses_n):
            pos_graph.add_edge(pos_i, self._home_pos[0], time=home_times[pos_i])
        for pos_id in ['unknown', 'rest']:
            pos_graph.add_edge(pos_id, self._home_pos[0], time=1)
        self._pos_graph = pos_graph

        # Set AWB pose: the cameras look directly at soft_right
        self._awb_pos = dict(pos=[.1, -.9, .83], lookat=[.0, -1.1, .83])
        # Set the position to pick camera settings
        self._cam_settings_pos = dict(pos=[1., -.2, .6], lookat=[1., -.6, .2])

    def move_to(self, pos, velocity=0.01):
        r"""Moves robot to the position pos.

        Parameters
        ----------
        pos : tuple with a single int
            in range [0, 99] (inclusive).
        velocity : float
            Velocity of movement, 0.1 at max.
        """
        pos_i = int(pos[0])
        path = self.get_path(self._cur_pos_id, pos_i)
        self.set_velocity(velocity)
        tposes = [self._lookat(**self._trajectory[next_pos_i], up=self._up) for next_pos_i in path]
        self._cur_pos_id = 'unknown'
        ret = self.rob.movels(tposes, acc=self.acceleration, vel=self.velocity)
        self._cur_pos_id = pos_i
        return ret

    def get_path(self, from_pos_i, to_pos_i):
        if from_pos_i == to_pos_i:
            return [to_pos_i]
        return nx.shortest_path(self._pos_graph, from_pos_i, to_pos_i, weight='time')[1:]

    def lookat(self, pos, lookat, up):
        r"""Move TCP to `pos` and look to `lookat`.

        Parameters
        ----------
        pos : array-like
            Position in Base coordinate system, (x, y, z).
        lookat : array-like
            Point of view in Base coordinate system, (x, y, z).
        up : array-like
            The "up" direction, (x, y, z).

        Returns
        -------
        pose : math3d.Transform
            Final transform from Base to TCP.
        """
        self._cur_pos_id = 'unknown'
        return super().lookat(pos, lookat, up)

    def move_home(self, velocity=.01):
        r"""Moves the robot to "home" position on the trajectory, close to the base."""
        return self.move_to(self._home_pos, velocity)

    def move_for_awb(self, velocity=.01):
        self.move_home(velocity)
        self.lookat(**self._awb_pos, up=self._up)

    def move_for_cam_settings(self, velocity=.01):
        self.move_home(velocity)
        self.lookat(**self._cam_settings_pos, up=self._up)

    def rest(self, velocity=.01):
        r"""Moves the robot to "rest" position."""
        super().rest(velocity)
        self._cur_pos_id = 'rest'

    def generate_trajectory_points(self):
        r"""Generates the trajectory points.

        Returns
        -------
        points : torch.Tensor
            of shape [points_n, 1].
        """
        points = torch.arange(len(self._trajectory)).view(-1, 1)
        return points


def make_middle_circle():
    r_corner_middle = .655
    center_corner = (.9, -.8)
    x_corner_middle = 0.4
    circle_corner = lambda a, r: np.sqrt(r ** 2 - (a - center_corner[0]) ** 2) + center_corner[1]

    pts = []
    for i in range(9):
        if i == 0:
            pts.append(dict(
                pos=[x_corner_middle - .12 + .05, circle_corner(x_corner_middle - .13, r_corner_middle) - .05, .5],
                lookat=[center_corner[0], center_corner[1], .4]
            ))
            pts.append(dict(
                pos=[x_corner_middle - .12 + .05, circle_corner(x_corner_middle - .13, r_corner_middle) - .05, .5],
                lookat=[center_corner[0], center_corner[1], .3],
            ))
        else:
            pts.append(dict(
                pos=[x_corner_middle - .15 + .05 * i + .05, circle_corner(x_corner_middle - .15 + .05 * i, r_corner_middle) - .05, .5],
                lookat=[center_corner[0], center_corner[1], .4]
            ))
            pts.append(dict(
                pos=[x_corner_middle - .15 + .05 * i + .05, circle_corner(x_corner_middle - .15 + .05 * i, r_corner_middle) - .05, .5],
                lookat=[center_corner[0], center_corner[1], .3]
            ))
    return pts


def make_small_circle():
    x_corner_small = .29
    r_corner_small = .6
    center_corner = (.65, -.65)
    circle_corner = lambda a, r: np.sqrt(r ** 2 - (a - center_corner[0]) ** 2) + center_corner[1]

    pts = []
    for i in range(6):
        pts.append(dict(
            pos=[x_corner_small + .05 * i, circle_corner(x_corner_small + .05 * i, r_corner_small), .495],
            lookat=[center_corner[0], center_corner[1], .2]
        ))
        if i >= 1:
            pts.append(dict(
                pos=[x_corner_small + .05 * i, circle_corner(x_corner_small + .05 * i, r_corner_small + .05), .55],
                lookat=[center_corner[0], center_corner[1] - .05, .35]
            ))
            pts.append(dict(
                pos=[x_corner_small + .05 * i, circle_corner(x_corner_small + .05 * i, r_corner_small + .05), .55],
                lookat=[center_corner[0] + .1, center_corner[1] - .05, .35]
            ))
            pts.append(dict(
                pos=[x_corner_small + .05 * i, circle_corner(x_corner_small + .05 * i, r_corner_small + .05), .495],
                lookat=[center_corner[0] + .1, center_corner[1], .2]
            ))
    return pts


def make_cylinder(x_n=6, z_n=9):
    x_st = .51
    y_st = -.05
    z_st = .26
    watch_y = -.6
    r = watch_y - y_st
    circle = lambda a: np.sqrt(r ** 2 - (a - z_st) ** 2) + watch_y

    pts = []
    flip_col = False
    for x in np.linspace(x_st, 1.05, x_n):
        pts_col = []
        for z in np.linspace(z_st, 0.56, z_n):
            y = circle(z)
            # pts_col.append(dict(pos=[x, y, z], lookat=[x, watch_y, z_st]))  # look forward
            if x > 0.7:
                pts_col.append(dict(pos=[x, y, z], lookat=[x - .08, watch_y, z_st]))  # look right
            if x < 0.7:
                pts_col.append(dict(pos=[x, y, z], lookat=[x + .08, watch_y, z_st]))  # look left
        if flip_col:
            pts_col = list(reversed(pts_col))
        flip_col = not flip_col
        pts.extend(pts_col)
    return pts


edge_times = [
    4.0, 2.1, 4.6, 1.8, 4.8, 2.0, 4.3, 4.7, 6.0, 1.7, 2.3, 1.7, 2.5, 1.7, 2.3, 1.7, 2.2, 1.7, 2.0, 1.8, 1.9, 1.9, 1.9,
    2.0, 1.8, 1.9, 8.7, 2.2, 2.4, 1.5, 1.8, 2.1, 2.3, 1.5, 1.9, 1.9, 2.3, 1.5, 1.9, 1.9, 2.3, 1.5, 1.9, 2.0, 2.3, 1.5,
    1.9, 3.8, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.7, 2.4, 1.7, 1.7, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 2.4, 1.6, 1.6, 1.6,
    1.6, 1.6, 1.6, 1.7, 1.7, 2.4, 1.7, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 2.4, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.7,
    2.4, 1.7, 1.6, 1.6, 1.6, 1.6, 1.6
]
home_times = [
    5.21, 4.71, 4.61, 5.21, 5.41, 4.91, 5.01, 5.41, 3.41, 4.91, 4.71, 3.91, 3.81, 2.30, 2.30, 0.40, 1.70, 2.10, 2.10,
    3.01, 2.91, 3.71, 3.61, 4.21, 4.51, 5.21, 5.51, 11.42, 10.12, 12.43, 12.42, 11.32, 9.82, 12.02, 11.92, 10.72, 9.72,
    11.92, 11.62, 10.32, 9.72, 11.82, 11.52, 10.23, 9.82, 11.82, 11.42, 10.12, 14.93, 13.73, 12.83, 12.02, 11.32, 10.62,
    10.02, 9.32, 8.82, 8.32, 8.82, 9.32, 10.02, 10.72, 11.52, 12.43, 13.43, 14.73, 14.83, 13.83, 13.03, 12.32, 11.62,
    11.12, 10.62, 10.12, 9.72, 9.82, 10.22, 10.62, 11.12, 11.72, 12.32, 13.13, 13.93, 15.03, 15.53, 14.33, 13.33, 12.63,
    11.92, 11.32, 10.82, 10.42, 10.02, 10.62, 10.92, 11.32, 11.82, 12.43, 13.13, 13.93
]
