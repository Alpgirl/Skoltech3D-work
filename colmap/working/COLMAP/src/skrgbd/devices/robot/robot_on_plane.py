import numpy as np

from skrgbd.devices.robot.robot import Robot


class RobotOnPlane(Robot):
    r"""Robot moving on a vertical plane orthogonal to the base X axis."""
    trajectory_name = 'calibration_plane'

    x = None
    y_min = None
    y_max = None
    z_min = None
    z_max = None
    up = None

    def __init__(self, simulation=False):
        super().__init__(simulation=simulation)
        self._set_parameters()

    def _set_parameters(self):
        self.x = .59
        self.y_min = -.47  # limited by the robot joint meeting the rig and wires
        self.y_max = .68
        self.z_min = .27  # limited by the wires meeting the table
        self.z_max = .99  # limited by the frame
        self.up = np.array([0, 0, 1.])

    def move_to(self, pos, velocity=.01):
        r"""Moves robot to the position pos.

        Parameters
        ----------
        pos : iterable of float
            (x, y), 0 <= x <= 1, 0 <= y <= 1,
            where (0, 0) corresponds to the bottom left corner, and (1, 1) corresponds to the top right corner.
        velocity : float
            Velocity of movement, 0.1 at max.
        """
        traj_x, traj_y = pos
        if traj_x > 1.0 or traj_x < 0.0 or traj_y > 1.0 or traj_y < 0.0:
            raise ValueError
        y = self.y_min + (self.y_max - self.y_min) * (1 - traj_x)
        z = self.z_min + (self.z_max - self.z_min) * traj_y

        self.set_velocity(velocity)
        pos = np.array([self.x, y, z])
        lookat = pos + [10, 0, 0]
        return self.lookat(pos, lookat, self.up)

    def move_home(self, velocity=.01):
        r"""Moves the robot to the "home" position on the trajectory, close to the base."""
        return self.move_to((.8, 0), velocity)

    def move_to_in_area(self, pos, area, velocity=.01):
        r"""Moves robot to the position relative to area.

        Parameters
        ----------
        pos : iterable of float
            (x, y), 0 <= x <= 1, 0 <= y <= 1,
            where (0, 0) corresponds to the bottom left corner, and (1, 1) corresponds to the top right corner.
        area : iterable of float
            (x_min, x_max, y_min, y_max)
        velocity : float
            Velocity of movement, 0.1 at max.
        """
        traj_x, traj_y = pos
        min_x, max_x, min_y, max_y = area
        traj_x = min_x + (max_x - min_x) * traj_x
        traj_y = min_y + (max_y - min_y) * traj_y
        return self.move_to((traj_x, traj_y), velocity)

    def move_over_area(self, step_size=None, endpoint=False, points_n=None, area=(0, 1, 0, 1), metric_step_size=False,
                       velocity=0.01, closure=None, closure_args=None, closure_kwargs=None, show_progress=True):
        r"""Moves the robot over the grid of points on the plane either with a specific step size
        or with a specific number of points in each dimension.

        Parameters
        ----------
        step_size : float
            Step size.
        endpoint : bool
            If True, include the endpoint for each dimension, even if it is not `step_size` away from the previous point.
        points_n : iterable of int
            (y_points_n, x_points_n).
        area : iterable of float
            (x_min, x_max, y_min, y_max)
        metric_step_size : bool
            If True, the step size is in meters. Otherwise it is relative to the size of the whole calibration plane area.
        velocity : float
            Endpoint movement velocity.
        closure : callable
            If not None, call this at each point, as closure(point_id, (x, y), *closure_args, **closure_kwargs).
        closure_args : iterable
        closure_kwargs : dict
        """
        points = self.meshgrid(step_size, endpoint, points_n, area, metric_step_size)
        points = points.reshape(-1, 2)
        self.move_over_points(points, velocity=velocity, closure=closure, closure_args=closure_args,
                              closure_kwargs=closure_kwargs, show_progress=show_progress)

    def meshgrid(self, step_size=None, endpoint=False, points_n=None, area=(0, 1, 0, 1), metric_step_size=False):
        r"""Makes the grid of points on the plane either with a specific step size
        or with a specific number of points in each dimension.

        Parameters
        ----------
        step_size : float
            Step size.
        endpoint : bool
            If True, include the endpoint for each dimension, even if it is not `step_size` away from the previous point.
        points_n : iterable of int
            (y_points_n, x_points_n).
        area : iterable of float
            (x_min, x_max, y_min, y_max)
        metric_step_size : bool
            If True, the step size is in meters. Otherwise it is relative to the size of the whole calibration plane area.

        Returns
        -------
        points : np.ndarray
            of shape [y_points_n, x_points_n, 2].
        """
        if (step_size is not None) and (points_n is not None):
            raise ValueError('Specify only either step_size or points_n.')

        x_min, x_max, y_min, y_max = area
        if points_n is not None:
            y_points_n, x_points_n = points_n
            points = np.stack(np.meshgrid(
                np.linspace(x_min, x_max, x_points_n),
                np.linspace(y_min, y_max, y_points_n),
                indexing='xy'
            ))
        if step_size is not None:
            if metric_step_size:
                x_step_size = step_size / (self.y_max - self.y_min)
                y_step_size = step_size / (self.z_max - self.z_min)
            else:
                x_step_size = y_step_size = step_size
            x = np.arange(x_min, x_max, x_step_size)
            y = np.arange(y_min, y_max, y_step_size)
            if endpoint:
                if x[-1] != x_max:
                    x = x.tolist()
                    x.append(x_max)
                if y[-1] != y_max:
                    y = y.tolist()
                    y.append(y_max)
            points = np.stack(np.meshgrid(x, y, indexing='xy'))
        points = points.transpose(1, 2, 0)
        return points
