import numpy as np
import torch

from skrgbd.devices.robot.robot import Robot
from skrgbd.utils.math import spherical_to_cartesian


class RobotOnHumanSphere(Robot):
    r"""Robot moving on a sphere around a human.
    The sphere is stretched vertically and horizontally to account for the shape of the human body."""
    trajectory_name = 'human_sphere'

    def __init__(self, simulation=False):
        Robot.__init__(self, tcp_pos=(0, 0, .14, 0, 0, np.pi / 2), simulation=simulation)
        self._set_parameters()

    def _set_parameters(self):
        r"""Do not change these parameters without careful testing with the robot."""
        center_z = .42
        center_x = 1.07
        center_y = -.65
        self.center = torch.tensor([center_x, center_y, center_z])
        self.sphere_radius = .80

        self.theta_min = np.deg2rad(90)
        self.theta_max = np.deg2rad(180)
        self.phi_min = np.deg2rad(60)
        self.phi_max = np.deg2rad(90)
        self.dz_min = 0
        self.dz_max = .10
        self.dy_min = 0
        self.dy_max = .25

        self.up = torch.tensor([0, 0, 1.])
        self._awb_pos = dict(pos=[.1, -.9, .77], lookat=[.0, -1.1, .77])
        self._away_pos = dict(pos=[0.39, -0.16, 1.3], lookat=[10., -0.16, 1.3])

    def rest(self, velocity=.01):
        r"""Moves the robot to "rest" position."""
        if np.any(np.array(self._tcp_pos) != np.array((0, 0, 0.14, 0, 0, np.pi / 2))):
            raise RuntimeError('Resting with a non-default TCP pos is dangerous')
        self.move_home(velocity)
        self.lookat([0.39, -0.16, 0.32], [10., -0.16, 0.32], [0, 0, 1.])

    def move_to(self, pos, velocity=0.01):
        r"""Moves robot to the position pos.
        
        Parameters
        ----------
        pos : tuple of float
            (theta, phi), where (0, 0) corresponds to the bottom left corner, and (1, 1) corresponds to the top right corner.
        velocity : float
            Velocity of movement, 0.1 at max.
        """
        theta, phi = pos

        dy = self.dy_min + (self.dy_max - self.dy_min) * (1 - theta)
        dz = self.dz_min + (self.dz_max - self.dz_min) * phi
        theta = self.theta_min + (self.theta_max - self.theta_min) * theta
        phi = self.phi_min + (self.phi_max - self.phi_min) * (1 - phi)

        center = self.center.clone()
        center[1] += dy
        center[2] += dz

        pos = center + spherical_to_cartesian(torch.tensor([self.sphere_radius, theta, phi]))
        self.set_velocity(velocity)
        return self.lookat(pos, center, self.up)

    def move_home(self, velocity=.01):
        r"""Moves the robot to "home" position on the trajectory, close to the base."""
        return self.move_to((.75, .0), velocity)

    def move_for_awb(self, velocity=.01):
        self.move_home(velocity)
        self.lookat(**self._awb_pos, up=self.up)

    def move_away(self, velocity=.01):
        self.set_velocity(velocity)
        self.lookat(**self._away_pos, up=self.up)

    def move_for_cam_settings(self, velocity=.01):
        self.move_home(velocity)

    def generate_trajectory_points(self, pts_n=30):
        r"""Generates a trajectory with evenly spaced points.
        The trajectory starts in (1, 0), then goes all the way to the left, then one step up,
        then all the way to the right, then one step up, and so on.

        Parameters
        ----------
        pts_n : int
            The number of points.

        Returns
        -------
        points : torch.Tensor
            of shape [pts_n, 2].
        """
        zs = torch.linspace(0, 1, 11)
        ranges = torch.tensor([
            (.70, .85),  # .0
            (.30, .85),  # .1
            (.30, .88),  # .2
            (.27, .92),  # .3
            (.27, 1.0),  # .4
            (.27, 1.0),  # .5
            (.27, 1.0),  # .6
            (.27, 1.0),  # .7
            (.27, 1.0),  # .8
            (.27, 1.0),  # .9
            (.27, 1.0),  # 1.
        ])

        cum_ranges = (ranges[:, 1] - ranges[:, 0]).cumsum(0)
        thetas = torch.linspace(0, cum_ranges[-1], pts_n)
        z_ids = ((thetas.unsqueeze(1) - cum_ranges) <= 0).max(1)[1]

        starts = cum_ranges.roll(1)
        starts[0] = 0

        pts = []
        flip = False
        for i, z in enumerate(zs):
            thetas_z = thetas[z_ids == i] - starts[i] + ranges[i][0]

            flip = not flip
            if flip:
                thetas_z = thetas_z.flip(0)
            pts_z = torch.stack([thetas_z, torch.full_like(thetas_z, z)], 1)
            pts.append(pts_z)

        pts = torch.cat(pts, 0)
        return pts
