import torch

from skrgbd.devices.robot.robot_on_sphere import RobotOnSphere


class RobotOnWall(RobotOnSphere):
    def __init__(self):
        super().__init__()
        self.board_center = torch.tensor([.12, 1.5, .71])
        self.wall_home = torch.tensor([.166, .34, .71])
        self.cu = torch.tensor([.166, .34, 1.09])
        self.cd = torch.tensor([.166, .34, .4])
        self.rc = torch.tensor([.3, .55, .71])
        self.lc = torch.tensor([-.2, .55, .71])

    def move_to_bridge(self, velocity=.01):
        x, y, z = .7, -.16, .71
        self.set_velocity(velocity)
        self.lookat([x, y, z], [x + 10, y, z], self.up)

    def move_to_wall_home(self, velocity=.01):
        self.move_on_wall_to(self.wall_home, velocity)

    def move_on_wall_to(self, pos, velocity):
        self.set_velocity(velocity)
        self.lookat(pos, self.board_center, self.up)

    def move_from_pt_to_board(self, pt, dist=.51, velocity=.01):
        pos = self.board_center + torch.nn.functional.normalize(pt - self.board_center, dim=0) * dist
        self.set_velocity(velocity)
        self.lookat(pos, self.board_center, self.up)


