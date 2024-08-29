from skrgbd.utils.logging import logger
from skrgbd.scanning.scan_helper.extensions.task import Task, log_context


class _Robot:
    def __init__(self, robot):
        self.robot = robot

        self.generate_trajectory_points = robot.generate_trajectory_points
        self.get_point_id = robot.get_point_id

    def move_to(self, *args, **kwargs):
        desc = 'Move robot'

        def target(task):
            logger.debug(f'{log_context}: {desc}')
            self.robot.move_to(*args, **kwargs)
            logger.debug(f'{log_context}: {desc} DONE')
        return Task(target, desc)


