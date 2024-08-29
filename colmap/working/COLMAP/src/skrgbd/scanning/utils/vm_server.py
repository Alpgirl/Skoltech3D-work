import sys

sys.path.append('Z:\sk_robot_rgbd_data\src')
from skrgbd.devices.rv.communication import RVServer


if __name__ == '__main__':
    server = RVServer()
    server.serve_forever()
