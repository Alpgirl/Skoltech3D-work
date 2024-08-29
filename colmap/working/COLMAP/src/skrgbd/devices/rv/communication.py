import json
import os
from time import sleep

from portalocker import RLock

from skrgbd.utils.logging import logger
from skrgbd.devices.rv.rv import RVGui


class RVClient:
    r"""Drop-in replacement for the RVGui class, which interacts with ScanCenter GUI using a python process
    running inside the virtual machine, which allows to work on the host in parallel.
    Use RVClient on the host and RVServer (below) in the virtual machine."""

    name = 'RVClient'

    def __init__(self):
        self.comm = Comm(in_vm=False)

    def init_project(self, name):
        self.call('init_project', name)

    def wait_for_scanning_end(self):
        self.call('wait_for_scanning_end')

    def scan(self):
        self.call('scan')

    def export_scans(self):
        self.call('export_scans')

    def call(self, method, *args):
        op = f'{method}({", ".join(str(_) for _ in args)})'
        logger.debug(f'{self.name}: Wait for {op}')
        self.comm.put_todo(method, *args)
        self.comm.wait_for_done()
        logger.debug(f'{self.name}: Wait for {op} DONE')


class RVServer:
    r"""Run in the virtual machine."""
    name = 'RVServer'

    def __init__(self):
        self.comm = Comm(in_vm=True)

    def serve_forever(self):
        rv = RVGui(in_vm=True)
        logger.debug(f'{self.name}: Waiting for commands.')
        while True:
            method, args = self.comm.get_new_todo()
            logger.debug(f'{self.name}: Got {method}({", ".join(str(_) for _ in args)})')
            getattr(rv, method)(*args)
            self.comm.put_done()


class Comm:
    name = 'Comm'

    def __init__(self, in_vm):
        if in_vm:
            self.todo = r'Z:\STL Shared Folder\todo.lock'
            self.done = r'Z:\STL Shared Folder\done.lock'
        else:
            self.todo = '/mnt/data/sk3d/stl_shared_folder/todo.lock'
            self.done = '/mnt/data/sk3d/stl_shared_folder/done.lock'
        open(self.todo, 'w+')
        open(self.done, 'w+')

    def get_new_todo(self, poll_interval=.5):
        while True:
            with RLock(self.todo, 'r') as f:
                cmd = f.readline()
                if cmd != '':
                    break
            sleep(poll_interval)
        with RLock(self.todo, 'w') as f:
            f.flush()
            os.fsync(f.fileno())
        method, args = json.loads(cmd)
        return method, args

    def put_done(self):
        logger.debug(f'{self.name}: Put done')
        with RLock(self.done, 'w') as f:
            f.write('DONE')
            f.flush()
            os.fsync(f.fileno())
        logger.debug(f'{self.name}: Put done DONE')

    def put_todo(self, method, *args):
        logger.debug(f'{self.name}: Put todo')
        cmd = json.dumps([method, args])
        with RLock(self.done, 'w') as f:
            f.flush()
            os.fsync(f.fileno())
        with RLock(self.todo, 'w') as f:
            f.write(cmd)
            f.flush()
            os.fsync(f.fileno())
        logger.debug(f'{self.name}: Put todo DONE')

    def wait_for_done(self, poll_interval=.5):
        while True:
            with RLock(self.done, 'r') as f:
                cmd = f.readline()
                if cmd == 'DONE':
                    break
            sleep(poll_interval)
