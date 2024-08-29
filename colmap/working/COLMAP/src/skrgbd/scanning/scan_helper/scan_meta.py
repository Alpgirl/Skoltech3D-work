import sqlite3

from skrgbd.utils.logging import logger


class ScanMeta:
    name = 'ScanMeta'

    def __init__(self, scene_dir):
        self.conn = sqlite3.connect(f'{scene_dir}/meta.db')
        self.init_tables()

    def __del__(self):
        self.conn.close()

    def init_tables(self):
        logger.debug(f'{self.name}: Init tables')
        with self.conn as conn:
            conn.execute('create table if not exists scanned_with_cameras (point_id text primary key)')
        logger.debug(f'{self.name}: Init tables DONE')

    def add_scanned_with_cameras(self, point_id):
        logger.debug(f'{self.name}: Add {point_id} to meta.scanned_with_cameras')
        with self.conn as conn:
            conn.execute('insert into scanned_with_cameras(point_id) values(?)', [point_id])
        logger.debug(f'{self.name}: Add {point_id} to meta.scanned_with_cameras DONE')

    def is_scanned_with_cameras(self, point_id):
        with self.conn as conn:
            is_scanned = conn.execute(
                'select exists(select null from scanned_with_cameras where point_id=?)', [point_id]).fetchone()[0]
        return bool(is_scanned)

    def reset_scanned_with_cameras(self, point_ids=None):
        logger.debug(f'{self.name}: Reset scanned_with_cameras')
        with self.conn as conn:
            if point_ids is None:
                conn.execute('delete from scanned_with_cameras')
            else:
                placeholder = ','.join('?' * len(point_ids))
                conn.execute(f'delete from scanned_with_cameras where point_id in ({placeholder})', point_ids)
        logger.debug(f'{self.name}: Reset scanned_with_cameras DONE')
