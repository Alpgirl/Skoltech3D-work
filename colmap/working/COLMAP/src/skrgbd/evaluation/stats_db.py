from pathlib import Path
import sqlite3

from skrgbd.utils.logging import tqdm


class StatsDB:
    r"""Wrapper around SQLite for storing evaluation statistics.

    Parameters
    ----------
    file_db : str
    mode : {'r', 'w', 'a'}
    """

    def __init__(self, file_db, mode='r'):
        if mode == 'w':
            Path(file_db).unlink(missing_ok=True)
        mode = {'r': 'ro', 'w': 'rwc', 'a': 'rwc'}[mode]
        self.conn = sqlite3.connect(f'file:{file_db}?mode={mode}', uri=True)

        self.measures = ('accuracy', 'completeness', 'mean_ref_to_rec', 'mean_rec_to_ref')
        if mode == 'rwc':
            self.conn = sqlite3.connect(file_db)
            with self.conn:
                for measure in 'accuracy', 'completeness':
                    self.conn.execute(
                        f'create table if not exists {measure}'
                        ' (method text, version text, scene text, camera text, light text, threshold real, value real'
                        ', primary key(method, version, scene, camera, light, threshold))')

                for measure in 'mean_ref_to_rec', 'mean_rec_to_ref':
                    self.conn.execute(
                        f'create table if not exists {measure}'
                        ' (method text, version text, scene text, camera text, light text, value real'
                        ', primary key(method, version, scene, camera, light))')

    def __del__(self):
        self.conn.close()

    def set_measure(self, measure, method, version, scene, camera, light, values, thresholds=None):
        r"""Sets value of a measure.

        Parameters
        ----------
        measure : {'accuracy', 'completeness', 'mean_ref_to_rec', 'mean_rec_to_ref'}
            For 'mean_ref_to_rec' and 'mean_rec_to_ref', thresholds must be None, and values must be a single float.
        method : str
        version : str
        scene : str
        camera : str
        light : str
        values : iterable of float or float
        thresholds : iterable of float
        """
        if light is None:
            light = 'none'
        if thresholds is None:
            with self.conn:
                statement = (f'insert or replace into {measure} '
                             '(method, version, scene, camera, light, value) values(?, ?, ?, ?, ?, ?)')
                params = [method, version, scene, camera, light, values]
                self.conn.execute(statement, params)
        else:
            with self.conn:
                statement = (f'insert or replace into {measure} '
                             '(method, version, scene, camera, light, threshold, value) values(?, ?, ?, ?, ?, ?, ?)')
                params = ([method, version, scene, camera, light, threshold, value]
                          for (threshold, value) in zip(thresholds, values))
                self.conn.executemany(statement, params)

    def merge_from(self, files_db, show_progress=True):
        r"""Merges data from other databases.

        Parameters
        ----------
        files_db : iterable of str
        show_progress : bool
        """
        show_progress = tqdm if show_progress else (lambda x: x)
        for db in show_progress(files_db):
            with self.conn:
                self.conn.execute('attach database ? as db', [db])
                for measure in self.measures:
                    self.conn.execute('insert or replace into ' + measure + ' select * from db.' + measure)
                self.conn.execute('commit')
                self.conn.execute('detach db')
