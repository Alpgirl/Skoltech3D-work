import datetime
from time import sleep
import sqlite3
import subprocess

from matplotlib import dates
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt


def main(log_timeout=10):
    r"""TODO

    Parameters
    ----------
    log_timeout : float
        Logging timeout in seconds.
    """
    devices = 'phone_left', 'phone_right'
    colors = 'tab:blue', 'tab:orange'
    dev_data = dict()

    # Read the old data
    db = DB()
    for dev in devices:
        data = dict()
        db_data = np.array(db.get_phone_data(dev))
        if db_data.shape == (0,):
            db_data = db_data.reshape([0, 4])
        data['timestamps'] = db_data[:, 0].tolist()
        data['temperature'] = db_data[:, 1].tolist()
        data['battery'] = db_data[:, 2].tolist()
        data['disk'] = db_data[:, 3].tolist()
        dev_data[dev] = data

    # Prepare the plotting
    figure, [ax_temp, ax_bat, ax_disk] = plt.subplots(1, 3, figsize=(8 * 3, 8))
    formatter = dates.DateFormatter('%H:%M')

    def update(i, plot=True):
        for phone in devices:
            ts, temp, bat, disk, bat_dph = get_phone_data(phone)
            db.append_phone_data(phone, ts, temp, bat, disk)
            dev_data[phone]['timestamps'].append(ts)
            dev_data[phone]['temperature'].append(temp)
            dev_data[phone]['battery'].append(bat)
            dev_data[phone]['disk'].append(disk)
            dev_data[phone]['bat_dph'] = bat_dph

        if plot:
            ax_temp.cla()
            ax_temp.set_title('Frame temperature')
            for dev, c in zip(devices, colors):
                ax_temp.plot(dev_data[dev]['timestamps'], dev_data[dev]['temperature'], label=dev, c=c)
            ax_temp.xaxis.set_major_formatter(formatter)
            ax_temp.legend()

            ax_bat.cla()
            ax_bat.set_title('Battery')
            for dev, c in zip(devices, colors):
                ax_bat.plot(dev_data[dev]['timestamps'], dev_data[dev]['battery'], c=c)
                t = dev_data[dev]['timestamps'][-1]
                b = dev_data[dev]['battery'][-1]
                bat_dph = dev_data[dev]['bat_dph']
                bb = b + bat_dph
                if bb > 100:
                    d_b = 100 - b
                    bb = 100
                    dt = d_b / bat_dph
                elif bb < 0:
                    d_b = b
                    bb = 0
                    dt = d_b / -bat_dph
                else:
                    dt = 1
                tt = t + datetime.timedelta(hours=dt)
                ax_bat.plot([t, tt], [b, bb], alpha=.3, c=c)

            ax_disk.cla()
            ax_disk.set_title('Disk')
            for dev, c in zip(devices, colors):
                ax_disk.plot(dev_data[dev]['timestamps'], dev_data[dev]['disk'], c=c)
            ax_disk.xaxis.set_major_formatter(formatter)

    # Make several quick updates to see that the plotting works
    for _ in range(2):
        update(None, plot=False)
        sleep(.5)
    update(None)

    # Monitor the data
    animation = FuncAnimation(figure, update, interval=log_timeout * 1000)
    plt.show()


class DB:
    r"""TODO"""

    def __init__(self):
        log_dir = '/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/logs/warmup'
        today = datetime.datetime.now().strftime('%y_%m_%d')
        db_filename = f'{log_dir}/{today}.db'
        self.con = sqlite3.connect(db_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        self.init_tables()

    def init_tables(self):
        with self.con as con:
            for phone in 'phone_left', 'phone_right':
                con.execute(f'create table if not exists {phone}_data'
                            ' (timestamp timestamp primary key, temperature real, battery integer, disk real)')

    def append_phone_data(self, phone_name, timestamp, temp, bat, disk):
        with self.con as con:
            con.execute(f'insert into {phone_name}_data(timestamp,temperature,battery,disk)'
                        ' values(?,?,?,?)', [timestamp, temp, bat, disk])

    def get_phone_data(self, phone_name):
        with self.con as con:
            data = con.execute(f'select timestamp, temperature, battery, disk from {phone_name}_data').fetchall()
        return data


def get_phone_data(phone_name):
    r"""Gets phone data.

    Parameters
    ----------
    phone_name : str

    Returns
    -------
    now : datetime.datetime
    temperature : float
        Frame temperature in degrees Celsius.
    battery : int
        Current battery percentage.
    disk : float
        Free disk space in Gigabytes.
    battery_delta_per_hour : float
        Estimated battery delta per hour.
    """

    phone_adress = phone_name_to_adress[phone_name]
    timestamp = datetime.datetime.now()
    temp, disk, bat, cur_now = subprocess.check_output([
        'adb', '-s', f'{phone_adress}', 'shell',
        'cat /sys/class/thermal/thermal_zone20/temp;'
        '(df -k /data | tail -1 | cut -F 4);'
        'cat /sys/class/power_supply/Battery/{capacity,current_now}'
    ]).decode().split('\n')[:4]
    temp = int(temp) / 1000
    bat = int(bat)  # perce
    disk = int(disk) / 1048576  # Gigabytes
    bat_dph = int(cur_now) / 4000 * 100

    return timestamp, temp, bat, disk, bat_dph


# phone_name_to_adress = {'phone_left': f'{1:016}', 'phone_right': f'{2:016}'}
phone_name_to_adress = {'phone_left': '192.168.1.3:5555', 'phone_right': '192.168.1.4:5555'}


if __name__ == '__main__':
    main()
