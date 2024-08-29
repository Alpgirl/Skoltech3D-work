import signal


try:
    sensor_freeze_sig = signal.SIGUSR1


    def sensor_freeze_handler(signum, frame):
        # input('Detected sensor freeze.')
        pass


    def announce_sensor_freeze():
        signal.raise_signal(sensor_freeze_sig)


    signal.signal(sensor_freeze_sig, sensor_freeze_handler)
except AttributeError:
    pass
