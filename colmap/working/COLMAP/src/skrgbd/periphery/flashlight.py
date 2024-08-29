from skrgbd.utils.parallel import ThreadSet, PropagatingThread as Thread


class Flashlight:
    def __init__(self, phones):
        self.phones = phones

    def on(self):
        threads = ThreadSet([Thread(target=phone.flash_on) for phone in self.phones])
        threads.start()
        threads.join()

    def off(self):
        threads = ThreadSet([Thread(target=phone.flash_off) for phone in self.phones])
        threads.start()
        threads.join()
