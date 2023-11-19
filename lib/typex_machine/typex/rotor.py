from typex.encryptor import Encryptor


class Rotor(Encryptor):
    def __init__(self, wiring=None, initial_position=0, notchings=[]):
        super().__init__(wiring, initial_position)
        self.notchings = notchings

    def step(self):
        self.position = (self.position + 1) % len(self.wiring)
        return self.position in self.notchings