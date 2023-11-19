from typex.encryptor import Encryptor


class Reflector(Encryptor):
    # Construct a new reflector (with hardcoded wiring)
    def __init__(self):
        super().__init__(wiring=''.join(list(reversed(self.ALPHABET))))

    def step(self):
        raise NotImplementedError('step() cannot be called on a Reflector')
