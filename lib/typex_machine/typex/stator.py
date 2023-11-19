from typex.encryptor import Encryptor


class Stator(Encryptor):
    def step(self):
        # Stators never rotate - they are static
        # They always tell their neighbor to rotate
        return True
