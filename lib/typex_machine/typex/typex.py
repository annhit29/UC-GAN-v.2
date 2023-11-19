# Our classes
from typex.stator import Stator
from typex.rotor import Rotor
from typex.reflector import Reflector
from typex.encryptor import Encryptor


class TypeX:
    def __init__(self, encryptors=[]):
        # Encryptors must be a list, everything in that list
        # must be either a stator or a rotor and stators
        # must be before rotors
        if type(encryptors) != list:
            raise TypeError("Encryptors must be a list")
        else:
            found_rotors = False
            for e in encryptors:
                if isinstance(e, Rotor):
                    found_rotors = True
                elif isinstance(e, Stator):
                    if found_rotors:
                        raise ValueError("Stators must come before Rotors")
                else:
                    raise TypeError("All encryptors must be Stators or Rotors")
        self.encryptors = encryptors

        self.reflector = Reflector()

    def __clean(self, string):
        clean_list = []
        for char in string:
            if char.upper() in Encryptor.ALPHABET:
                clean_list.append(char.upper())
            elif char == ' ':
                clean_list.append('X')

        return ''.join(clean_list)

    def __step(self):
        for e in self.encryptors:
            if not e.step():
                break

    def encrypt(self, string):
        # Clean up the input string and
        # encrypt it
        encrypted_chars = []
        for char in self.__clean(string):
            # Before encrypting each letter, step the rotors
            self.__step()
            for e in self.encryptors:
                char = e.encrypt_character(char)
            char = self.reflector.encrypt_character(char)
            for e in reversed(self.encryptors):
                char = e.reverse_encrypt_character(char)
            encrypted_chars.append(char)

        return ''.join(encrypted_chars)
