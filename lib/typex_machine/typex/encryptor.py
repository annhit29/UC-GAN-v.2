class Encryptor:
    ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, wiring=None, initial_position=0):

        if wiring is None:
            self.wiring = self.ALPHABET
        elif type(wiring) != str:
            raise TypeError("Wiring must be a string")
        elif len(wiring) != len(self.ALPHABET):
            raise ValueError("Wiring must be {0} letters "
                             "long".format(len(self.ALPHABET)))
        elif ''.join(sorted(list(wiring.upper()))) != self.ALPHABET:
            raise ValueError("Wiring must have unique, alphabetic characters")
        else:
            self.wiring = wiring.upper()

        if type(initial_position) != int:
            raise TypeError("Initial position must be an integer")
        elif not (0 <= initial_position < len(self.ALPHABET)):
            raise ValueError("Initial position must be between 0 "
                             "and {0}".format(len(self.ALPHABET)-1))
        else:
            self.position = initial_position

    def __char_to_pos(self, char):
        if type(char) != str:
            raise TypeError("Input must be a string")
        elif len(char) != 1:
            raise ValueError("Input must be a single character")
        elif char not in self.ALPHABET:
            raise ValueError("Input must be a capital letter")
        else:
            return ord(char) - 65

    # Perform a substitution for a single character using the
    # current wiring and position of the encryptor
    def encrypt_character(self, char):
        pos = self.__char_to_pos(char) + self.position
        return self.wiring[pos % len(self.wiring)]

    def reverse_encrypt_character(self, char):
        pos = self.wiring.index(char) - self.position
        return list(self.ALPHABET)[pos % len(self.wiring)]

    # If appropriate, increments self's current position by
    # one step. Return True if the next Encryptor in sequence
    # should also step(); otherwise, return False.
    def step(self):
        raise NotImplementedError('step() must be implemented by Encryptor '
                                  'subclasses')
