alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
alphabet_char_to_index = {char: i for i, char in enumerate(alphabet)}
valid_reflector_names = {"A", "B", "C", "UKWB", "UKWC"}
ring_settings = [f'0{i}' if i < 10 else str(i) for i in range(1, 27)]

beta = "LEYJVCNIXWPBQMDRTAKZGFUHOS"
gamma = "FSOKANUERHMBTIYCWLQPZXVGJD"
I = "EKMFLGDQVZNTOWYHXUSPAIBRCJ"
II = "AJDKSIRUXBLHWTMCQGZNPYFVOE"
III = "BDFHJLCPRTXVZNYEIWGAKMUSQO"
IV = "ESOVPZJAYQUIRHXLNFTGKDCMWB"
V = "VZBRGITYUPSDNHLXAWMJQOFECK"
A = "EJMZALYXVBWFCRQUONTSPIKHGD"
B = "YRUHQSLDPXNGOKMIEBFZCWVJAT"
C = "FVPJIAOYEDRZXWGCTKUQSBNMHL"

# Additional rotors & reflectors used by the Enigma M4
VI = "JPGVOUMFYQBENHZRDKASXLICTW"
VII = "NZJHGRCXMYSWBOUFAIVLPEKQDT"
VIII = "FKQHTLXOCBJSPDZRAMEWNIUYGV"
UKWB = "ENKQAUYWJICOPBLMDXZVFTHRGS"
UKWC = "RDOBJNTKVEHMLFCWZAXGYIPSUQ"

# Rotors used in the British Typex machine
TYPEX_A = "MCYLPQUVRXGSAOWNBJEZDTFKHI"
TYPEX_B = "KHWENRCBISXJQGOFMAPVYZDLTU"
TYPEX_C = "BYPDZMGIKQCUSATREHOJNLFWXV"
TYPEX_D = "ZANJCGDLVHIXOBRPMSWQUKFYET"
TYPEX_E = "QXBGUTOVFCZPJIHSWERYNDAMLK"

notches_m4 = {"Z", "M"}
notches_typex = {"A", "E", "G", "M", "P", "T", "V"}
notches = {
    "I": {"Q"},
    "II": {"E"},
    "III": {"V"},
    "IV": {"J"},
    "V": {"Z"},
    "VI": notches_m4,
    "VII": notches_m4,
    "VIII": notches_m4,
    "TYPEX_A": notches_typex,
    "TYPEX_B": notches_typex,
    "TYPEX_C": notches_typex,
    "TYPEX_D": notches_typex,
    "TYPEX_E": notches_typex,
}

""" Create rotor mapping
Returns a dict with fields used for the internal wiring and notches
notch_indices is a set containing all the notch positions as an index relative to the alphabet
right_to_left and left_to_right are hashmaps, mapping input characters to output characters based on the internal wiring
"""
def create_rotor_mapping(letters, name):
    notch_indices = {alphabet_char_to_index[n] for n in notches[name]} if name in notches else None
    # Create hashmaps representing the internal wiring of the rotor
    # This allows faster look up of (on average) O(1) vs O(n) compared to searching through an array or string
    letters_map = {char: i for i, char in enumerate(letters)}
    alphabet_indices = {i: alphabet_char_to_index[char] for i, char in enumerate(letters)}
    letters_indices = {i: letters_map[char] for i, char in enumerate(alphabet)}
    values = {
        "notch_indices": notch_indices,
        "right_to_left": alphabet_indices,
        "left_to_right": letters_indices,
    }
    return values


rotors = {
    "Beta": create_rotor_mapping(beta, "Beta"),
    "Gamma": create_rotor_mapping(gamma, "Gamma"),
    "I": create_rotor_mapping(I, "I"),
    "II": create_rotor_mapping(II, "II"),
    "III": create_rotor_mapping(III, "III"),
    "IV": create_rotor_mapping(IV, "IV"),
    "V": create_rotor_mapping(V, "V"),
    "A": create_rotor_mapping(A, "A"),
    "B": create_rotor_mapping(B, "B"),
    "C": create_rotor_mapping(C, "C"),
    # Enigma M4
    "VI": create_rotor_mapping(VI, "VI"),
    "VII": create_rotor_mapping(VII, "VII"),
    "VIII": create_rotor_mapping(VIII, "VIII"),
    "UKWB": create_rotor_mapping(UKWB, "UKWB"),
    "UKWC": create_rotor_mapping(UKWC, "UKWC"),
    # Typex
    "TYPEX_A": create_rotor_mapping(TYPEX_A, "TYPEX_A"),
    "TYPEX_B": create_rotor_mapping(TYPEX_B, "TYPEX_B"),
    "TYPEX_C": create_rotor_mapping(TYPEX_C, "TYPEX_C"),
    "TYPEX_D": create_rotor_mapping(TYPEX_D, "TYPEX_D"),
    "TYPEX_E": create_rotor_mapping(TYPEX_E, "TYPEX_E"),
}

""" Confirm char is valid
Throws if a character is not considered valid, otherwise returns the character
Valid char must be uppercase from "A" to "Z"
"""
def confirm_char_is_valid(char):
    if len(char) != 1 or not char.isalpha():
        raise ValueError("invalid character, must be a single alpha character")
    if char != char.upper():
        raise ValueError("invalid character, must be uppercase")
    return char


""" Convert a setting
Takes a setting that is either "01" - "26" or "A" to "Z"
Return an equivalent index from 0 - 25
"""
def setting_to_int(setting):
    if setting.isalpha():
        return alphabet_char_to_index[setting.upper()]
    index = ring_settings.index(setting)
    if index > -1:
        return index
    raise ValueError(f"invalid setting provided: {setting}")


def rotor_from_name(name):
    return Rotor(name)


""" PlugLead
Accepts two characters, mapping such that for AB then A -> B and B -> A
If encode is called with a character in the provided mapping, the other character is returned
Otherwise the input character is returned
If the optional parameter is_typex is set to True, acts as a Typex plugboard
This allow A -> B and B -> C. The first char maps to the second, but not vice versa
"""
class PlugLead:
    def __init__(self, mapping, is_typex=False):
        if len(mapping) != 2:
            raise ValueError("invalid PlugLead length, mapping must consist of two characters")
        one, two = confirm_char_is_valid(mapping[0]), confirm_char_is_valid(mapping[1])
        if one == two:
            raise ValueError("invalid PlugLead characters, characters must not be equal")
        self.data = (one, two)
        self.is_typex = is_typex

    def encode(self, char):
        if char is self.data[0]:
            return self.data[1]
        if not self.is_typex and char is self.data[1]:
            return self.data[0]
        return char


""" Plugboard
Accepts up to 10 plug leads, with a method to run an input character into each plug lead
If is_typex is True, accepts up to 26 leads
"""
class Plugboard:
    def __init__(self, plug_leads=None, is_typex=False):
        self.plug_leads = []
        self.is_typex = is_typex
        if plug_leads:
            self._validate_plug_leads(plug_leads)
            self.plug_leads = plug_leads.copy()

    def _validate_plug_leads(self, plug_leads):
        max_plug_leads = 26 if self.is_typex else 10
        if len(plug_leads) > max_plug_leads:
            raise ValueError(f"too many plug leads passed to Plugboard, maximum of {max_plug_leads} permitted")
        if not all([isinstance(plug_lead, PlugLead) for plug_lead in plug_leads]):
            raise ValueError("invalid PlugLead provided to Plugboard")

    def add(self, plug_lead):
        self._validate_plug_leads([plug_lead])
        self.plug_leads.append(plug_lead)

    def encode(self, char):
        for _, plug_lead in enumerate(self.plug_leads):
            switch_char = plug_lead.encode(char)
            if char != switch_char:
                return switch_char
        return char


""" Rotor
A rotor in an Enigma M3, Enigma M4 or Typex machine
Has an offset and position, both used to affect the pin that is contacted for an input and output
May encode from right to left and left to right
Input and output characters are normalised relative to the alphabet
"""
class Rotor:
    def __init__(self, name, offset=0):
        if name not in rotors:
            raise ValueError("invalid rotor")
        self.name = name
        self.offset = offset
        self.position = 0
        self.rotor = rotors[name]

    def encode_right_to_left(self, char):
        return self._encode("right_to_left", char)

    def encode_left_to_right(self, char):
        return self._encode("left_to_right", char)

    def get_notch_indices(self):
        return self.rotor["notch_indices"]

    def set_pos(self, pos):
        self.position = pos

    def _encode(self, direction, char):
        char_index = alphabet_char_to_index[char]
        offset = (char_index - self.offset) % 26
        position = (offset + self.position) % 26

        char_index_wired_to = self.rotor[direction][position]
        offset = (char_index_wired_to + self.offset) % 26
        position = (offset - self.position) % 26
        return alphabet[position]


""" MultiRotor
Handles the stepper mechanism and the process of passing inputs through each rotor
Acts as a Typex machine if five rotors are provided
In this case the Typex stepping mechanism is used
"""
class MultiRotor:
    def __init__(self, rotors, reflector, ring_settings, initial_positions):
        split_rotors = rotors.split()
        if len(split_rotors) < 3:
            raise ValueError("must provide at least three rotors")
        configured_ring_settings = [setting_to_int(setting) for setting in ring_settings.split()]
        configured_initial_positions = [setting_to_int(setting) for setting in initial_positions.split()]
        configured_rotors = [Rotor(name, configured_ring_settings[i]) for i, name in enumerate(split_rotors)]
        configured_reflector = Rotor(reflector, 0)
        self.ring_settings = configured_ring_settings
        self.positions = configured_initial_positions
        self.rotors = configured_rotors
        self.reflector = configured_reflector
        self.emulate_typex = len(self.positions) == 5

    def encode(self, char):
        if self.is_typex():
            self._rotate_positions_typex()
        else:
            self._rotate_positions_enigma()
        value = self._encode_right_to_left(char)
        value = self._encode_reflector(value)
        value = self._encode_left_to_right(value)
        return value

    def is_typex(self):
        return self.emulate_typex

    def _encode_right_to_left(self, char):
        for i in range(len(self.rotors) - 1, -1, -1):
            rotor = self.rotors[i]
            rotor.set_pos(self.positions[i])
            char = rotor.encode_right_to_left(char)
        return char

    def _encode_left_to_right(self, char):
        for i in range(len(self.rotors)):
            rotor = self.rotors[i]
            char = rotor.encode_left_to_right(char)
        return char

    def _encode_reflector(self, char):
        return self.reflector.encode_right_to_left(char)

    def _increment_rotor_value(self, value):
        return value + 1 if value < 25 else 0

    """ Rotate Typex rotors
    Rotors the first three rotors in the same way as the last three on an Enigma machine
    """
    def _rotate_positions_typex(self):
        first_notch = self.rotors[2].get_notch_indices()
        second_notch = self.rotors[1].get_notch_indices()
        if second_notch and self.positions[1] in second_notch:
            self.positions[1] = self._increment_rotor_value(self.positions[1])
            self.positions[0] = self._increment_rotor_value(self.positions[0])
        elif first_notch and self.positions[2] in first_notch:
            self.positions[1] = self._increment_rotor_value(self.positions[1])
        self.positions[2] = self._increment_rotor_value(self.positions[2])

    """ Rotate Enigma rotors
    The rightmost rotor rotates on every key press
    The second rightmost rotor rotates if the first is on its notch, or if it itself is on its notch
    The third rightmost rotor rotates if the second is on its notch
    The fourth rotor never rotates
    """
    def _rotate_positions_enigma(self):
        first_notch = self.rotors[-1].get_notch_indices()
        second_notch = self.rotors[-2].get_notch_indices()
        if second_notch and self.positions[-2] in second_notch:
            self.positions[-2] = self._increment_rotor_value(self.positions[-2])
            self.positions[-3] = self._increment_rotor_value(self.positions[-3])
        elif first_notch and self.positions[-1] in first_notch:
            self.positions[-2] = self._increment_rotor_value(self.positions[-2])
        self.positions[-1] = self._increment_rotor_value(self.positions[-1])


""" MultiEnigma ("Multi" of MultiEnigma, coz uses MultiRotor. Just a matter of name) / Typex machine
Validates an input character, then passes it through the Plugboard and MultiRotor
If the MultiRotor is in the Typex configuration, it will reverse the mapping of input and output characters
This allows the machine to act as an Enigma M3, Enigma M4 and Typex machine
"""
class MultiEnigma: # Typex uses MultiRotor
    def __init__(self, multi_rotor, plug_board):
        self.multi_rotor = multi_rotor
        self.plug_board = plug_board
        self.is_typex = multi_rotor.is_typex()

    def encode_decode(self, message):
        result = ""
        for char in message:
            if self.is_typex:
                char = self._change_typex_char(confirm_char_is_valid(char))
            char = self.plug_board.encode(char)
            char = self.multi_rotor.encode(char)
            char = self.plug_board.encode(char)
            if self.is_typex:
                char = self._change_typex_char(char)
            result += char
        return result

    """Map Typex input character
    The British Typex machine had an input plugboard that acted counter clockwise
    This means that A=A, B=Z, C=Y and so on up until Z=B
    """
    def _change_typex_char(self, char):
        counter_clockwise_chars = "AZYXWVUTSRQPONMLKJIHGFEDCB"
        return counter_clockwise_chars[alphabet_char_to_index[char]]


if __name__ == "__main__":
    print("\nADVANCED WORK\n")

    """ #1 M4 Functionality
    """
    t_pb = "BQ CR DI EJ KW MT OS PX UZ GH"
    t_rotors = "Beta VI VII VIII"
    t_reflector = "UKWB"
    t_ring_settings = "01 01 01 01"
    t_initial_positions = "A A A A"
    print("#1 M4 Functionality\n")
    print("This implementation can act as an Enigma M4")
    print("Here we'll create an enigma machine with the following settings:\n")
    print(f'Rotors: {t_rotors}')
    print(f'Reflector: {t_reflector}')
    print(f'Ring settings: {t_ring_settings}')
    print(f'Initial positions: {t_initial_positions}')
    print(f'Plugboard: {t_pb}')
    # Encode a message
    plugboard = Plugboard([PlugLead(mapping) for mapping in t_pb.split()])
    multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)
    multienigma = MultiEnigma(multirotor, plugboard)
    message = "THISISASECRETMESSAGEFROMTHEENIGMAMFOUR"
    encoded = multienigma.encode_decode(message)
    # Now decode it with the same settings as before
    multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)
    multienigma = MultiEnigma(multirotor, plugboard)
    decoded = multienigma.encode_decode(encoded)
    print(f"We'll take the message: {message}")
    print(f"This encodes into: {encoded}")
    print(f"And running this through the decoder, we again get: {decoded}")

    print("\n\n")

    """ #2 Typex Machine Functionality
    """
    t_pb = ""
    t_rotors = "TYPEX_A TYPEX_B TYPEX_C TYPEX_D TYPEX_E" # # of rotors = 5, so it's a TypeX
    t_reflector = "B"
    t_ring_settings = "01 01 01 01 01"
    t_initial_positions = "A A A A A"
    print("#2 Typex Machine Functionality\n")
    print("This implementation can emulate a British Typex machine")
    print("The Typex could emulate an enigma machine")
    print("It could print characters and was cryptographically superior")
    print("And had 5 reversible rotors, the first 3 stepping as the Enigma's did")
    print("The rotor wirings remain classified, but had anywhere from 3-9 notches")
    print("The input wiring was reversed, so instead of A=A, B=B, C=C it followed A=A, B=Z, C=X")
    print("The plugboard could also map to any character, such that it was possible to have A-B, B-C")
    print("Here we'll create a Typex machine with the following settings:\n")
    print(f'Rotors: {t_rotors}')
    print(f'Reflector: {t_reflector}')
    print(f'Ring settings: {t_ring_settings}')
    print(f'Initial positions: {t_initial_positions}')
    print(f'Plugboard: {t_pb}')
    # Encode a message
    plugboard = Plugboard([PlugLead(mapping) for mapping in t_pb.split()])
    multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)
    multienigma = MultiEnigma(multirotor, plugboard)
    message = "THEBRITISHTYPEXCOULDDECRYPTRATHERQUICKLY"
    encoded = multienigma.encode_decode(message)
    # Now decode it with the same settings as before
    multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)
    multi = MultiEnigma(multirotor, plugboard)
    decoded = multienigma.encode_decode(encoded)
    print(f"We'll take the message: {message}")
    print(f"This encodes into: {encoded}")
    print(f"And running this through the decoder, we again get: {decoded}")

    print("\n\n")

    """ #3 Use Multiple Rotors
    """
    t_pb = "PC XZ FM QA ST NB HY OR EV IU"
    t_rotors = "Beta Gamma I II III IV V"
    t_reflector = "A"
    t_ring_settings = "01 02 26 09 04 20 13"
    t_initial_positions = "B D F A V E P"
    print("#3 Use more than four rotors\n")
    print("This machine can use any number of rotors")
    print("Here we'll create an enigma machine with the following settings:\n")
    print(f'Rotors: {t_rotors}')
    print(f'Reflector: {t_reflector}')
    print(f'Ring settings: {t_ring_settings}')
    print(f'Initial positions: {t_initial_positions}')
    print(f'Plugboard: {t_pb}')
    # Encode a message
    plugboard = Plugboard([PlugLead(mapping) for mapping in t_pb.split()])
    multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)
    multienigma = MultiEnigma(multirotor, plugboard)
    message = "THISENIGMAMACHINECANUSEANYNUMBEROFROTORS"
    encoded = multienigma.encode_decode(message)
    # Now decode it with the same settings as before
    multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)
    multienigma = MultiEnigma(multirotor, plugboard)
    decoded = multienigma.encode_decode(encoded)
    print(f"We'll take the message: {message}")
    print(f"This encodes into: {encoded}")
    print(f"And running this through the decoder, we again get: {decoded}")

    pass


# python enigma.py