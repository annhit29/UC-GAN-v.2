from enigma import MultiRotor, Plugboard, PlugLead, Enigma, alphabet

from itertools import permutations

code = "CMFSUPKNCBMUYEQVVDYKLRQZTPUFHSWWAKTUGXMPAMYAFITXIJKMH"
crib = "UNIVERSITY"

rotors = "Beta I III"
reflector = "B"
ring_settings = "23 02 10"
plugboard_pairs = "VH PT ZG BJ EY FS"

if __name__ == '__main__':
    for initial_positions in permutations(alphabet, 3):
        multi_rotor = MultiRotor(rotors, reflector, ring_settings, ' '.join(initial_positions))
        plug_board = Plugboard([PlugLead(plug_lead) for plug_lead in plugboard_pairs.split()])
        enigma = Enigma(multi_rotor, plug_board)
        decoded = enigma.encode_decode(code)
        if crib in decoded:
            print(f'Missing initial positions: {" ".join(initial_positions)}')
            print(f'Plaintext: {decoded}')
