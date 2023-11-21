from enigma import MultiRotor, Plugboard, PlugLead, Enigma

code = "DMEXBMKYCVPNQBEDHXVPZGKMTFFBJRPJTLHLCHOTKOYXGGHZ"
crib = "SECRETS"

rotors = "Beta Gamma V"
ring_settings = "04 02 14"
initial_positions = "M J M"
plugboard_pairs = "KI XN FL"

if __name__ == '__main__':
    reflectors = ["A", "B", "C"]
    for reflector in reflectors:
        multi_rotor = MultiRotor(rotors, reflector, ring_settings, initial_positions)
        plug_board = Plugboard([PlugLead(plug_lead) for plug_lead in plugboard_pairs.split()])
        enigma = Enigma(multi_rotor, plug_board)
        decoded = enigma.encode_decode(code)
        if crib in decoded:
            print(f'Missing reflector: {reflector}')
            print(f'Plaintext: {decoded}')
