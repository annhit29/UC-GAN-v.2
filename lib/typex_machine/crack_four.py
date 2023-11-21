from enigma import MultiRotor, Plugboard, PlugLead, Enigma, alphabet

code = "SDNTVTPHRBNWTLMZTQKZGADDQYPFNHBPNHCQGBGMZPZLUAVGDQVYRBFYYEIXQWVTHXGNW"
crib = "TUTOR"

rotors = "V III IV"
reflector = "A"
ring_settings = "24 12 10"
initial_positions = "S W U"
plugboard_pairs = "WP RJ VF HN CG BS A? I?"
known_plugboard_pairs = "WP RJ VF HN CG BS"

alphabet_set = set(alphabet)
known_set = set([letter for letter in plugboard_pairs if letter.isalpha()])

possible_plugboard_pairs = []
for letter in alphabet:
    if letter not in known_set:
        a = f'A{letter}'
        for another_letter in alphabet:
            if another_letter not in known_set and another_letter is not letter:
                i = f'I{another_letter}'
                possible_plugboard_pairs.append(f'{known_plugboard_pairs} {a} {i}')

if __name__ == '__main__':
    for plugboard_pairs in possible_plugboard_pairs:
        multi_rotor = MultiRotor(rotors, reflector, ring_settings, initial_positions)
        plug_board = Plugboard([PlugLead(plug_lead) for plug_lead in plugboard_pairs.split()])
        enigma = Enigma(multi_rotor, plug_board)
        decoded = enigma.encode_decode(code)
        if crib in decoded:
            print(f'Missing plugboard pairs: {plugboard_pairs}')
            print(f'Plaintext: {decoded}')
