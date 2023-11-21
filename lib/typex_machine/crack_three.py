from enigma import MultiRotor, Plugboard, PlugLead, Enigma

from itertools import permutations

code = "ABSKJAKKMRITTNYURBJFWQGRSGNNYJSDRYLAPQWIAGKJYEPCTAGDCTHLCDRZRFZHKNRSDLNPFPEBVESHPY"
crib = "THOUSANDS"

rotors = permutations(["Beta", "Gamma", "II", "IV"], 3)
reflectors = ["A", "B", "C"]
ring_setting_possible_values = [f'0{i}' if i < 10 else str(i) for i in range(1, 27) if all([int(n) % 2 == 0 for n in str(i)])]
ring_settings = permutations(ring_setting_possible_values, 3)
initial_positions = "E M Y"
plugboard_pairs = "FH TS BE UQ KD AL"

if __name__ == '__main__':
    for rotor_set in rotors:
        ring_settings = permutations(ring_setting_possible_values, 3)
        for current_ring_setting in ring_settings:
            for reflector in reflectors:
                multi_rotor = MultiRotor(' '.join(rotor_set), reflector, ' '.join(current_ring_setting), initial_positions)
                plug_board = Plugboard([PlugLead(plug_lead) for plug_lead in plugboard_pairs.split()])
                enigma = Enigma(multi_rotor, plug_board)
                decoded = enigma.encode_decode(code)
                if crib in decoded:
                    print(f'Missing rotors: {" ".join(rotor_set)}')
                    print(f'Missing reflector: {reflector}')
                    print(f'Missing ring_setting: {" ".join(current_ring_setting)}')
                    print(f'Plaintext: {decoded}')
