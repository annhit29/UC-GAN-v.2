from enigma import MultiRotor, Plugboard, PlugLead, Enigma, alphabet, alphabet_char_to_index, create_rotor_mapping, A, B, C

code = "HWREISXLGTTBYVXRCWWJAKZDTVZWKBDJPVQYNEQIOTIFX"
# A few of the most popular social media sites
cribs = [
    "FACEBOOK",
    "TWITTER",
    "INSTAGRAM",
    "LINKEDIN",
]
rotors = "V II IV"
ring_settings = "06 18 07"
initial_positions = "A J L"
plugboard_pairs = "UG IE PO NX WT"

""" Perform a swap of a wiring on a reflector
Generates all possible permutations for a given swap of one wire to another wire
Returns a list with the wiring and a list of changes that have been performed
"""
def swap_wires(letters, changes=None):
    result = []
    for mapped_index in range(len(letters) - 1 // 2):
        for next_mapped_index in range(mapped_index + 1, len(letters) // 2):
            top_left = mapped_index # A -> C
            top_right = next_mapped_index # B -> D
            bottom_left = alphabet_char_to_index[letters[top_left]] # C -> A
            bottom_right = alphabet_char_to_index[letters[top_right]] # D -> B
            if len({top_left, top_right, bottom_left, bottom_right}) != 4:
                continue
            # There are only two possible swaps for AB CD, AC BD and AD BC
            perm = [
                [bottom_right, top_right, bottom_left, top_left],
                [top_right, bottom_right, top_left, bottom_left],
            ]
            for p in perm:
                l = list(letters)
                old_tl, old_bl, old_tr, old_br = l[top_left], l[bottom_left], l[top_right], l[bottom_right]
                l[top_left], l[bottom_left], l[top_right], l[bottom_right] = alphabet[p[0]], alphabet[p[1]], alphabet[p[2]], alphabet[p[3]]
                joined = ''.join(l)
                new_changes = [
                    [old_tl, old_bl, old_tr, old_br],
                    [alphabet[p[0]], alphabet[p[1]], alphabet[p[2]], alphabet[p[3]]],
                    joined,
                ]
                all_changes = changes.copy() if changes else []
                all_changes.append(new_changes)
                with_changes = {
                    "wiring": joined,
                    "changes": all_changes,
                }
                result.append(with_changes)
    return result


def gen_reflector_permutations(letters):
    result = []
    checked = set()
    for swapped_once in swap_wires(letters):
        for swapped_twice in swap_wires(swapped_once["wiring"], swapped_once["changes"]):
            if swapped_twice["wiring"] not in checked:
                val = {
                    "rotor_mapping": create_rotor_mapping(swapped_twice["wiring"], ""),
                    "changes": swapped_twice["changes"],
                }
                result.append(val)
            checked.add(swapped_twice["wiring"])
    return result


def run_crack(name, wiring):
    plug_board = Plugboard([PlugLead(plug_lead) for plug_lead in plugboard_pairs.split()])

    print(f'Attemping to crack permutations of rotor {name}')
    rotor_permutations = gen_reflector_permutations(wiring)
    print(f'Total wiring possibilities for rotor {name}: {len(rotor_permutations)}, beginning crack...')
    for new_rotor in rotor_permutations:
        multi_rotor = MultiRotor(rotors, "A", ring_settings, initial_positions)
        multi_rotor.reflector.rotor = new_rotor["rotor_mapping"]
        enigma = Enigma(multi_rotor, plug_board)
        decoded = enigma.encode_decode(code)
        for crib in cribs:
            if crib in decoded:
                print(f'Crib: {crib}')
                print(f'Missing reflector: {name}')
                reflector_wiring = ''.join([alphabet[new_rotor["rotor_mapping"]["right_to_left"][index]] for index in range(26)])
                print(f'Original reflector wiring: {wiring}')
                print(f'New reflector wiring: {reflector_wiring}')
                for change in new_rotor["changes"]:
                    from_val = ''.join(change[0])
                    to_val = ''.join(change[1])
                    new_reflector = change[2]
                    print(f'Reflector swap from {from_val} to {to_val}')
                    print(f'Reflector setting is now {new_reflector}')
                print(f'Plaintext: {decoded}')
                sys.exit()

if __name__ == '__main__':
    rotor_configs = [
        {
            "rotor": "A",
            "wiring": A,
        },
        {
            "rotor": "B",
            "wiring": B,
        },
        {
            "rotor": "C",
            "wiring": C,
        },
    ]
    for rotor_config in rotor_configs:
        run_crack(rotor_config["rotor"], rotor_config["wiring"])
