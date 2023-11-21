from unittest import TestCase

from enigma import *

class TestPlugLead(TestCase):
    def test_encode(self):
        lead = PlugLead("AG")
        self.assertEqual(lead.encode("A"), "G")
        self.assertEqual(lead.encode("D"), "D")

        lead = PlugLead("DA")
        self.assertEqual(lead.encode("A"), "D")
        self.assertEqual(lead.encode("D"), "A")

        lead = PlugLead("DA", True)
        self.assertEqual(lead.encode("D"), "A")
        self.assertEqual(lead.encode("A"), "A")


class TestPlugboard(TestCase):
    def test_encode(self):
        plugboard = Plugboard()
        plugboard.add(PlugLead("SZ"))
        plugboard.add(PlugLead("GT"))
        plugboard.add(PlugLead("DV"))
        plugboard.add(PlugLead("KU"))
        self.assertEqual(plugboard.encode("K"), "U")
        self.assertEqual(plugboard.encode("U"), "K")
        self.assertEqual(plugboard.encode("A"), "A")
        self.assertEqual(plugboard.encode("Z"), "S")
        self.assertEqual(plugboard.encode("U"), "K")

    def test_encode_typex(self):
        plugboard = Plugboard([], True)
        plugboard.add(PlugLead("AB", True))
        plugboard.add(PlugLead("BC", True))
        plugboard.add(PlugLead("CD", True))
        plugboard.add(PlugLead("ZT", True))
        self.assertEqual(plugboard.encode("A"), "B")
        self.assertEqual(plugboard.encode("B"), "C")
        self.assertEqual(plugboard.encode("C"), "D")
        self.assertEqual(plugboard.encode("Z"), "T")


class TestMultiRotor(TestCase):
    def test_encode_simple(self):
        multi_rotor = MultiRotor("I II III", "B", "01 01 01", "A A B")
        value = "A"
        expect = "D"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)

    def test_encode_simple_one(self):
        multi_rotor = MultiRotor("I II III", "B", "01 01 01", "A A C")
        value = "A"
        expect = "Z"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)

    def test_encode_simple_two(self):
        multi_rotor = MultiRotor("I II III", "B", "01 01 01", "A A Z")
        value = "A"
        expect = "U"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)

    def test_encode_simple_three(self):
        multi_rotor = MultiRotor("I II III", "B", "01 01 01", "A A A")
        value = "A"
        expect = "B"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)

    def test_encode_simple_four(self):
        multi_rotor = MultiRotor("I II III", "B", "01 01 01", "Q E V")
        value = "A"
        expect = "L"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)

    def test_encode_simple_five(self):
        multi_rotor = MultiRotor("IV V Beta", "B", "14 09 24", "A A A")
        value = "H"
        expect = "Y"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)

    def test_encode_simple_six(self):
        multi_rotor = MultiRotor("I II III IV", "C", "07 11 15 19", "Q E V Z")
        value = "Z"
        expect = "V"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)

    def test_encode_simple_offhand(self):
        multi_rotor = MultiRotor("I II III IV", "C", "07 11 15 19", "Q E V Z")
        value = "Z"
        expect = "V"
        encoded = multi_rotor.encode(value)
        self.assertEqual(expect, encoded)


class TestRotor(TestCase):
    def test_encode_right_left(self):
        rotor = Rotor("I", 0)
        char = rotor.encode_right_to_left("A")
        expect = "E"
        self.assertEqual(expect, char)

    def test_encode_right_left_one(self):
        rotor = Rotor("I", 1)
        char = rotor.encode_right_to_left("A")
        expect = "K"
        self.assertEqual(expect, char)

    def test_encode_right_left_iii(self):
        rotor = Rotor("III", 0)
        char = rotor.encode_right_to_left("A")
        expect = "B"
        self.assertEqual(expect, char)

    def test_encode_right_left_iii_two(self):
        rotor = Rotor("III", 0)
        char = rotor.encode_left_to_right("A")
        expect = "T"
        self.assertEqual(expect, char)

    def test_encode_right_left_iv(self):
        rotor = Rotor("IV", 0)
        char = rotor.encode_right_to_left("Z")
        expect = "B"
        self.assertEqual(expect, char)

    def test_encode_right_left_iv_two(self):
        rotor = Rotor("IV", 0)
        char = rotor.encode_left_to_right("Z")
        expect = "F"
        self.assertEqual(expect, char)


class TestEnigma(TestCase):
    def test_encode_decode_one(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "HL MO AJ CX BZ SR NI YW DG PK".split()])
        multirotor = MultiRotor("I II III", "B", "01 01 01", "A A Z")
        enigma = Enigma(multirotor, plugboard)
        message = "HELLOWORLD"
        expect = "RFKTMBXVVW"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_two(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "PC XZ FM QA ST NB HY OR EV IU".split()])
        multirotor = MultiRotor("IV V Beta I", "A", "18 24 03 05", "E Z G P")
        enigma = Enigma(multirotor, plugboard)
        message = "BUPXWJCDPFASXBDHLBBIBSRNWCSZXQOLBNXYAXVHOGCUUIBCVMPUZYUUKHI"
        expect = "CONGRATULATIONSONPRODUCINGYOURWORKINGENIGMAMACHINESIMULATOR"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_M4(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "BQ CR DI EJ KW MT OS PX UZ GH".split()])
        multirotor = MultiRotor("Beta VI VII VIII", "UKWB", "01 01 01 01", "A A A A")
        enigma = Enigma(multirotor, plugboard)
        message = "THISISASECRETMESSAGEFROMTHEENIGMAMFOUR"
        expect = "VTLFVMTKLMJTXXQGGQNGBDZOXFIHJLHXHYQWPG"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_typex(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "".split()])
        multirotor = MultiRotor("TYPEX_A TYPEX_B TYPEX_C TYPEX_D TYPEX_E", "B", "01 01 01 01 01", "A A A A A")
        enigma = Enigma(multirotor, plugboard)
        message = "THEBRITISHTYPEXCOULDDECRYPTRATHERQUICKLY"
        expect = "DOFXPECMZSIZCIAEWZXTBJKPGLLXBDARXZIAFXZZ"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_random_1(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "".split()])
        multirotor = MultiRotor("II Gamma Beta", "B", "01 01 01", "E M Y")
        enigma = Enigma(multirotor, plugboard)
        message = "firsttrialthisissparta".upper()
        expect = "DDBLAZQEMQUUJRLEUCHNME"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_random_2(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "".split()])
        multirotor = MultiRotor("Gamma IV Beta", "B", "01 01 01", "E M Y")
        enigma = Enigma(multirotor, plugboard)
        message = "secondtrialthisissparta".upper()
        expect = "IWBXTOPESMGSILNLVXLEYPZ"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_random_3(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "".split()])
        multirotor = MultiRotor("Gamma II Beta", "B", "01 01 01", "E M Y")
        enigma = Enigma(multirotor, plugboard)
        message = "thirdtrialthisissparta".upper()
        expect = "BJPPLQLVRFZTGDUNQVSLWB"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_random_4(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "".split()])
        multirotor = MultiRotor("Gamma IV II", "B", "01 01 01", "E M Y")
        enigma = Enigma(multirotor, plugboard)
        message = "fourthtrialthisissparta".upper()
        expect = "TWAMUCPWVOUFLVOEHZKHUSH"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_random_3_long(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "".split()])
        multirotor = MultiRotor("Gamma II Beta", "B", "01 01 01", "E M Y")
        enigma = Enigma(multirotor, plugboard)
        message = "thirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisisspartathirdtrialthisissparta".upper()
        expect = "BJPPLQLVRFZTGDUNQVSLWBUOXCLMTCFENLQKXJWLBALJHNCKMCSPOXRFOKPZJUWCPCVLDKEQSGJBYMCRPORFJAVBZTGGKIYQSRWCXMXOVBNPUDNLQZJHQNBHLGLUCLYZLCBNRFOWMJAOWVPJUNDZAXNKUUYMCSWZSGJHVYXJGDYGKKHOWCXIQNWCNDUBPZQKCKHGVRLGLLYQLRLGBJPOOKCDRZZGPJUAICJDNSUOXOCRBRFWNRVYXCJOQTKFHNCLXMRIOSRQUBPAHEEMHPVLDZLUROJIYKBJPPLQLVRFZTGDUNQVSLWBUOXCLMTCFENLQKXJWLBALJHNCKMCSPOXRFOKPZJUWCPCVLDKEQSGJBYMCRPORFJAVBZTGGKIYQSRWCXMXOVBNPUD"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_additional_one(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "AB CD EF GH IJ KL MN OP QR ST".split()])
        multirotor = MultiRotor("V VI III II", "C", "05 03 21 19", "Z N F K")
        enigma = Enigma(multirotor, plugboard)
        message = "ALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUTALARGEINPUT"
        expect = "WVWUSNNAVDITHRLXIRAYOLIIBQJKKAMLVWNFCVCWKQWDLIJNXUTDSTCOKVANFTVLKCQFBWFSDRZCADFXXHYJDZKUUUWPDXDABEHJJNCVUQFSOZKEFEVJHIRXREYNSPKFUZHYJUELYDOFGIR"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_additional_two(self):
        plugboard = Plugboard()
        multirotor = MultiRotor("Beta III II I", "A", "02 04 06 08", "A C E G")
        enigma = Enigma(multirotor, plugboard)
        message = "NOPLUGBOARDISUSEDINTHISEXAMPLE"
        expect = "YTGELWEHYEEHNIFAUGIUYVJOWXCLNG"
        self.assertEqual(expect, enigma.encode_decode(message))

    def test_encode_decode_additional_three(self):
        plugboard = Plugboard([PlugLead(mapping) for mapping in "TY RF AL NQ".split()])
        multirotor = MultiRotor("V III Gamma", "B", "01 01 01", "A A A")
        enigma = Enigma(multirotor, plugboard)
        message = "THEGAMMAANDBETAROTORSLACKANOTCH"
        expect = "ZFVFPJXLMGHERXKEUOZFJVRACXUKMXF"
        self.assertEqual(expect, enigma.encode_decode(message))
