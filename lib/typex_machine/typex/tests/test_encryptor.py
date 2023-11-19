from typex.encryptor import Encryptor
import unittest2


class EncryptorTest(unittest2.TestCase):
    def test_constructor(self):
        """Check if Encryptor constructor works"""
        self.assertTrue(isinstance(Encryptor(), Encryptor))
        self.assertTrue(isinstance(Encryptor(
            wiring='ZAQXSWCDEVFRBGTNHYMJUKILOP'), Encryptor))
        self.assertTrue(isinstance(Encryptor(initial_position=25), Encryptor))

    def test_wiring_type(self):
        """Check that Encryptor constructor cares about wiring type"""
        with self.assertRaises(TypeError):
            Encryptor(wiring=42)

    def test_wiring_length(self):
        """Check that Encryptor constructor cares about wiring length"""
        with self.assertRaises(ValueError):
            Encryptor(wiring='ABC')

        with self.assertRaises(ValueError):
            Encryptor(wiring='ABCDEFGHIJKLMNOPQRSTUVWXYZa')

    def test_wiring_is_alphabetic(self):
        """Check that Encryptor constructor works only """
        """if wiring contains alphabet"""
        self.assertTrue(isinstance(Encryptor(
            wiring='abcdefghijklmnopqrstuvwxyz'), Encryptor))

        with self.assertRaises(ValueError):
            Encryptor(wiring='ABCDEFGHIJKLMNOPQRSTUVWX.!')

    def test_wiring_characters_are_unique(self):
        """Check that Encryptor constructor works only if """
        """each wiring character is unique"""
        self.assertTrue(isinstance(Encryptor(
            wiring='ABCDEFGHIJKLMNOPQRSTUVWXYZ'), Encryptor))

        with self.assertRaises(ValueError):
            Encryptor(wiring='ABCDEFGHIJKLMABCDEFGHIJKLMN')

    def test_initial_position_is_valid(self):
        """Check that Encryptor constructor works only if """
        """initial_position is valid"""
        self.assertTrue(isinstance(Encryptor(initial_position=7), Encryptor))

        with self.assertRaises(ValueError):
            Encryptor(initial_position=-1)

        with self.assertRaises(TypeError):
            Encryptor(initial_position='a pencil')

        with self.assertRaises(ValueError):
            Encryptor(initial_position=27)

    def test_encrypt_character(self):
        """Check that enctypt_character method of """
        """Encryptor works properly"""
        e = Encryptor(wiring=Encryptor.ALPHABET, initial_position=1)

        self.assertEqual(e.encrypt_character('A'), 'B')
        self.assertEqual(e.encrypt_character('Q'), 'R')

        with self.assertRaises(TypeError):
            e.encrypt_character(-3)

        with self.assertRaises(ValueError):
            e.encrypt_character('an aardvark')

    def test_reverse_encrypt_character(self):
        """Checks that reverse_encrypt_character method """
        """of Encryptor works properly"""
        e = Encryptor(wiring=Encryptor.ALPHABET, initial_position=1)

        self.assertEqual(e.reverse_encrypt_character('B'), 'A')
        self.assertEqual(e.reverse_encrypt_character('R'), 'Q')

        with self.assertRaises(TypeError):
            e.reverse_encrypt_character(-3)

        with self.assertRaises(ValueError):
            e.reverse_encrypt_character('flamingo')


if __name__ == '__main__':
    unittest2.main()
