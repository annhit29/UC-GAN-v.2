from typex.typex import TypeX
from typex.rotor import Rotor
from typex.stator import Stator
from typex.reflector import Reflector
import unittest2


class TypeXTest(unittest2.TestCase):

    def test_constructor(self):
        """Check that TypeX constructor works properly"""
        self.assertTrue(isinstance(TypeX(), TypeX))
        self.assertTrue(isinstance(TypeX(encryptors=[]), TypeX))

    def test_if_encryptors_are_valid(self):
        """Check that TypeX constructor works only with valid encryptors"""
        self.assertTrue(isinstance(TypeX(encryptors=[
            Stator(wiring='ABCDEFOPQGHIJKLMNRSTXYZUVW', initial_position=3),
            Rotor(wiring='QWERTYUIOPASDFGHJKLZXCVBNM', initial_position=4,
                  notchings=[0, 5, 12, 16, 25])]),
            TypeX))

        with self.assertRaises(TypeError):
            TypeX(encryptors='a flamingo wrapped in tinsel')

        # Pass in encryptors that include a non-Encryptor
        with self.assertRaises(TypeError):
            TypeX(encryptors=[Stator(wiring='ABCDEFOPQGHIJKLMNRSTXYZUVW',
                                     initial_position=3), 'flamingo'])

        # Pass in encryptors that include a Reflector
        with self.assertRaises(TypeError):
            TypeX(encryptors=[Stator(wiring='ABCDEFOPQGHIJKLMNRSTXYZUVW',
                                     initial_position=3), Reflector()])

        # Pass in encryptors that are out of order
        with self.assertRaises(ValueError):
            TypeX(encryptors=[
                Rotor(wiring='QWERTYUIOPASDFGHJKLZXCVBNM', initial_position=4,
                      notchings=[0, 5, 12, 16, 25]),
                Stator(wiring='ABCDEFOPQGHIJKLMNRSTXYZUVW', initial_position=3)
                ])


if __name__ == '__main__':
    unittest2.main()
