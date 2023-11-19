TypeX in Python
===============

This repository implements a `TypeX
machine <https://en.wikipedia.org/wiki/Typex>`__ in Python 3.

Usage
=====

::

    # Start with a message you want to encrypt
    $ cat message.txt
    The wind is in the buffalo.

    # Feed it into typex.py, either through stdin or as a file
    $ typex message.txt
    YZWIHBSOVWUYEIQSMCMWOYDWRC

    # Decrypt the output. Note that this implementation currently exhibits
    the Enigma/TypeX behavior that spaces are replaced with 'X'.
    $ typex message.txt  | typex
    THEXWINDXISXINXTHEXBUFFALO

Background
==========

A TypeX machine is a `rotor
machine <https://en.wikipedia.org/wiki/Rotor_machine>`__ of English
design that improved upon the famous `Enigma
machine <https://en.wikipedia.org/wiki/Enigma_machine>`__ by introducing
several novel features that made the machine easier to use and its
encrypted messages harder to break: - A pair of static rotors called
`stators <stator.py>`__ to augment the Enigma
`plugboard <https://en.wikipedia.org/wiki/Enigma_machine#Plugboard>`__.
Stators have a similar role to the plugboard, without the cryptographic
weakness of being reciprocal. - On the non-static rotors
(`Rotor <rotor.py>`__), support for variable "notching" that determines
which rotation positions of a given rotor will trigger a one-step
rotation in the rotor to its left. In Enigma, a rotor would rotate one
step only when the rotor to its right completed a full revolution;
notchings were added as a mechanism to sprinkle in extra degrees of
cryptographic permutation to the encryption process. - A built-in
printer, emulated here by ``print()``. In the original Enigma machine,
output was read from a board of lights representing individual letters
and transcribed by hand, requiring two simultaneous operators for
practical operation.

The following image shows the input letter *Q* being encrypted to the
output letter *E* through a TypeX machine circuit:

.. figure:: http://www.cryptomuseum.com/crypto/uk/typex/img/circuit_typex.png
   :alt: TypeX wiring diagram

   TypeX wiring diagram

For reasons that we will discuss below (see Theory), this same wiring
diagram with all of the rotors in the same position would necessarily
map the input letter *E* to the letter *Q*.

*Note: this diagram shows a plugboard between the operator's typewriter
and the rest of the encryption circuit. This project does not currently
implement a plugboard (see TODO).*

Theory
======

Enigma-style machines operate by sending a character c through a series
of character substitution transformations T1, T2, ..., TN, then through
a reflection R, then through the inverse transformations TN-1, ...,
T2-1, T1-1 as follows:

Encryption function E(c) = T1-1(T2-1(...(TN-1(R(TN(...(T2(T1(c)))))))))

The person receiving the message enters each encrypted character into
corresponding decryption function D(c) *that is exactly the same as the
encryption function E(c)*. In other words, to decrypt the message,
simply run it through the same encryption machine! For reasons that we
will see below, this process produces the original input message.

How is it possible for the person receiving the message able to recreate
these transformation functions identically? By setting all of the
plugboard, stator, rotor, and reflector settings to be identical to the
machine state that existed at the *beginning* of the sender's encryption
process. Since the person decrypting the message has an identical
configuration of stators, rotors, rotor notchings, and reflector, *the
transformation functions TN themselves will mutate in exactly the same
way over time as the rotor rotate during each step of the decryption
process*.

Thus at every step in the decryption process, we will we have

Decrypted output of E(c) = D(E(c)) =
T1-1(T2-1(...(TN-1(R(TN(...(T2(T1(**E(c)**)))))))))

But taking the very innermost portion of that expression T1(E(c)) and
replacing it with the formula for E(c), we would get

T1-1(T1-1(T2-1(....(TN-1(R(TN(...(T2(T1(c))))))))))

and the first two terms T1 and T1-1 collapse into an identity function.
This continues until we arrive at the term R(R(c)), which means "a
reflection of a reflection of c". However, given the reciprocal natural
of a reflector (A<->Z, B<->Y, etc.), it is also clear that R(R(c)) is
just an identity function as well. Continuing through the rest of the
transformation and inverse-transformation functions, everything cancels
to identity functions and we arrive at

D(E(c)) = c

In other words, running the encrypted text through the same TypeX (or
Enigma machine) yields the original text.

Two simple cases of this theory are worth noting:

-  If the TypeX or Enigma machine consists of only a reflector, then the
   machine produces a functioning, but trivial substitution cipher A ->
   Z, B -> Y, ..., Z -> A. Entering the "encrypted" message into the
   machine decrypts it flawlessly.
-  If none of the rotors turn, then the substitution functions TN and
   their inverses never change from step to step. In this case the
   machine just acts exactly like a simple reciprocal substitution
   cipher differing from the prior example only in the exact pairs of
   letters that are interchanged. However, the message does get
   encrypted, and decryption does work as expected.

Thinking about these trivial cases makes it easier to understand the
general concept behind Enigma-style machines: each individual character
is encrypted and decrypted through a reciprocal substitution cipher.
However, that cipher changes for each step (letter) of the encryption or
decryption process.

Code
====

The relevant classes are:

-  `Encryptor <encryptor.py>`__: The base class for all elements of the
   TypeX machine the perform encryption, including:
-  `Stator <stator.py>`__: A non-rotating wiring that provides a static
   mapping from each input character to a corresponding output character
-  `Rotor <rotor.py>`__: A wiring that provides a mapping from input
   characters to output characters (much like a Stator), but which also
   rotates (under certain conditions) to continually scramble the
   encryption scheme in use.
-  `Reflector <reflector.py>`__: A simple A<->Z, B<->Y, C<->X, etc.
   mapping that "reflects" characters at the end of encryption chain
   before sending them back through the inverse encryption process.
-  `TypeX <typex.py>`__: Instantiated with a collection of Encryptors,
   it can encrypt or decrypt messages. A message encrypted by a TypeX
   machine with a given configuration of Encryptor

Bugs
====

Currently the typex script contains a TypeX constructor with a hardcoded
list of Stators and Rotors, and needs to be edited to modify the
encryption settings. This should be considered a bug. See TODO.

The Enigma and TypeX algorithms support only 26-character alphabets.
This means that spaces, punctuation, and other nice features of readable
language are discarded. This isn't a bug in this code repo, but it sure
is annoying. See TODO.

TODO
====

-  Turn this into a pip module and publish it to
   `PyPI <https://pypi.python.org/pypi>`__
-  Support a plugboard configuration in addition to stators, for even
   more cryptographic strength
-  Modify typex to accept TypeX constructor arguments using
   `Argparse <https://docs.python.org/3/howto/argparse.html>`__ options
   or something similar
-  Support a pre-generated "codebook" of TypeX configurations for use by
   date and/or per communication partner
-  Add support for rotors larger than 26 characters, i.e. to support
   spaces, punctuation, carriage returns, etc.

References
==========

Chang, Kelly. `“Cryptanalysis of
TypeX.” <http://www.cryptomuseum.com/crypto/uk/typex/files/kelly.pdf>`__
SJSU Scholarly Works, 1 Apr. 2012

License
=======

The repository is made available to the public under the `MIT
License <LICENSE.md>`__.

Contributing
============

If you want to make improvements, please fork this repository and submit
a pull request! See the TODO list for places where your contribution
would be appreciated.

Feedback
========

Please submit feedback via `Github
issues <https://github.com/gregariouspanda/typex/issues>`__
