from enigma_typex import *

#----Set up the TypeX machine----
t_pb = ""  # Set your plugboard settings here if needed
t_rotors = "TYPEX_A TYPEX_B TYPEX_C TYPEX_D TYPEX_E"  # 5 TypeX rotors
t_reflector = "B"  # Choose the reflector for TypeX
t_ring_settings = "01 01 01 01 01"  # Set ring settings for each rotor
t_initial_positions = "A A A A A"  # Set initial positions for each rotor


#----Encryption----

# Your plaintext message: Valid char must be uppercase from "A" to "Z"
plaintext = "WasMachstDu"#"YOURPLAINTEXTMESSAGEHERE"

#1. change to uppercase letters (<- the curr only valid alphabets):
plaintext = plaintext.upper() #todo: this typex only receives uppercase letters, I'll change it to receive A~Z and a~z. I mean: all the letters receivable by Brown Corpus.
#2. Create a Plugboard instance
plugboard = Plugboard([PlugLead(mapping) for mapping in t_pb.split()])

#3. Create a MultiRotor instance for TypeX
multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)

#4. Create a TypeX instance with the MultiRotor and Plugboard
typex = MultiEnigma(multirotor, plugboard)

#5. Encrypt the plaintext message
encrypted_text = typex.encode_decode(plaintext)
#Encryption done

#6. change to lowercase letters
encrypted_text = encrypted_text.lower()
print("Encrypted Text:", encrypted_text) #HHSYRABMQULAAGWYXWCXFJND

#todo: How to accept a~z, 0~9 too?
#todo: in given plaintext dataset from UC-GAN src code, do they allow blank spaces?


#----Decryption----
# Your encrypted ciphertext
ciphertext = "ududmrmauks"

#1. change to uppercase letters (<- the curr only valid alphabets):
ciphertext = ciphertext.upper() #todo: this typex only receives uppercase letters, I'll change it to receive A~Z and a~z. I mean: all the letters receivable by Brown Corpus.

#2. Create a Plugboard instance
plugboard = Plugboard([PlugLead(mapping) for mapping in t_pb.split()])

#3. Create a MultiRotor instance for TypeX, by Resetting the machine:
multirotor = MultiRotor(t_rotors, t_reflector, t_ring_settings, t_initial_positions)

#4. Create a TypeX instance with the MultiRotor and Plugboard
typex = MultiEnigma(multirotor, plugboard)

#5. Decrypt the ciphertext message
decrypted_text = typex.encode_decode(ciphertext)
#Decryption done

#6.
decrypted_text = decrypted_text.lower() #change to lowercase letters

print("Decrypted Text:", decrypted_text)


'''
run the prgm, cmds:
cd C:\année_scolaire_23-24\BA5\BRP\UC-GAN-v.2\lib
cd .\typex_machine\
`python .\typex_enc_dec_eg.py`
'''
