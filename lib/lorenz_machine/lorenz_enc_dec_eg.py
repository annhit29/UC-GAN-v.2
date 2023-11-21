from lorenz.machines import SZ40
from lorenz.patterns import KH_CAMS, ZMUG_CAMS, BREAM_CAMS # Some sample patterns (KH_CAMS, ZMUG_CAMS, and BREAM_CAMS) are provided in the lorenz.patterns module.

# import telegrahy utility library
from lorenz.telegraphy import Teleprinter

#---Encryption---
# encode the message as five-bit ITA2
input_PT = "ATTACK99AT99DAWn" #input_PT is a strPT

#1. transform the input PT from str to list format
listPT = Teleprinter.encode(input_PT)

#2. 
# use the `KH` pattern to encrypt the message.
machine = SZ40(KH_CAMS)

#3. encrypt the PT list format to CT list format
listCT = machine.feed(listPT) # list CT

#4. transform the CT list format to str format
strCT = Teleprinter.decode(listCT) #str CT
#Encryption done

# strCT = strCT.lower()


print(strCT) # 9w3umkegpjzqokxc


#---Decryption---
input_CT = strCT #input_CT is a strCT

#1. transform the input CT from str to list format
dec_listCT = Teleprinter.encode(input_CT)

#2. reset the machine:
# use the `KH` pattern to encrypt the message.
machine = SZ40(KH_CAMS)

#3. encrypt the CT list format to PT list format
dec_listPT = machine.feed(dec_listCT)

#4. transform the PT list format to str format
dec_strPT = Teleprinter.decode(dec_listPT)
#Decryption done

# dec_strPT = dec_strPT.lower()

print(dec_strPT) # attack99at99dawn


# src: https://github.com/hughcoleman/lorenz