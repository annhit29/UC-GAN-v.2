### ref : https://github.com/CrypTools/CaesarCipher

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def encrypt(initial):
    """ Use : encrypt("message", 98)
    => 'gymmuay'
    """
    initial = initial.lower()
    output = ""

    key = 'qwertyuiopasdfghjklzxcvbnm' # key for encrypt

    shift = []

    for j in range(len(key)):
        x = ord(key[j]) - 97
        shift.append(x)

    cnt = 0
    for char in initial:
        if char in LETTERS:
            output += LETTERS[shift[LETTERS.index(char)]]
            cnt += 1
	
    return output

def decrypt(initial):
    """ Use : decrypt('gymmuay', 98)
    => 'message'
    """
    initial = initial.lower()
    output = ""

    key = 'kxvmcnophqrszyijadlegwbuft'  # inverse key for decrypt

    shift = []

    for j in range(len(key)):
        x = ord(key[j]) - 97
        shift.append(x)

    cnt = 0
    for char in initial:
        if char in LETTERS:
            output += LETTERS[shift[LETTERS.index(char)]]
            cnt += 1
	
    return output
