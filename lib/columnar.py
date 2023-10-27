import math

def row_encrypt(s,key):
    
    # to remove repeated alphabets in key
    temp=[]
    for i in key:
        if i not in temp:
            temp.append(i)
    k=""
    for i in temp:
        k+=i
    print("The key used for encryption is: ",k)
    
    # ceil is used to adjust the count of
    # rows according to length of message
    b=math.ceil(len(s)/len(k))
    
    # if b is less than length of key, then it will not form square matrix when
    # length of meessage not equal to rowsize*columnsize of square matrix
    if(b<len(k)):
        b=b+(len(k)-b)
    # if b is greater than length of key, then it will not from a
    # square matrix, but if less then length of key, we have to add padding
    
    arr=[['_' for i in range(len(k))]
         for j in range(b)]
    i=0
    j=0
    # arranging the message into matrix
    for h in range(len(s)):
        arr[i][j]=s[h]
        j+=1
        if(j>len(k)-1):
            j=0
            i+=1
    print("The message matrix is: ")
    for i in arr:
        print(i)

    cipher_text=""
    # To get indices as the key numbers instead of alphabets in the key, according
    # to algorithm, for appending the elementsof matrix formed earlier, column wise.
    kk=sorted(k)
    
    for i in kk:
        # gives the column index
        h=k.index(i)
        for j in range(len(arr)):
            cipher_text+=arr[j][h]
    print("The cipher text is: ",cipher_text)
    return cipher_text
        
msg=input("Enter the message: ")
key=input("Enter the key in alphabets: ")
row_encrypt(msg,key)

'''
----------OUTPUT----------
Enter the message: My computer is owned by me
Enter the key in alphabets: expensive
The key used for encryption is:  expnsiv
The message matrix is: 
['M', 'y', ' ', 'c', 'o', 'm', 'p']
['u', 't', 'e', 'r', ' ', 'i', 's']
[' ', 'o', 'w', 'n', 'e', 'd', ' ']
['b', 'y', ' ', 'm', 'e', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
The cipher text is:  Mu b___mid____crnm___ ew ___o ee___ps ____ytoy___
>>>
'''


def row_decrypt(s,key):

    # to remove repeated alphabets in key
    temp=[]
    for i in key:
        if i not in temp:
            temp.append(i)
    k=""
    for i in temp:
        k+=i
    print("The key used for encryption is: ",k)
    
    arr=[['' for i in range(len(k))]
         for j in range(int(len(s)/len(k)))]

    # To get indices as the key numbers instead of alphabets in the key, according
    # to algorithm, for appending the elementsof matrix formed earlier, column wise.
    kk=sorted(k)
    
    d=0
    # arranging the cipher message into matrix
    # to get the same matrix as in encryption
    for i in kk:
        h=k.index(i)
        for j in range(len(k)):
            arr[j][h]=s[d]
            d+=1
                
    print("The message matrix is: ")
    for i in arr:
        print(i)

    # the plain text
    plain_text=""
    for i in arr:
        for j in i:
            plain_text+=j
    print("The plain text is: ",plain_text)
    return plain_text

        
msg=input("Enter the message to be decrypted: ")
key=input("Enter the key in alphabets: ")
row_decrypt(msg,key)

'''
----------OUTPUT----------
Enter the message to be decrypted: Mu b___mid____crnm___ ew ___o ee___ps ____ytoy___
Enter the key in alphabets: expensive
The key used for encryption is:  expnsiv
The message matrix is: 
['M', 'y', ' ', 'c', 'o', 'm', 'p']
['u', 't', 'e', 'r', ' ', 'i', 's']
[' ', 'o', 'w', 'n', 'e', 'd', ' ']
['b', 'y', ' ', 'm', 'e', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
The plain text is:  My computer is owned by me_______________________
>>>
'''


# https://github.com/Abhiramborige/Crypto-systems/blob/master/columnar_transposition_decryption.py