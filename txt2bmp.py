#!/usr/bin/env python
import argparse
import math
from PIL import Image

'''
convert txt to bmp files with colors
'''
# def txt2rgb(text):
#     # Add NULL bytes and new lines
#     prepend = "\x00\r\n\r\n"
#     formed_string = prepend + text + "\x00"

#     pixel_len = math.ceil(len(formed_string) / 3)  # 3 rgb values per pixel
#     # width = math.ceil(math.sqrt(pixel_len))
#     width = 100#math.ceil(width / 4) * 4  # Make width a multiple of 4 to prevent a byte being added for the 4-byte alignment
#     height = 1#math.ceil(pixel_len / width)

#     while len(formed_string) != (width * height * 3):  # Padding
#         formed_string += "\x00"

#     offset = 0
#     matrix = [[0 for x in range(width)] for y in range(height)]
#     for y in range(height - 1, -1, -1):  # Loop from the bottom to top
#         for x in range(0, width):  # Loop from left to right
#             r = ord(formed_string[offset + 2])
#             g = ord(formed_string[offset + 1])
#             b = ord(formed_string[offset])
#             matrix[y][x] = (r, g, b)
#             offset += 3

#     return matrix

arr = [0.,     0.0392, 0.0784, 0.1176, 0.1569, 0.1961,
        0.2353, 0.2745, 0.3137, 0.3529, 0.3922, 0.4314,
        0.4706, 0.5098, 0.5490, 0.5882, 0.6275, 0.6667,
        0.7059, 0.7451, 0.7843, 0.8235, 0.8627, 0.9020,
        0.9412, 1.0000]
'''
convert txt to bmp files with only black, white, and grey colors
'''
def txt2bw(text):
    formed_string = text.encode('utf-8')

    pixel_len = len(formed_string)
    width = 100
    height = 1

    # Padding: Adds null bytes for alignment to match the desired dimensions.
    while len(formed_string) < (width * height):  # Padding
        formed_string += b'\x00'

    # This line initializes a 2D matrix (list of lists) of size width by height with all values set to 0.
    matrix = [[0 for _ in range(width)] for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            char_value = formed_string[x]
            index = int((char_value / 255) * (len(arr) - 1))  # Mapping the char value to the range of arr
            grey_value = int(arr[index] * 255)  # Scale the arr value to the range of grayscale (0-255)
            matrix[y][x] = (grey_value, grey_value, grey_value)
            # bit_value = 0
            # if x < pixel_len:
            #     # for i in range(8):
            #     #     bit_value |= (char_value & (1 << i)) >> i
                
            #     # # Make the black color more prominent
            #     # matrix[y][x] = (0, 0, 0) if bit_value else (255, 255, 255)

            #     # Count the number of set bits in the character byte
            #     bit_count = bin(char_value).count('1')
            #     # Scale the bit count to a range of grey values (0-255)
            #     grey_value = int((bit_count / 8) * 255)
            #     matrix[y][x] = (grey_value, grey_value, grey_value)
    
    return matrix

'''
creating a textual representation of the RGB (or in the context of the black and white conversion, the grayscale values) matrix
'''
def matrix2rep(matrix):
    height = len(matrix)
    width = len(matrix[0])
    representation = "Width={}, height={}\n".format(width, height)
    for y in range(height):
        for x in range(width):
            r, g, b = matrix[y][x]
            representation += "rgb({0: >3}, {1: >3}, {2: >3})  ".format(r, g, b)
        representation += "\n"
    return representation


def matrix2bmp(matrix, filename):
    height = len(matrix)
    width = len(matrix[0])
    im = Image.new("RGB", (width, height))
    pixels = im.load()

    for y in range(height):
        for x in range(width):
            pixels[x, y] = matrix[y][x]

    try:
        im.save(filename, "bmp")
    except:
        return False
    return True


def get_text(args):
    if args.input is not None:
        return args.input
    else:
        try:
            with open(args.file, "r") as f:
                return f.read()
        except:
            print("Can't read file.")
            return False


def main():
    argument_parser = argparse.ArgumentParser(description="Convert text to bmp.")
    group = argument_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", type=str, help="Input text from cli")
    group.add_argument("-f", "--file", type=str, help="Input text from file")
    argument_parser.add_argument("-b", "--bitmap", type=str, help="Save image as bitmap")
    argument_parser.add_argument("-t", "--text", type=str, help="Save rgb values to file")
    argument_parser.add_argument("-p", "--pixels", action="store_true", help="Print RGB values")
    args = argument_parser.parse_args()

    text = get_text(args)

    if text is not False:
        matrix = txt2bw(text)#txt2rgb(text)
        if args.bitmap is not None:
            if matrix2bmp(matrix, args.bitmap):
                print("Saved as bitmap in: {}".format(args.bitmap))
            else:
                print("Failed to save as bitmap...")
        elif args.text is not None:
            try:
                with open(args.text, "w") as f:
                    f.write(matrix2rep(matrix))
                print("Saved as text in: {}".format(args.text))
            except:
                print("Failed to save as text...")
        else:
            print(matrix2rep(matrix))


if __name__ == "__main__":
    main()