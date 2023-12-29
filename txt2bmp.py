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


'''
change to 32 values +1/31 each time because of Lorenz cipher that can encode to /394+8
Each character corresponds to a value of `arr` in the solver_rotor: a-> 0, b-> 0.03225806451612903, etc., z->0.8064516129032258, /-> 0.8387096774193548 ..., 8-> 1.0
'''
arr_when_with_lorenz = [0.0, 0.03225806451612903, 0.06451612903225806, 0.0967741935483871, 0.12903225806451613, 0.16129032258064516, 
    0.1935483870967742, 0.22580645161290322, 0.25806451612903225, 0.29032258064516125, 0.3225806451612903, 0.3548387096774194, 
    0.3870967741935484, 0.4193548387096774, 0.45161290322580644, 0.4838709677419355, 0.5161290322580645, 0.5483870967741935, 
    0.5806451612903225, 0.6129032258064516, 0.6451612903225806, 0.6774193548387096, 0.7096774193548387, 0.7419354838709677, 
    0.7741935483870968, 0.8064516129032258, 0.8387096774193548, 0.8709677419354839, 0.9032258064516129, 0.9354838709677419, 
    0.967741935483871, 1.0]

'''
Each alphabet corresponds to a value of `arr` in the solver_rotor: 0-> a, 0.0392->b, etc., 1.0-> z
'''
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

    # pixel_len = len(formed_string)
    width = 100
    height = 1

    # Padding: Adds null bytes for alignment to match the desired dimensions.
    while len(formed_string) < (width * height):  # Padding
        formed_string += b'\x00'

    # Convert text data into an image
    image = Image.frombytes('L', (width, height), formed_string)
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    
    # print(grayscale_image.size) #(100, 1)

    # Extract grayscale pixel data
    grayscale_pixels = list(grayscale_image.getdata())

    # Reshape pixel data into a matrix
    matrix = [grayscale_pixels[i * width:(i + 1) * width] for i in range(height)]

    return matrix



def matrix2bmp(matrix, filename):
    height = len(matrix)
    width = len(matrix[0])
    im = Image.new('L', (width, height))
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
        #todo: `matrix2bmp`'s stuff already done in txt2bw, no need to convert from Image back to matrix THEN bacl to Image by calling matrix2bmp
        if args.bitmap is not None:
            if matrix2bmp(matrix, args.bitmap):
                print("Saved as bitmap in: {}".format(args.bitmap))
            else:
                print("Failed to save as bitmap...")

if __name__ == "__main__":
    main()



#inspired src: https://github.com/Hamz-a/txt2bmp/tree/master?tab=readme-ov-file