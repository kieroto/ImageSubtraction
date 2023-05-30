import inference
import os
from os import listdir
from os.path import isfile, join

file = open('parameters.txt', 'r')
Lines = file.readlines()
bgfilepath = ""
fill_color = (0,0,0)
mode = "fillcolor"

count = 0
for line in Lines:
    if count == 0:
        bgfilepath = line.split(": ")[1].strip("\n")
    elif count == 1:
        mode = line.split(": ")[1].strip("\n")
    elif count == 2:
        line3 = line.split(": ")
        line3 = line3[1].split(",")
        fill_color = (int(line3[0]),int(line3[1]),int(line3[2].strip("\n")))
    count+=1

dir = (os.getcwd()).encode('unicode_escape').decode()
input_dir = dir + '/input'
output_dir = dir + '/output'

for f in listdir(input_dir):
    filename = f.split(".")
    supported = ["jpeg", "jpg", "webp", "bmp", "png"]
    if filename[1] in supported:
        print(filename[0]) 
        # save file to /static/uploads
        filepath = join(input_dir, f)
        outputpath = join(output_dir, filename[0]+'.png')
        inference.predict(filepath, outputpath, fill_color, mode, bgfilepath)
