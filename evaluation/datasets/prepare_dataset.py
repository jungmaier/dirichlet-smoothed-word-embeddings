#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:20:44 2019

@author: jakob
"""

import sys

if len(sys.argv) != 7:
    sys.exit("usage: " + sys.argv[0] + " word_1-position" + " word_2-position" + " value-position" + " separator" + " source-dataset" + " prepared-dataset")

word_1_pos = int(sys.argv[1])
word_2_pos = int(sys.argv[2])
value_pos = int(sys.argv[3])
separator = str(sys.argv[4])
source_dataset = sys.argv[5]
prepared_dataset = sys.argv[6]

if separator == "tab":
    separator = "\t"
elif separator == "space":
    separator == "\s"
elif separator == "comma":
    separator = ","
elif separator == "semicolon":
    separator = ";"

with open(source_dataset) as file_in:
    with open(prepared_dataset, "w") as file_out:
        for line in file_in:
            line_list = line.strip().split(separator)
            print(line_list)
            file_out.write(line_list[word_1_pos] + "\t" + line_list[word_2_pos] + "\t" + line_list[value_pos] + "\n")
        
        
        
        
    
    