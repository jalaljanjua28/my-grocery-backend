import re

def process_text_demo(text):
    lines = text.strip().split("\n")
    data_list = []
    for line in lines:
        if not line: continue
        print(line)

process_text_demo("Apple 1\nBanana 2.50\nMilk 1 gal 3")
