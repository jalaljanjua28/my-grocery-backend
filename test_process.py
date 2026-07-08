import re

def process_text_mock(text):
    lines = text.strip().split("\n")
    data_list = []
    for line in lines:
        if any(char.isalpha() for char in line):
            line = re.sub(r"[^a-zA-Z0-9\s\.$]", " ", line)
        else:
            continue
            
        line = line.strip()
        parts = line.split()
        if len(parts) < 2:
            continue
            
        # Try to find a price
        price = "$0.0"
        price_match = re.search(r'\$?(\d+\.\d{2})', line)
        if price_match:
            price = f"${price_match.group(1)}"
            # remove price from parts
            line = line.replace(price_match.group(0), "").strip()
            parts = line.split()
            
        quantity = "1"
        if parts and parts[-1].isdigit():
            quantity = parts.pop()
            
        name = " ".join(parts)
        if not name:
            continue
            
        data_list.append({"Name": name, "Price": price, "Quantity": quantity})
    return data_list

print(process_text_mock("Apple 1.99 2\nBanana 3.50\nMilk 1 gal 4.99 1"))
