import re

line = "Organic Bananas 1.99"
# Try to find price
price_match = re.search(r'\$?(\d+\.\d{2})', line)
if price_match:
    price = price_match.group(1)
    print("Price:", price)
