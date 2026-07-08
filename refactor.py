import os
import glob
import re

def sanitize_errors(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py'):
                continue
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # For generic e/exc where we want to replace with standard msg
            # We'll use a regex to replace "jsonify({'error': str(e)}), 500" with a standard message
            new_content = re.sub(r"jsonify\(\{['\"]error['\"]:\s*str\((?:e|exc)\)\}\),\s*500", "jsonify({'error': 'An internal error occurred.'}), 500", content)
            new_content = re.sub(r"jsonify\(\{\"error\":\s*str\((?:e|exc)\)\}\),\s*500", "jsonify({'error': 'An internal error occurred.'}), 500", new_content)
            
            # also for {"error": str(exc)}, 500 in core.py
            new_content = re.sub(r"\{\"error\":\s*str\((?:e|exc)\)\},\s*500", "{'error': 'An internal error occurred.'}, 500", new_content)

            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated {filepath}")

sanitize_errors('modules')
