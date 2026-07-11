from pathlib import Path
import re

app_path = Path("app.py")
text = app_path.read_text(encoding="utf-8")

route_names = [
    "food_handling_advice_using_gpt",
    "food_waste_reduction_using_gpt",
    "ethical_eating_suggestion_using_gpt",
    "get_fun_facts_using_gpt",
    "cooking_tips_using_gpt",
    "current_trends_using_gpt",
    "mood_changer_using_gpt",
    "nutritional_value_using_gpt",
    "allergy_information_using_gpt",
    "healthier_alternatives_using_gpt",
    "healthy_eating_advice_using_gpt",
    "health_advice_using_gpt",
    "healthy_items_usage_using_gpt",
    "nutritional_analysis_using_gpt",
    "health_incompatibilities_using_gpt",
    "user_defined_dish_using_gpt",
    "fusion_cuisine_using_gpt",
    "unique_recipes_using_gpt",
    "recipes_using_gpt",
    "diet_schedule_using_gpt",
]

module_dir = Path("modules")
module_dir.mkdir(exist_ok=True)
module_path = module_dir / "chatgpt_handlers.py"

module_lines = [
    "import json",
    "import logging",
    "import os",
    "import random",
    "import re",
    "import tempfile",
    "import time",
    "from datetime import date, datetime, timedelta",
    "",
    "import requests",
    "from flask import jsonify, request",
    "",
    "from app import (",
    "    bucket,",
    "    db,",
    "    get_data_from_json,",
    "    openai_client,",
    "    save_data_to_cloud_storage,",
    ")",
    "",
]

new_text = text

for name in route_names:
    function_name = f"{name}_function"
    for target_name in (name, function_name):
        pattern = re.compile(
            rf"(?ms)^def\s+{re.escape(target_name)}\(\):\n(?:.*?)(?=^def\s+\w+\(\):|\Z)"
        )
        match = pattern.search(new_text)
        if not match:
            continue
        block = match.group(0)
        replacement = (
            f"def {target_name}():\n"
            f"    from modules.chatgpt_handlers import {target_name}\n"
            f"    return {target_name}()\n"
        )
        new_text = new_text[: match.start()] + replacement + new_text[match.end() :]
        module_lines.append(block.rstrip() + "\n")

module_path.write_text("\n".join(module_lines).rstrip() + "\n", encoding="utf-8")
app_path.write_text(new_text, encoding="utf-8")
print(f"Wrote {module_path}")
