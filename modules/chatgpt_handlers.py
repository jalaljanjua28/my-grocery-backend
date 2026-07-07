import logging
import random
import time

from flask import jsonify, request

from modules.chatgpt_utils import _get_inventory_items, _call_openai, _save_prompt_output


def food_handling_advice_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        advice_list = []
        for item in food_items:
            time.sleep(0.2)
            prompt = f"Provide advice on how to handle {item['Name']} to increase its shelf life:"
            advice = _call_openai(prompt, max_tokens=800)
            advice_list.append({"Food Item": item["Name"], "Handling Advice": advice})
        _save_prompt_output("ChatGPT/HomePage", "food_handling_advice", advice_list)
        return jsonify({"handlingadvice": advice_list})
    except Exception as exc:
        logging.exception("food_handling_advice_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def food_waste_reduction_using_gpt_function():
    try:
        user_input = (request.get_json(silent=True) or {}).get("user_input", "Suggest a recipe that helps reduce food waste")
        suggestions = []
        prompt = user_input
        generated = _call_openai(prompt, max_tokens=800)
        suggestions.append({"Prompt": prompt, "Food Waste Reduction Suggestion": generated})
        _save_prompt_output("ChatGPT/HomePage", "Food_Waste_Reduction_Suggestions", suggestions)
        return jsonify({"foodWasteReductionSuggestions": suggestions})
    except Exception as exc:
        logging.exception("food_waste_reduction_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def ethical_eating_suggestion_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        group_of_items = [item["Name"] for item in food_items[:5]]
        prompt = "Consider the ethical aspects of the following ingredients:\n\n"
        for item in group_of_items:
            prompt += f"- {item}\n"
        response_text = _call_openai(prompt, max_tokens=400)
        payload = [{"Group of Items": group_of_items, "Ethical Eating Suggestions": response_text}]
        _save_prompt_output("ChatGPT/HomePage", "Ethical_Eating_Suggestions", payload)
        return jsonify({"ethicalEatingSuggestions": payload})
    except Exception as exc:
        logging.exception("ethical_eating_suggestion_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def get_fun_facts_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        facts = []
        for item in food_items:
            time.sleep(0.2)
            prompt = f"Provide interesting and fun facts about {item['Name']}:"
            fact = _call_openai(prompt, max_tokens=800)
            facts.append({"Food Item": item["Name"], "Fun Fact": fact})
        _save_prompt_output("ChatGPT/HomePage", "generated_fun_facts", facts)
        return jsonify({"funFacts": facts})
    except Exception as exc:
        logging.exception("get_fun_facts_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def cooking_tips_using_gpt_function():
    try:
        tips = []
        for _ in range(3):
            prompt = f"Seek advice on {random.choice(['cooking techniques', 'tips for improving a dish', 'alternative ingredients for dietary restrictions'])}."
            tip = _call_openai(prompt, max_tokens=500)
            tips.append({"Prompt": prompt, "Cooking Tip": tip})
        _save_prompt_output("ChatGPT/HomePage", "Cooking_Tips", tips)
        return jsonify({"cookingTips": tips})
    except Exception as exc:
        logging.exception("cooking_tips_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def current_trends_using_gpt_function():
    try:
        trends = []
        prompt = f"Stay updated on {random.choice(['exciting', 'cutting-edge', 'latest'])} food trends, {random.choice(['innovations', 'revolutions', 'breakthroughs'])}, or {random.choice(['unique', 'extraordinary', 'exceptional'])} culinary experiences."
        trend = _call_openai(prompt, max_tokens=400)
        trends.append({"Prompt": prompt, "Fun Facts": trend})
        _save_prompt_output("ChatGPT/HomePage", "Current_Trends", trends)
        return jsonify({"currentTrends": trends})
    except Exception as exc:
        logging.exception("current_trends_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def mood_changer_using_gpt_function():
    try:
        user_mood = (request.get_json(silent=True) or {}).get("user_mood", "Sad, I'm feeling tired, I'm going to bed")
        suggestions = []
        prompt = f"Suggest a food that can improve my mood when I'm feeling {user_mood}."
        suggestion = _call_openai(prompt, max_tokens=300)
        suggestions.append({"User Mood": user_mood, "Prompt": prompt, "Food Suggestion": suggestion})
        _save_prompt_output("ChatGPT/HomePage", "Mood_Changer", suggestions)
        return jsonify({"moodChangerSuggestions": suggestions})
    except Exception as exc:
        logging.exception("mood_changer_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def nutritional_value_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        advice = []
        for item in food_items[:5]:
            time.sleep(0.2)
            prompt = f"Provide nutritional advice for incorporating {item['Name']} into a balanced diet:"
            generated = _call_openai(prompt, max_tokens=800)
            advice.append({"Food Item": item["Name"], "Nutritional Advice": generated})
        _save_prompt_output("ChatGPT/Health", "generated_nutritional_advice", advice)
        return jsonify({"nutritionalValue": advice})
    except Exception as exc:
        logging.exception("nutritional_value_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def allergy_information_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        allergy_info = []
        for item in food_items[:4]:
            time.sleep(0.2)
            prompt = f"Allergy side effects of {item['Name']}:"
            generated = _call_openai(prompt, max_tokens=600)
            allergy_info.append({"Food Item": item["Name"], "Allergy Information": generated})
        _save_prompt_output("ChatGPT/Health", "allergy_information", allergy_info)
        return jsonify({"AllergyInformation": allergy_info})
    except Exception as exc:
        logging.exception("allergy_information_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def healthier_alternatives_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        alternatives = []
        for item in food_items[:4]:
            time.sleep(0.2)
            prompt = f"Suggest a healthier alternative to {item['Name']}:"
            generated = _call_openai(prompt, max_tokens=600)
            alternatives.append({"Food Item": item["Name"], "Healthy Alternative": generated})
        _save_prompt_output("ChatGPT/Health", "Healthy_alternatives", alternatives)
        return jsonify({"alternatives": alternatives})
    except Exception as exc:
        logging.exception("healthier_alternatives_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def healthy_eating_advice_using_gpt_function():
    try:
        prompt = "Provide general advice for maintaining healthy eating habits:"
        response_text = _call_openai(prompt, max_tokens=500)
        payload = [{"Prompt": prompt, "Health Advice": response_text}]
        _save_prompt_output("ChatGPT/Health", "healthy_eating_advice", payload)
        return jsonify({"eatingAdviceList": payload})
    except Exception as exc:
        logging.exception("healthy_eating_advice_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def health_advice_using_gpt_function():
    try:
        prompt = "Get general information or tips on healthy eating."
        response_text = _call_openai(prompt, max_tokens=500)
        payload = [{"Prompt": prompt, "Health Advice": response_text}]
        _save_prompt_output("ChatGPT/Health", "Health_Advice", payload)
        return jsonify({"healthAdviceList": payload})
    except Exception as exc:
        logging.exception("health_advice_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def healthy_items_usage_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        suggestions = []
        for item in food_items[:4]:
            time.sleep(0.2)
            prompt = f"Suggest ways to incorporate {item['Name']} into a healthy diet:"
            generated = _call_openai(prompt, max_tokens=700)
            suggestions.append({"Food Item": item["Name"], "Suggestion": generated})
        _save_prompt_output("ChatGPT/Health", "healthy_usage", suggestions)
        return jsonify({"suggestions": suggestions})
    except Exception as exc:
        logging.exception("healthy_items_usage_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def nutritional_analysis_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        analysis = []
        for item in food_items[:4]:
            time.sleep(0.2)
            prompt = f"Give a concise nutritional analysis for {item['Name']}:"
            generated = _call_openai(prompt, max_tokens=700)
            analysis.append({"Food Item": item["Name"], "Nutritional Analysis": generated})
        _save_prompt_output("ChatGPT/Health", "Nutritional_Analysis", analysis)
        return jsonify({"nutritionalAnalysis": analysis})
    except Exception as exc:
        logging.exception("nutritional_analysis_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def health_incompatibilities_using_gpt_function():
    try:
        food_items = _get_inventory_items()
        incompatibilities = []
        for item in food_items[:4]:
            time.sleep(0.2)
            prompt = f"List health incompatibilities or cautions related to {item['Name']}:"
            generated = _call_openai(prompt, max_tokens=700)
            incompatibilities.append({"Food Item": item["Name"], "Health Incompatibility": generated})
        _save_prompt_output("ChatGPT/Health", "health_incompatibility_information_all", incompatibilities)
        return jsonify({"healthIncompatibilities": incompatibilities})
    except Exception as exc:
        logging.exception("health_incompatibilities_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def user_defined_dish_using_gpt_function():
    try:
        prompt = (request.get_json(silent=True) or {}).get("prompt", "Describe a creative dish using common pantry ingredients")
        response_text = _call_openai(prompt, max_tokens=700)
        payload = [{"Prompt": prompt, "Fun Facts": response_text}]
        _save_prompt_output("ChatGPT/Recipe", "User_Defined_Dish", payload)
        return jsonify({"definedDishes": payload})
    except Exception as exc:
        logging.exception("user_defined_dish_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def fusion_cuisine_using_gpt_function():
    try:
        prompt = (request.get_json(silent=True) or {}).get("prompt", "Suggest a fusion cuisine dish using current pantry ingredients")
        response_text = _call_openai(prompt, max_tokens=700)
        payload = [{"Prompt": prompt, "Fusion Cuisine Suggestion": response_text}]
        _save_prompt_output("ChatGPT/Recipe", "Fusion_Cuisine_Suggestions", payload)
        return jsonify({"fusionSuggestions": payload})
    except Exception as exc:
        logging.exception("fusion_cuisine_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def unique_recipes_using_gpt_function():
    try:
        prompt = (request.get_json(silent=True) or {}).get("prompt", "Create a unique recipe using pantry ingredients")
        response_text = _call_openai(prompt, max_tokens=900)
        payload = [{"Prompt": prompt, "Recipe": response_text, "Encouragement": "Great job exploring a new recipe!"}]
        _save_prompt_output("ChatGPT/Recipe", "Unique_Recipes", payload)
        return jsonify({"uniqueRecipes": payload})
    except Exception as exc:
        logging.exception("unique_recipes_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def recipes_using_gpt_function():
    try:
        prompt = (request.get_json(silent=True) or {}).get("prompt", "Suggest a quick recipe based on ingredients in the kitchen")
        response_text = _call_openai(prompt, max_tokens=900)
        payload = [{"Prompt": prompt, "Recipe": response_text}]
        _save_prompt_output("ChatGPT/Recipe", "generated_recipes", payload)
        return jsonify({"generatedRecipes": payload})
    except Exception as exc:
        logging.exception("recipes_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


def diet_schedule_using_gpt_function():
    try:
        prompt = (request.get_json(silent=True) or {}).get("prompt", "Create a simple diet schedule for a healthy week")
        response_text = _call_openai(prompt, max_tokens=800)
        payload = [{"Prompt": prompt, "Diet Schedule": response_text}]
        _save_prompt_output("ChatGPT/Recipe", "diet_schedule", payload)
        return jsonify({"dietSchedule": payload})
    except Exception as exc:
        logging.exception("diet_schedule_using_gpt_function failed")
        return jsonify({"error": str(exc)}), 500


# Thin wrappers used by the Flask routes in app.py

def food_handling_advice_using_gpt():
    return food_handling_advice_using_gpt_function()


def food_waste_reduction_using_gpt():
    return food_waste_reduction_using_gpt_function()


def ethical_eating_suggestion_using_gpt():
    return ethical_eating_suggestion_using_gpt_function()


def get_fun_facts_using_gpt():
    return get_fun_facts_using_gpt_function()


def cooking_tips_using_gpt():
    return cooking_tips_using_gpt_function()


def current_trends_using_gpt():
    return current_trends_using_gpt_function()


def mood_changer_using_gpt():
    return mood_changer_using_gpt_function()


def nutritional_value_using_gpt():
    return nutritional_value_using_gpt_function()


def allergy_information_using_gpt():
    return allergy_information_using_gpt_function()


def healthier_alternatives_using_gpt():
    return healthier_alternatives_using_gpt_function()


def healthy_eating_advice_using_gpt():
    return healthy_eating_advice_using_gpt_function()


def health_advice_using_gpt():
    return health_advice_using_gpt_function()


def healthy_items_usage_using_gpt():
    return healthy_items_usage_using_gpt_function()


def nutritional_analysis_using_gpt():
    return nutritional_analysis_using_gpt_function()


def health_incompatibilities_using_gpt():
    return health_incompatibilities_using_gpt_function()


def user_defined_dish_using_gpt():
    return user_defined_dish_using_gpt_function()


def fusion_cuisine_using_gpt():
    return fusion_cuisine_using_gpt_function()


def unique_recipes_using_gpt():
    return unique_recipes_using_gpt_function()


def recipes_using_gpt():
    return recipes_using_gpt_function()


def diet_schedule_using_gpt():
    return diet_schedule_using_gpt_function()
