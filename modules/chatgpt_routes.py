from flask import Blueprint, jsonify

import modules.core as core
import modules.chatgpt_handlers as handlers  # noqa: F401 (used via handlers.*)
from modules.chatgpt_utils import filter_error_entries

bp = Blueprint("chatgpt", __name__, url_prefix="/api")


@bp.route("/food-handling-advice-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def food_handling_advice_using_gpt():
    return handlers.food_handling_advice_using_gpt()


@bp.route("/food-waste-reduction-using-gpt", methods=["POST"])
@core.authenticate_user_function
def food_waste_reduction_using_gpt():
    return handlers.food_waste_reduction_using_gpt()


@bp.route("/ethical-eating-suggestion-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def ethical_eating_suggestion_using_gpt():
    return handlers.ethical_eating_suggestion_using_gpt()


@bp.route("/get-fun-facts-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def get_fun_facts_using_gpt():
    return handlers.get_fun_facts_using_gpt()


@bp.route("/cooking-tips-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def cooking_tips_using_gpt():
    return handlers.cooking_tips_using_gpt()


@bp.route("/current-trends-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def current_trends_using_gpt():
    return handlers.current_trends_using_gpt()


@bp.route("/mood-changer-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def mood_changer_using_gpt():
    return handlers.mood_changer_using_gpt()


@bp.route("/nutritional-value-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def nutritional_value_using_gpt():
    return handlers.nutritional_value_using_gpt()


@bp.route("/allergy-information-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def allergy_information_using_gpt():
    return handlers.allergy_information_using_gpt()


@bp.route("/healthier-alternatives-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def healthier_alternatives_using_gpt():
    return handlers.healthier_alternatives_using_gpt()


@bp.route("/healthy-eating-advice-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def healthy_eating_advice_using_gpt():
    return handlers.healthy_eating_advice_using_gpt()


@bp.route("/health-advice-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def health_advice_using_gpt():
    return handlers.health_advice_using_gpt()


@bp.route("/healthy-items-usage-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def healthy_items_usage_using_gpt():
    return handlers.healthy_items_usage_using_gpt()


@bp.route("/nutritional-analysis-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def nutritional_analysis_using_gpt():
    return handlers.nutritional_analysis_using_gpt()


@bp.route("/health_incompatibilities_using_gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def health_incompatibilities_using_gpt():
    return handlers.health_incompatibilities_using_gpt()


@bp.route("/user-defined-dish-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def user_defined_dish_using_gpt():
    return handlers.user_defined_dish_using_gpt()


@bp.route("/fusion-cuisine-suggestion-using-gpt", methods=["GET", "POST"])
@core.authenticate_user_function
def fusion_cuisine_using_gpt():
    return handlers.fusion_cuisine_using_gpt()


@bp.route("/unique-recipes-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def unique_recipes_using_gpt():
    return handlers.unique_recipes_using_gpt()


@bp.route("/recipes-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def recipes_using_gpt():
    return handlers.recipes_using_gpt()


@bp.route("/diet-schedule-using-gpt", methods=["POST", "GET"])
@core.authenticate_user_function
def diet_schedule_using_gpt():
    return handlers.diet_schedule_using_gpt()


@bp.route("/nutritional-analysis-using-json", methods=["GET"])
@core.authenticate_user_function
def nutritional_analysis_using_json():
    data = core.get_data_from_json("ChatGPT/Health", "Nutritional_Analysis")
    if isinstance(data, dict) and "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"nutritionalAnalysis": filter_error_entries(data)})


@bp.route("/fusion-cuisine-suggestions-using-json", methods=["GET"])
@core.authenticate_user_function
def fusion_cuisine_suggestions_using_json():
    data = core.get_data_from_json("ChatGPT/Recipe", "Fusion_Cuisine_Suggestions")
    if isinstance(data, dict) and "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"fusionSuggestions": filter_error_entries(data)})


@bp.route("/food-handling-advice-using-json", methods=["GET"])
@core.authenticate_user_function
def food_handling_advice_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/HomePage", "food_handling_advice")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"handlingadvice": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/food-waste-reduction-using-json", methods=["GET"])
@core.authenticate_user_function
def food_waste_reduction_using_json():
    try:
        data = core.get_data_from_json(
            "ChatGPT/HomePage", "Food_Waste_Reduction_Suggestions"
        )
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"foodWasteReductionSuggestions": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/ethical-eating-suggestion-using-json", methods=["GET"])
@core.authenticate_user_function
def ethical_eating_suggestion_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/HomePage", "Ethical_Eating_Suggestions")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"ethicalEatingSuggestions": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/get-fun-facts-using-json", methods=["GET"])
@core.authenticate_user_function
def get_fun_facts_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/HomePage", "generated_fun_facts")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"funFacts": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/cooking-tips-using-json", methods=["GET"])
@core.authenticate_user_function
def cooking_tips_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/HomePage", "Cooking_Tips")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"cookingTips": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/current-trends-using-json", methods=["GET"])
@core.authenticate_user_function
def current_trends_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/HomePage", "Current_Trends")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"currentTrends": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/mood-changer-using-json", methods=["GET"])
@core.authenticate_user_function
def mood_changer_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/HomePage", "Mood_Changer")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"moodChangerSuggestions": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/nutritional-value-using-json", methods=["GET"])
@core.authenticate_user_function
def nutritional_value_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Health", "generated_nutritional_advice")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"nutritionalValue": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/allergy-information-using-json", methods=["GET"])
@core.authenticate_user_function
def allergy_information_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Health", "allergy_information")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"AllergyInformation": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/healthier-alternatives-using-json", methods=["GET"])
@core.authenticate_user_function
def healthier_alternatives_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Health", "Healthy_alternatives")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"alternatives": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/healthy-eating-advice-using-json", methods=["GET"])
@core.authenticate_user_function
def healthy_eating_advice_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Health", "healthy_eating_advice")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"eatingAdviceList": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/health-advice-using-json", methods=["GET"])
@core.authenticate_user_function
def health_advice_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Health", "Health_Advice")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"healthAdviceList": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/healthy-items-usage-using-json", methods=["GET"])
@core.authenticate_user_function
def healthy_items_usage_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Health", "healthy_usage")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"suggestions": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/health_incompatibilities_using_json", methods=["GET"])
@core.authenticate_user_function
def health_incompatibilities_using_json():
    try:
        data = core.get_data_from_json(
            "ChatGPT/Health", "health_incompatibility_information_all"
        )
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"healthIncompatibilities": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/user-defined-dish-using-json", methods=["GET"])
@core.authenticate_user_function
def user_defined_dish_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Recipe", "User_Defined_Dish")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"definedDishes": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/unique-recipes-using-json", methods=["GET"])
@core.authenticate_user_function
def unique_recipes_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Recipe", "Unique_Recipes")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"uniqueRecipes": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/recipes-using-json", methods=["GET"])
@core.authenticate_user_function
def recipes_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Recipe", "generated_recipes")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"generatedRecipes": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/diet-schedule-using-json", methods=["GET"])
@core.authenticate_user_function
def diet_schedule_using_json():
    try:
        data = core.get_data_from_json("ChatGPT/Recipe", "diet_schedule")
        if isinstance(data, dict) and "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"dietSchedule": filter_error_entries(data)}), 200
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500


@bp.route("/jokes-with-timestamp/<timestamp>", methods=["GET"])
@core.authenticate_user_function
def jokes_using_timestamp(timestamp):
    return handlers.get_jokes_with_timestamp(timestamp)
