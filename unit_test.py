import unittest
from app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # def test_food_handling_advice_using_json(self):
    #     response = self.app.get('/api/food-handling-advice-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("handlingAdvice", response.json)

    # def test_food_handling_advice_using_gpt(self):
    #     response = self.app.post('/api/food-handling-advice-using-gpt', json={"key": "store raw meat separately"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("handlingAdvice", response.json)

    # def test_food_waste_reduction_using_json(self):
    #     response = self.app.get('/api/food-waste-reduction-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("foodWasteReductionSuggestions", response.json)

    # def test_food_waste_reduction_using_gpt(self):
    #     response = self.app.post('/api/food-waste-reduction-using-gpt', json={"user_input": "how to use vegetable scraps"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("foodWasteReductionSuggestions", response.json)

    # def test_ethical_eating_using_json(self):
    #     response = self.app.get('/api/ethical-eating-suggestion-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("ethicalEatingSuggestion", response.json)

    # def test_ethical_eating_suggestion_using_gpt(self):
    #     response = self.app.post('/api/ethical-eating-suggestion-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("ethicalEatingSuggestion", response.json)

    # def test_get_fun_facts_using_json(self):
    #     response = self.app.get('/api/get-fun-facts-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("funFacts", response.json)

    # def test_get_fun_facts_using_gpt(self):
    #     response = self.app.post('/api/get-fun-facts-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("funFacts", response.json)

    # def test_cooking_tips_using_json(self):
    #     response = self.app.get('/api/cooking-tips-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("cookingTips", response.json)

    # def test_cooking_tips_using_gpt(self):
    #     response = self.app.post('/api/cooking-tips-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("cookingTips", response.json)

    # def test_current_trends_using_json(self):
    #     response = self.app.get('/api/current-trends-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("currentTrend", response.json)

    # def test_current_trends_using_gpt(self):
    #     response = self.app.post('/api/current-trends-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("currentTrend", response.json)

    # def test_mood_changer_using_json(self):
    #     response = self.app.get('/api/mood-changer-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("moodChangerSuggestions", response.json)

    # def test_mood_changer_using_gpt(self):
    #     response = self.app.post('/api/mood-changer-using-gpt', json={"user_mood": "happy"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("moodChangerSuggestions", response.json)

    # def test_jokes_json(self):
    #     response = self.app.get('/api/jokes-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("jokes", response.json)

    # def test_jokes_using_gpt(self):
    #     response = self.app.post('/api/jokes-using-gpt', json={"theme": "knock-knock"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("jokes", response.json)
        
######################################################################################################

    # def test_nutritional_value_using_json(self):
    #     response = self.app.get('/api/nutritional-value-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("nutritionalValue", response.json)

    # def test_nutritional_value_using_gpt(self):
    #     response = self.app.post('/api/nutritional-value-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("nutritionalValue", response.json)

    # def test_allergy_information_using_json(self):
    #     response = self.app.get('/api/allergy-information-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("AllergyInformation", response.json)

    # def test_allergy_information_using_gpt(self):
    #     response = self.app.post('/api/allergy-information-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("AllergyInformation", response.json)

    # def test_healthier_alternatives_using_json(self):
    #     response = self.app.get('/api/healthier-alternatives-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("alternatives", response.json)

    # def test_healthier_alternatives_using_gpt(self):
    #     response = self.app.post('/api/healthier-alternatives-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("alternatives", response.json)

    # def test_healthy_eating_advice_using_json(self):
    #     response = self.app.get('/api/healthy-eating-advice-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("eatingAdviceList", response.json)

    # def test_healthy_eating_advice_using_gpt(self):
    #     response = self.app.post('/api/healthy-eating-advice-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("eatingAdviceList", response.json)

    # def test_health_advice_using_json(self):
    #     response = self.app.get('/api/health-advice-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("healthAdviceList", response.json)

    # def test_health_advice_using_gpt(self):
    #     response = self.app.post('/api/health-advice-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("healthAdviceList", response.json)

    # def test_healthy_items_usage_using_json(self):
    #     response = self.app.get('/api/healthy-items-usage-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("suggestions", response.json)

    # def test_healthy_items_usage_using_gpt(self):
    #     response = self.app.post('/api/healthy-items-usage-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("suggestions", response.json)

    # def test_nutritional_analysis_using_json(self):
    #     response = self.app.get('/api/nutritional-analysis-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("nutritionalAnalysis", response.json)

    # def test_nutritional_analysis_using_gpt(self):
    #     response = self.app.post('/api/nutritional-analysis-using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("nutritionalAnalysis", response.json)

    # def test_health_incompatibilities_using_json(self):
    #     response = self.app.get('/api/health_incompatibilities_using_json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("healthIncompatibilities", response.json)

    # def test_health_incompatibilities_using_gpt(self):
    #     response = self.app.post('/api/health_incompatibilities_using-gpt', json={})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("healthIncompatibilities", response.json)
        
######################################################################################################

    # def test_user_defined_dish_using_json(self):
    #     response = self.app.get('/api/user-defined-dish-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("definedDishes", response.json)

    # def test_user_defined_dish_using_gpt(self):
    #     response = self.app.post('/api/user-defined-dish-using-gpt', json={"user_dish": "Pasta"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("definedDishes", response.json)

    # def test_fusion_cuisine_suggestions_using_json(self):
    #     response = self.app.get('/api/fusion-cuisine-suggestions-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("fusionSuggestions", response.json)

    # def test_fusion_cuisine_using_gpt(self):
    #     response = self.app.post('/api/fusion-cuisine-suggestion-using-gpt', json={"user_input": "Mexican and Chinese"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("fusionSuggestions", response.json)

    # def test_unique_recipes_using_json(self):
    #     response = self.app.get('/api/unique-recipes-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("uniqueRecipes", response.json)

    # def test_unique_recipes_using_gpt(self):
    #     response = self.app.post('/api/unique-recipes-using-gpt', json={"unique_recipe": "strawberry and spinach"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("uniqueRecipes", response.json)

    def test_diet_schedule_using_json(self):
        response = self.app.get('/api/diet-schedule-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("dietSchedule", response.json)

    def test_diet_schedule_using_gpt(self):
        response = self.app.post('/api/diet-schedule-using-gpt', json={})
        self.assertEqual(response.status_code, 200)
        self.assertIn("dietSchedule", response.json)
        
    def test_recipes_using_json(self):
        response = self.app.get('/api/recipes-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("generatedRecipes", response.json)

    def test_recipes_using_gpt(self):
        response = self.app.get('/api/diet-schedule-using-gpt', json={})
        self.assertEqual(response.status_code, 200)
        self.assertIn("generatedRecipes", response.json)
if __name__ == '__main__':
    unittest.main()
