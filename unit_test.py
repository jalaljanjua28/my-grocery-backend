import unittest
from app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_food_handling_advice_using_json(self):
        response = self.app.get('/api/food-handling-advice-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("handlingAdvice", response.json)

    def test_food_handling_advice_using_gpt(self):
        response = self.app.post('/api/food-handling-advice-using-gpt', json={"key": "proper food storage"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json)

    # def test_food_waste_reduction_using_json(self):
    #     response = self.app.get('/api/food-waste-reduction-using-json')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("foodWasteReductionSuggestions", response.json)

    def test_ethical_eating_suggestion_using_json(self):
        response = self.app.get('/api/ethical-eating-suggestion-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("ethicalEatingSuggestion", response.json)

    def test_ethical_eating_suggestion_using_gpt(self):
        response = self.app.post('/api/ethical-eating-suggestion-using-gpt')
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json)

    def test_get_fun_facts_using_json(self):
        response = self.app.get('/api/get-fun-facts-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("funFacts", response.json)

    def test_get_fun_facts_using_gpt(self):
        response = self.app.post('/api/get-fun-facts-using-gpt')
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json)

    def test_cooking_tips_using_json(self):
        response = self.app.get('/api/cooking-tips-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("cookingTips", response.json)

    def test_cooking_tips_using_gpt(self):
        response = self.app.post('/api/cooking-tips-using-gpt')
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json)

    def test_current_trends_using_json(self):
        response = self.app.get('/api/current-trends-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("currentTrend", response.json)

    def test_current_trends_using_gpt(self):
        response = self.app.post('/api/current-trends-using-gpt')
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json)

    def test_mood_changer_using_json(self):
        response = self.app.get('/api/mood-changer-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("moodChangerSuggestions", response.json)

    def test_mood_changer_using_gpt(self):
        response = self.app.post('/api/mood-changer-using-gpt', json={"user_mood": "feeling sad"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json)

    def test_jokes_using_json(self):
        response = self.app.get('/api/jokes-using-json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("jokes", response.json)

    def test_jokes_using_gpt(self):
        response = self.app.post('/api/jokes-using-gpt')
        self.assertEqual(response.status_code, 200)
        self.assertIn("jokes", response.json)

if __name__ == '__main__':
    unittest.main()
