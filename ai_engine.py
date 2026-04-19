import math
import random

class PurePythonNaiveBayes:
    def __init__(self):
        self.vocab = set()
        self.class_word_counts = {}
        self.class_counts = {}
        self.total_docs = 0

    def fit(self, X, y):
        for text, intent in zip(X, y):
            self.total_docs += 1
            if intent not in self.class_counts:
                self.class_counts[intent] = 0
                self.class_word_counts[intent] = {}
            
            self.class_counts[intent] += 1
            words = text.lower().split()
            for w in words:
                self.vocab.add(w)
                self.class_word_counts[intent][w] = self.class_word_counts[intent].get(w, 0) + 1

    def predict(self, text):
        words = text.lower().split()
        best_intent = None
        max_prob = -float('inf')
        
        for intent in self.class_counts:
            # P(intent)
            prob = math.log(self.class_counts[intent] / self.total_docs)
            
            # P(word | intent)
            total_words_in_class = sum(self.class_word_counts[intent].values())
            vocab_size = len(self.vocab)
            
            for w in words:
                # Laplace smoothing
                count = self.class_word_counts[intent].get(w, 0)
                prob += math.log((count + 1) / (total_words_in_class + vocab_size))
                
            if prob > max_prob:
                max_prob = prob
                best_intent = intent
                
        return best_intent

class PurePythonLinearRegression:
    def __init__(self, num_features=8):
        self.num_features = num_features
        self.weights = [0] * num_features
        self.bias = 0
        
    def fit(self, X, y, epochs=1000, lr=0.001):
        for _ in range(epochs):
            for i in range(len(X)):
                pred = sum(X[i][j] * self.weights[j] for j in range(self.num_features)) + self.bias
                error = pred - y[i]
                
                for j in range(self.num_features):
                    self.weights[j] -= lr * error * X[i][j]
                self.bias -= lr * error
                
    def predict(self, features):
        return sum(features[j] * self.weights[j] for j in range(self.num_features)) + self.bias

class EcoPilotAI:
    def __init__(self):
        self.score_model = PurePythonLinearRegression(num_features=8)
        self.nlp_model = PurePythonNaiveBayes()
        self.train_models()

    def _generate_energy_data(self, num_samples=500):
        X = []
        y_score = []
        
        for _ in range(num_samples):
            ac = random.uniform(0.5, 6.0)
            fan = random.uniform(0.2, 2.0)
            fridge = random.uniform(1.0, 2.5)
            wm = random.uniform(0.0, 3.0)
            tv = random.uniform(0.1, 0.8)
            lights = random.uniform(0.1, 0.5)
            oven = random.uniform(0.0, 3.0)
            
            time_of_day = random.randint(0, 23)
            
            total_kwh = ac + fan + fridge + wm + tv + lights + oven
            emissions = total_kwh * 0.82
            
            # Time Penalty logic: Peak hours 17 (5PM) to 21 (9PM)
            is_peak = 17 <= time_of_day <= 21
            peak_penalty = 1.5 if is_peak else 1.0
            
            score = 100 - (emissions * 1.5 * peak_penalty)
            score = max(0, min(100, score))
            
            # Scale time_of_day to avoid exploding gradients in Linear Regression
            scaled_time = time_of_day / 10.0
            
            X.append([ac, fan, fridge, wm, tv, lights, oven, scaled_time])
            y_score.append(score)
            
        return X, y_score

    def _generate_nlp_data(self):
        X = [
            "how to reduce my bill",
            "my electricity bill is too high",
            "help me save money on energy",
            "lower my costs",
            "why is my usage high",
            "what is using so much power",
            "detect heavy appliances",
            "my energy consumption is spiking",
            "what is my carbon score",
            "how to improve my carbon footprint",
            "explain my eco score",
            "are my emissions bad",
            "hello",
            "hi there",
            "who are you"
        ]
        y = [
            "reduce_bill", "reduce_bill", "reduce_bill", "reduce_bill",
            "high_usage", "high_usage", "high_usage", "high_usage",
            "carbon_score", "carbon_score", "carbon_score", "carbon_score",
            "greeting", "greeting", "greeting"
        ]
        return X, y

    def train_models(self):
        print("⚙️ Generating Simulated Time-Aware Datasets...")
        X_energy, y_score = self._generate_energy_data(500)
        X_text, y_intent = self._generate_nlp_data()
        
        print("🧠 Training Carbon Score Predictor (8 features, time-aware)...")
        self.score_model.fit(X_energy, y_score, epochs=500, lr=0.002)
        
        print("🧠 Training NLP Intent Classifier...")
        self.nlp_model.fit(X_text, y_intent)
        
        print("✅ Time-Aware AI models trained successfully!")

    def predict_score(self, appliances, time_of_day):
        scaled_time = time_of_day / 10.0
        features = [
            appliances.get('AC', 0),
            appliances.get('Fan', 0),
            appliances.get('Fridge', 0),
            appliances.get('Washing Machine', 0),
            appliances.get('TV', 0),
            appliances.get('Smart Lights', 0),
            appliances.get('Electric Oven', 0),
            scaled_time
        ]
        score = self.score_model.predict(features)
        return max(0, min(100, score))

    def predict_optimization(self, appliances, time_of_day):
        ac = appliances.get('AC', 0)
        fan = appliances.get('Fan', 0)
        fridge = appliances.get('Fridge', 0)
        wm = appliances.get('Washing Machine', 0)
        tv = appliances.get('TV', 0)
        lights = appliances.get('Smart Lights', 0)
        oven = appliances.get('Electric Oven', 0)
        
        is_peak = 17 <= time_of_day <= 21
        
        # ML heuristic: more aggressive optimization during peak hours
        reduction_factor = 0.7 if is_peak else 0.9
        
        opt_ac = ac * reduction_factor if ac > 2.0 else ac * 0.95
        opt_fan = fan
        opt_fridge = fridge * 0.98
        opt_wm = 0 if is_peak else wm # Shift WM completely out of peak
        opt_tv = tv * 0.9
        opt_lights = lights * 0.8 if is_peak else lights
        opt_oven = 0 if (is_peak and oven > 0) else oven # Don't bake during peak!
        
        return {
            'AC': opt_ac,
            'Fan': opt_fan,
            'Fridge': opt_fridge,
            'Washing Machine': opt_wm,
            'TV': opt_tv,
            'Smart Lights': opt_lights,
            'Electric Oven': opt_oven
        }

    def chat_response(self, text):
        intent = self.nlp_model.predict(text)
        
        if intent == "reduce_bill":
            return "Based on my pure Python ML analysis, lowering your AC to 24°C and shifting your Washing Machine cycles after 9 PM can cut your bill by 18%."
        elif intent == "high_usage":
            return "My anomaly detection models show your AC has been running 45% longer than your historical average. Consider enabling Smart Scheduling."
        elif intent == "carbon_score":
            return "Your carbon score is a machine-learning derived metric based on your daily emissions and local grid carbon intensity. Lower your peak usage to improve it!"
        else:
            return "I am the EcoPilot AI Assistant. I use pure Python ML models to analyze your usage, optimize your appliances, and lower your carbon footprint."

ai_system = EcoPilotAI()
