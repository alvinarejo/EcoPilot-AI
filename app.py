from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from ai_engine import ai_system

app = Flask(__name__)
CORS(app)

CO2_PER_KWH = 0.82

def get_breakdown_and_emissions(appliances):
    total_kwh = sum(appliances.values())
    emissions = total_kwh * CO2_PER_KWH
    
    breakdown = []
    for name, kwh in appliances.items():
        breakdown.append({
            "name": name,
            "kwh": float(kwh),
            "co2": float(kwh * CO2_PER_KWH)
        })
    return total_kwh, emissions, breakdown

@app.route('/data', methods=['GET'])
def get_data():
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    time_of_day = data.get('time_of_day', 12)
    appliances = data['appliances']
    total_kwh, emissions, breakdown = get_breakdown_and_emissions(appliances)
    
    # 🧠 AI INFERENCE: Predict Score (Time-Aware)
    score = ai_system.predict_score(appliances, time_of_day)
    
    history = [
        {"day": "Mon", "co2": emissions * 1.1},
        {"day": "Tue", "co2": emissions * 0.95},
        {"day": "Wed", "co2": emissions * 1.05},
        {"day": "Thu", "co2": emissions * 0.9},
        {"day": "Fri", "co2": emissions * 1.2},
        {"day": "Sat", "co2": emissions * 0.8},
        {"day": "Sun", "co2": emissions}
    ]
    
    bad_habits = []
    if appliances.get('AC', 0) > 3.0:
        bad_habits.append("AI Anomaly: AC usage is significantly higher than your historical average.")
    if (17 <= time_of_day <= 21) and appliances.get('Washing Machine', 0) > 0:
        bad_habits.append(f"AI Alert: You are running the Washing Machine during peak hour ({time_of_day}:00). This drastically hurts your Carbon Score.")
        
    return jsonify({
        "current": {
            "total_kwh": round(total_kwh, 2),
            "emissions_kg": round(emissions, 2),
            "score": round(score),
            "breakdown": breakdown,
            "time_of_day": time_of_day
        },
        "history": history,
        "habits": bad_habits
    })

@app.route('/optimize', methods=['POST'])
def optimize():
    with open('data.json', 'r') as f:
        data = json.load(f)
        
    time_of_day = data.get('time_of_day', 12)
    appliances = data['appliances']
    _, before_co2, _ = get_breakdown_and_emissions(appliances)
    before_score = ai_system.predict_score(appliances, time_of_day)
    
    # 🧠 AI INFERENCE: Predict Optimal Usage (Time-Aware)
    optimized_appliances = ai_system.predict_optimization(appliances, time_of_day)
    
    _, after_co2, _ = get_breakdown_and_emissions(optimized_appliances)
    after_score = ai_system.predict_score(optimized_appliances, time_of_day)
    
    savings_percent = ((before_co2 - after_co2) / before_co2) * 100 if before_co2 > 0 else 0
    trees_saved = (before_co2 - after_co2) * 365 / 21
    km_saved = (before_co2 - after_co2) * 365 / 0.192
    
    return jsonify({
        "before": {
            "co2": round(before_co2, 2),
            "score": round(before_score)
        },
        "after": {
            "co2": round(after_co2, 2),
            "score": round(after_score)
        },
        "savings_percent": round(savings_percent, 1),
        "impact": {
            "trees_saved": round(trees_saved, 1),
            "km_saved": round(km_saved)
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').lower()
    
    # 🧠 AI INFERENCE: NLP Intent Classification
    response = ai_system.chat_response(message)
        
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
