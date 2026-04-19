# EcoPilot AI 🌍⚡

EcoPilot AI is an advanced, intelligent energy and carbon optimization system designed to transform standard smart homes into proactive, eco-friendly Virtual Power Plants (VPP).

Unlike traditional dashboards that simply report historical data, EcoPilot acts as an **Active Edge AI Agent**. It learns your household habits, negotiates with neighboring homes, and intelligently schedules appliance usage to drastically reduce your carbon footprint and energy bills—all while processing data 100% locally for maximum privacy.

## ✨ Core Features

*   **Neighborhood Virtual Power Plant (VPP)**: Peer-to-peer swarm intelligence. EcoPilot actively syncs with neighboring nodes to route excess solar energy and intelligently distribute loads during peak grid strain.
*   **Zero-Cloud Edge AI**: Custom Machine Learning algorithms built entirely from scratch in Pure Python. Your sensitive energy data never leaves your local network. No massive dependencies, no cloud latency.
*   **Time-Aware Carbon Scoring**: The predictive ML model ingests the time of day, aggressively penalizing energy consumption during peak grid hours (5 PM - 9 PM) to encourage grid-flattening habits.
*   **1-Click AI Auto-Pilot**: Run instant Monte Carlo-style optimizations to find the statistically perfect reduction matrix for your heavy appliances (AC, EV Charger, Washing Machine) without compromising comfort.
*   **Natural Language Chatbot**: A custom Naive Bayes NLP intent classifier that understands human questions ("Why is my bill so high?") and provides hyper-personalized, data-driven advice.

## 🛠️ Technology Stack

*   **Frontend**: React.js, Vite, Tailwind CSS, Lucide Icons, Chart.js.
*   **Backend**: Python, Flask, Flask-CORS.
*   **AI Engine**: Custom Pure-Python implementations of Gradient Descent Linear Regression and Naive Bayes Classifiers.
*   **Database**: Mocked JSON (`data.json`) for seamless hackathon demonstrations and dynamic state simulation.

## 🚀 How to Run Locally

### 1. Start the Flask AI Backend
```bash
cd backend
python app.py
```
*(The backend will automatically generate synthetic datasets and train the ML models upon startup!)*

### 2. Start the React Dashboard
Open a new terminal and run:
```bash
cd frontend
npm install
npm run dev
```
*(Access the dashboard at `http://localhost:5173`)*

## 🔮 Future Roadmap
*   Direct hardware integration via MQTT for IoT smart plugs.
*   Persistent SQLite database for historical AI model weighting.
*   Live Grid-Carbon Intensity API fetching (e.g., WattTime).

---
*Built for the future of sustainable energy.*
