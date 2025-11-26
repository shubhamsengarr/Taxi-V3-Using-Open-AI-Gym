# Taxi-v3 Q-Learning Using OpenAI Gym and Streamlit Dashboard

This project implements the Q-Learning algorithm on the classic Taxi-v3 environment from OpenAI Gym. It includes a full training pipeline, dataset generation, and an interactive Streamlit dashboard for visual step-by-step visualization of the agent’s behavior.

## Features

### Q-Learning Training Script (`taxi_with_dataset.py`)
- Implements Q-Learning with epsilon-greedy exploration
- Trains the Taxi-v3 agent for configurable episodes
- Saves:
  - taxi_q_table.npy
  - taxi_transitions_train.csv
  - taxi_episode_rewards.csv
  - taxi_policy_demo.csv
  - training_curve.png

### Streamlit Dashboard (`taxi_dashboard.py`)
- Visual 5x5 grid showing Taxi, Passenger, Destination, and Landmarks
- Reset, Step, Play, Pause controls
- Adjustable speed
- Episode statistics: steps, rewards, penalties, last action
- Training curve display
- Transition log viewer
- Dataset download options

## Project Structure

.
├── taxi_with_dataset.py  
├── taxi_dashboard.py  
├── requirements.txt  
└── taxi_output/  
    ├── taxi_q_table.npy  
    ├── taxi_transitions_train.csv  
    ├── taxi_episode_rewards.csv  
    ├── taxi_policy_demo.csv  
    └── training_curve.png  

## Installation

Install dependencies:
pip install -r requirements.txt

## Training the Agent
python taxi_with_dataset.py

## Running the Dashboard
streamlit run taxi_dashboard.py

## Q-Learning Formula
Q(s, a) = Q(s, a) + α * [ r + γ * max(Q(s', a')) - Q(s, a) ]

Where:
- α = learning rate  
- γ = discount factor  
- ε-greedy for exploration  

## Requirements

- Python 3.8+
- Streamlit  
- Gymnasium (+ toy_text)  
- NumPy  
- Pandas  
- Matplotlib  

## Author
Shubham Sengar
