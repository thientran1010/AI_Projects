# 🚀 Lunar Lander – Deep Q-Learning Agent

An implementation of a Deep Q-Network (DQN) agent that learns to land a spacecraft autonomously in the OpenAI Gym `LunarLander-v2` environment. The project uses reinforcement learning techniques to train the agent through trial and error, optimizing for soft landings and fuel efficiency.

---

## 📚 Table of Contents

- [About the Project](#about-the-project)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## 🧠 About the Project

This project applies Deep Reinforcement Learning to solve the Lunar Lander problem using the DQN algorithm. The agent learns to control the spacecraft by receiving rewards and penalties based on its actions. Over time, it learns an optimal policy for successful landings.

The project was completed as part of a university software engineering course, with team members collaborating on architecture, training, evaluation, and user experience.

---

## 🛠️ Architecture Overview

- **Environment**: LunarLander-v2 (OpenAI Gym)
- **Algorithm**: Deep Q-Learning (DQN)
- **Libraries**:
  - `PyTorch` – Neural network & training
  - `NumPy` – Math and matrix operations
  - `Matplotlib` – Reward visualization
  - `Gym` – RL environment

**Key Components**:
- `DQNAgent` – Core agent logic (Q-learning)
- `ReplayMemory` – Experience replay buffer
- `QNetwork` – Neural net approximator
- `Trainer` – Training loop for the agent
- `Visualizer` – Episode performance tracking

---

## ⚙️ Installation

> Python 3.8 or higher is required.

1. **Clone the repo**:
```bash
cd $PATH_TO_PROJECT

#Highly Suggest that a fork to the Git Repository is created and you use the forked linked to clone the project.
git clone $FORKED_REPO_URI

cd lunar_lander_rl

git remote add upstream https://github.com/thientran1010/AI_Projects
git fetch upstream
git pull upstream main
```

2. **Install Dependencies**:
```bash
cd $WORKSPACE

pip install -r requirements.txt
```

*If you encounter issues with box2d-py, install SWIG and ensure it’s in your PATH, or use Gym 0.21.0 for easier compatibility.*

---

## 🚀 Usage

### 🧪 To train the agent:
```bash
cd $WORKSPACE/lunar_lander

python train.py
```

### 🎮 To test a trained agent:
```bash
cd $WORKSPACE/lunar_lander

python lunar_test.py
```

### 📊 To view training progress:

Reward plots are saved under /plots after each training session.

---

## 🗂 Project Structure

<!-- TODO: Maybe rethink strucutre to look like this or change this to mimic the existing structure -->

```bash
lunar_lander_rl/
└── lunar_lander/
    ├── agents/             # Agent, replay memory, networks
    ├── env/                # Environment and wrappers
    ├── scripts/
    │   ├── train.py
    │   └── test.py
    ├── models/             # Saved models
    ├── plots/              # Reward and loss visualizations
    ├── utils.py
    ├── README.md
    └── requirements.txt
```

---

🤝 Contributing

This project was built by a student team and is open to further development. If you'd like to contribute:

1. Fork the repository
2. Create your feature branch (git checkout -b feature/YourFeature)
3. Commit your changes (git commit -m "Add feature")
4. Push to the branch (git push origin feature/YourFeature)
5. Open a Pull Request

---

✨ Acknowledgements

- OpenAI Gym Team
- PyTorch Contributors




## 🆕 New Features

### Live Monitoring Dashboard
- Real-time episode tracking and performance metrics
- Visual demo with live data export to JSON
- Web dashboard for monitoring training progress

**Usage:**
```bash
# Terminal 1: Start web server
python3 -m http.server 8080

# Terminal 2: Run visual demo  
python3 visual_demo.py

# Browser: Open dashboard
http://localhost:8080/dashboard.html