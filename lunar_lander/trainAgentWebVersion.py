from flask import Flask, Response, request, jsonify, send_from_directory
import threading
import time
import json
import queue
import torch  
from train_agent import TrainAgent
from hparams import HParams
import gymnasium as gym
from DQNAgent import DQNAgent

app = Flask(__name__)

clients = []
training_thread = None
training_stop_flag = False

def train_agent(data_queue):
    global training_stop_flag

    # Setup env and agent
    env = gym.make("LunarLander-v3")
    hparams = HParams()
    agent = DQNAgent(env=env, hparams=hparams)

    num_episodes = 100
    for episode in range(num_episodes):
        if training_stop_flag:
            break

        state, _ = env.reset()
        state = agent.state_processor(state)
        total_reward = 0
        total_loss = 0
        step = 0

        while True:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            reward_tensor = torch.tensor([reward], device=agent.device)
            next_state = None if done else agent.state_processor(obs)

            agent.memory.push(state, action, next_state, reward_tensor)
            state = next_state
            total_reward += reward

            loss = agent.optimize_model()
            if loss is not None:
                total_loss += loss

            agent.soft_update()
            step += 1

            if done:
                break

        # Send metrics to frontend
        data = {
            "episode": episode,
            "reward": total_reward,
            "step": step,
            "loss": round(total_loss / max(step, 1), 4),
            "status": "running" if episode < num_episodes - 1 else "complete"
        }
        data_queue.put(data)

    training_stop_flag = False

# SSE stream endpoint
@app.route('/stream/metrics')
def stream_metrics():
    def event_stream(q):
        try:
            while True:
                data = q.get()
                yield f"data: {json.dumps(data)}\n\n"
        except GeneratorExit:
            pass

    q = queue.Queue()
    clients.append(q)
    return Response(event_stream(q), mimetype='text/event-stream')

# Start training
@app.route('/train/start', methods=['POST'])
def start_training():
    global training_thread, training_stop_flag

    if training_thread and training_thread.is_alive():
        return jsonify({"message": "Training already running"}), 400

    training_stop_flag = False
    data_queue = queue.Queue()

    def training_func():
        train_agent(data_queue)
        for q in clients:
            q.put({"status": "complete"})

    training_thread = threading.Thread(target=training_func)
    training_thread.start()

    def broadcaster():
        while training_thread.is_alive():
            try:
                data = data_queue.get(timeout=0.5)
                for q in clients:
                    q.put(data)
            except queue.Empty:
                continue
    threading.Thread(target=broadcaster, daemon=True).start()

    return jsonify({"message": "Training started"})

@app.route('/train/stop', methods=['POST'])
def stop_training():
    global training_stop_flag
    training_stop_flag = True
    return jsonify({"message": "Training stopping"})

@app.route('/')
def index():
    return send_from_directory('.', './web/trainAgentWebVersion.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
