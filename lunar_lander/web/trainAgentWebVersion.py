# to run the webversion
# using conda enviornment
# pip install flask gym numpy
# python trainAgentWebVersion.py


from flask import Flask, Response, request, jsonify, send_from_directory
import threading
import time
import json
import queue

app = Flask(__name__)

clients = []
training_thread = None
training_stop_flag = False

def train_agent(data_queue):
    global training_stop_flag
    episode = 0
    step = 0
    while not training_stop_flag and episode < 100:
        # 模拟数据
        reward = episode * 5 + (episode % 3) * 2
        loss = 1 / (episode + 1) + 0.1 * (episode % 5)  # 模拟 loss
        
        data = {
            "episode": episode,
            "reward": reward,
            "step": step,
            "loss": loss,
            "status": "running" if episode < 99 else "complete"
        }
        data_queue.put(data)
        
        episode += 1
        step += 100  # 假设每个 episode 有 100 steps
        time.sleep(1)  # 模拟训练耗时
    
    training_stop_flag = False

# SSE事件流
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

# 启动训练
@app.route('/train/start', methods=['POST'])
def start_training():
    global training_thread, training_stop_flag

    if training_thread and training_thread.is_alive():
        return jsonify({"message": "Training already running"}), 400

    training_stop_flag = False
    data_queue = queue.Queue()

    def training_func():
        train_agent(data_queue)
        # 训练结束，给所有客户端发送complete状态
        for q in clients:
            q.put({"status": "complete"})

    training_thread = threading.Thread(target=training_func)
    training_thread.start()

    # 启动线程读取数据队列，推送给客户端
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

# 停止训练
@app.route('/train/stop', methods=['POST'])
def stop_training():
    global training_stop_flag
    training_stop_flag = True
    return jsonify({"message": "Training stopping"})

# 主页，返回dashboard.html
@app.route('/')
def index():
    return send_from_directory('.', 'trainAgentWebVersion.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
