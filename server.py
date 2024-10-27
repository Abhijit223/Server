import flwr as fl
import torch
import torch.nn as nn
import time
import socket
from collections import OrderedDict
from flask import Flask, jsonify
import threading

# Define the model (global and local)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Utility: Check if a port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0

# Utility: Find a free port starting from a base port
def find_free_port(starting_port=8080):
    port = starting_port
    while is_port_in_use(port):
        port += 1
    return port

# Task Initiator (Server-side)
class TaskInitiator:
    def __init__(self):
        self.global_model = Net()

    def decentralized_fedavg(self, client_blocks):
        # Perform FedAvg on parameters from all clients
        aggregated_params = self.aggregate(client_blocks)
        self.update_global_model(aggregated_params)

    def aggregate(self, client_blocks):
        # Simulated FedAvg (averaging parameters from client blocks)
        total_clients = len(client_blocks)
        agg_params = [torch.zeros_like(param) for param in self.global_model.state_dict().values()]
        for block in client_blocks:
            for idx, param in enumerate(block["client_params"]):
                agg_params[idx] += torch.tensor(param)
        agg_params = [param / total_clients for param in agg_params]
        return agg_params

    def update_global_model(self, aggregated_params):
        params_dict = zip(self.global_model.state_dict().keys(), aggregated_params)
        state_dict = OrderedDict({k: v for k, v in params_dict})
        self.global_model.load_state_dict(state_dict, strict=True)

# Flower server setup with custom strategy
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # Create a blockchain block with parameters from clients
        client_blocks = [result[1] for result in results]
        print(f"Round {rnd}: Aggregating {len(client_blocks)} client parameters")

        # Decentralized FedAvg on client blocks
        task_initiator = TaskInitiator()
        task_initiator.decentralized_fedavg(client_blocks)

        # Continue with the regular FedAvg aggregation
        return super().aggregate_fit(rnd, results, failures)

# Global variable to store the port
flower_port = None

# Start Flower server
def start_flower_server():
    global flower_port
    # Find a free port dynamically
    flower_port = find_free_port(8080)
    print(f"Starting Flower server on free port: {flower_port}")

    # Use the found port
    strategy = CustomFedAvg(min_available_clients=2)
    fl.server.start_server(server_address=f"127.0.0.1:{flower_port}", strategy=strategy)

# Flask setup to control Flower server
app = Flask(__name__)

@app.route('/', methods=['GET'])
def start_server():
    global flower_port
    # Run the Flower server in a separate thread
#     server_thread = threading.Thread(target=start_flower_server)
#     server_thread.start()
    start_flower_server()
    # Return the Flower server's port to the client
    print("flower sever started")
    return jsonify({"status": "Flower server started", "port": flower_port},200)

if __name__ == "__main__":
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001)
