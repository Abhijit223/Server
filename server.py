import flwr as fl
import torch
import socket
from collections import OrderedDict

# Global Model Definition
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

# Step 2: Select Client Based on Resources
def client_selection(clients):
    selected_clients = []
    for client in clients:
        if client["resources"]["gpus"] >= 2 and client["resources"]["computation_power"] > 10:
            selected_clients.append(client)
    return selected_clients

# Step 5: Block Creation (Blockchain Block)
def create_block(client_params):
    block = {"client_parameters": client_params}
    return block

# Step 6: Federated Averaging and Block Broadcast
def federated_avg(blocks):
    avg_parameters = []
    for i in range(len(blocks[0]["client_parameters"])):
        tensors = [torch.tensor(block["client_parameters"][i]) for block in blocks]
        avg_param = torch.mean(torch.stack(tensors), dim=0)
        avg_parameters.append(avg_param)
    return avg_parameters

# Step 7: Global Model Update and Continue Training
def update_global_model(global_model, avg_parameters):
    global_model.load_state_dict(OrderedDict(zip(global_model.state_dict().keys(), avg_parameters)))

# Utility: Check if the port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0

# Utility: Find a free port dynamically
def find_free_port(starting_port=8080):
    port = starting_port
    while is_port_in_use(port):
        port += 1
    return port

# FL Flower Server Initialization
def start_server():
    # Use num_rounds in the strategy, not the config
    strategy = fl.server.strategy.FedAvg(min_available_clients=2)
    
    # Find an available port dynamically
    free_port = find_free_port(8080)
    print(f"Starting server on free port: {free_port}")
    
    # Start the server without num_rounds in the config (handled by strategy)
    fl.server.start_server(server_address=f"127.0.0.1:{free_port}", strategy=strategy)

# Main FL Simulation (Server-side)
if __name__ == "__main__":
    clients = [
        {"resources": {"gpus": 4, "computation_power": 20}},
        {"resources": {"gpus": 1, "computation_power": 5}},
        {"resources": {"gpus": 3, "computation_power": 15}},
    ]

    selected_clients = client_selection(clients)
    global_model = Net()

    client_params = []
    for client_info in selected_clients:
        # Client-side simulation happens on client
        pass

    blocks = [create_block(params) for params in client_params]
    avg_parameters = federated_avg(blocks)
    update_global_model(global_model, avg_parameters)

    start_server()
