from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import torch
import torch.nn as nn
import logging
# Define server
app = Flask(__name__)

# Hyperparameters
MODEL_PATH = './Data/Model/Connect4_Agent_Model.pth'
GAMMA = 0.99

# Load Connect4 environment and DQN model
class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        return self.board

    def get_board(self):
        return self.board

    def is_valid_action(self, action):
        if not (0 <= action < self.board.shape[1]):
            return False
        return self.board[0, action] == 0

    def make_move(self, action):
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}")
        for row in range(5, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        self.current_player = 3 - self.current_player

    def check_winner(self):
        # Horizontal, vertical, and diagonal checks
        for row in range(6):
            for col in range(7 - 3):
                if (self.board[row, col] != 0 and
                        np.all(self.board[row, col:col + 4] == self.board[row, col])):
                    return self.board[row, col]
        for row in range(6 - 3):
            for col in range(7):
                if (self.board[row, col] != 0 and
                        np.all(self.board[row:row + 4, col] == self.board[row, col])):
                    return self.board[row, col]
        for row in range(6 - 3):
            for col in range(7 - 3):
                if (self.board[row, col] != 0 and
                        all(self.board[row + i, col + i] == self.board[row, col] for i in range(4))):
                    return self.board[row, col]
        for row in range(6 - 3):
            for col in range(3, 7):
                if (self.board[row, col] != 0 and
                        all(self.board[row + i, col - i] == self.board[row, col] for i in range(4))):
                    return self.board[row, col]
        return 0

    def is_draw(self):
        return np.all(self.board != 0)

    def get_valid_actions(self):
        return [col for col in range(7) if self.is_valid_action(col)]



class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Layer definitions
        self.fc1 = nn.Linear(6 * 7, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 7)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 6 * 7)  # Flatten the input

        # Layer forward passes with condition for batch normalization
        x = self.activation(self.bn1(self.fc1(x)) if x.size(0) > 1 else self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = self.activation(self.bn2(self.fc2(x)) if x.size(0) > 1 else self.fc2(x))
        x = self.activation(self.bn3(self.fc3(x)) if x.size(0) > 1 else self.fc3(x))
        x = self.dropout2(x)  # Apply dropout
        x = self.activation(self.bn4(self.fc4(x)) if x.size(0) > 1 else self.fc4(x))
        x = self.activation(self.bn5(self.fc5(x)) if x.size(0) > 1 else self.fc5(x))
        x = self.dropout3(x)  # Apply dropout
        x = self.activation(self.bn6(self.fc6(x)) if x.size(0) > 1 else self.fc6(x))
        return self.fc7(x)
# Load the model

import os
import torch
from collections import deque

def load_model(
    model_path,
    device = torch.device("cpu")
):
    """
    Loads a model checkpoint and initializes components if necessary.

    Args:
        model_path (str): Path to the checkpoint file.
        policy_net (torch.nn.Module): Policy network instance.
        target_net (torch.nn.Module): Target network instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        replay_buffer (deque): Replay buffer instance.
        learning_rate (float): Learning rate for the optimizer.
        buffer_size (int): Maximum size of the replay buffer.
        logger (logging.Logger): Logger instance for logging.
        device (str): Device for model loading ('cpu' or 'cuda').
        model_class (type): Class of the model to initialize networks (required if networks are None).

    Returns:
        tuple: Initialized policy_net, target_net, optimizer, replay_buffer, and start_episode.
    """
    if logger is None:

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    try:

        # Load checkpoint if it exists
        if os.path.exists(model_path):
            logger.info(f"Checkpoint file found at {model_path}, loading...")
            policy_net = DQN().to(device)
            policy_net.load_state_dict(policy_net.state_dict())
            policy_net.eval()

    except Exception as e:
        RuntimeError(f"Failed to load the model: {e}.")


    return policy_net

# Initialize game environment
env = Connect4()

@app.route("/reset", methods=["POST"])
def reset():
    """
    Reset the game.
    """
    env.reset()
    return jsonify({"message": "Game reset", "board": env.get_board().tolist()})

@app.route("/move", methods=["POST"])
def move():
    """
    Handle a move by the player.
    """
    device = torch.device("cpu")
    policy_net=load_model(MODEL_PATH,device)
    data = request.get_json()
    action = data.get("action")
    if action is None or not env.is_valid_action(action):
        return jsonify({"error": "Invalid move"}), 400

    # Player's move
    env.make_move(action)
    if env.check_winner():
        return jsonify({"winner": "player", "board": env.get_board().tolist()})
    if env.is_draw():
        return jsonify({"winner": "draw", "board": env.get_board().tolist()})

    # AI's move
    state = torch.tensor(env.get_board(), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state)
    valid_actions = env.get_valid_actions()
    valid_q_values = {action: q_values[0, action].item() for action in valid_actions}
    ai_action = max(valid_q_values, key=valid_q_values.get)

    env.make_move(ai_action)
    if env.check_winner():
        return jsonify({"winner": "ai", "board": env.get_board().tolist()})
    if env.is_draw():
        return jsonify({"winner": "draw", "board": env.get_board().tolist()})

    return jsonify({"board": env.get_board().tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
