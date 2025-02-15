from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import logging
import json
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[SERVER] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    total_rounds = config.get('num_rounds', 5)  # Default to 5 if not specified
    PARTICIPATION_THRESHOLD = config.get('participation_threshold', 0.5)  # Default to 0.5 if not specified
    logger.info(f"Loaded configuration: {total_rounds} rounds, participation threshold: {PARTICIPATION_THRESHOLD}")
except FileNotFoundError:
    logger.warning("config.json not found, using default: 5 rounds, 0.5 threshold")
    total_rounds = 5
    PARTICIPATION_THRESHOLD = 0.5
except json.JSONDecodeError:
    logger.error("Error parsing config.json, using default: 5 rounds, 0.5 threshold")
    total_rounds = 5
    PARTICIPATION_THRESHOLD = 0.5

app = Flask(__name__)
num_clients = 0  # Track number of connected clients
client_counter = 0  # Counter for generating unique client IDs
current_round = 1  # Current round number
client_responses = set()
waiting_clients = set()
clients = set()  # Store active client IDs

# Load test data for global model evaluation
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

def create_global_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Global variables
global_model = create_global_model()
global_weights = global_model.get_weights()
round_updates = []       # Store weights from participating clients
accuracy_log = []

# Global variable to store client gradients
client_gradients = {}

def evaluate_global_model():
    """Evaluate the global model and save metrics to CSV."""
    global accuracy_log
    
    # Get predictions
    predictions = global_model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    overall_accuracy = np.mean(y_pred == y_test.flatten())
    per_class_accuracy = []
    
    for class_idx in range(10):  # CIFAR-10 has 10 classes
        class_mask = y_test.flatten() == class_idx
        class_accuracy = np.mean(y_pred[class_mask] == y_test.flatten()[class_mask])
        per_class_accuracy.append(class_accuracy)
    
    # Create log entry
    log_entry = {
        'round': current_round,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_accuracy': overall_accuracy
    }
    # Add per-class accuracies
    for i, acc in enumerate(per_class_accuracy):
        log_entry[f'class_{i}_accuracy'] = acc
    
    # Append to log
    accuracy_log.append(log_entry)
    
    # Save to CSV
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    csv_path = os.path.join(log_dir, 'global_model_accuracy.csv')
    pd.DataFrame(accuracy_log).to_csv(csv_path, index=False)
    
    logger.info(f"Round {current_round} accuracies saved to {csv_path}")
    
    return overall_accuracy, per_class_accuracy

def aggregate_models():
    """Aggregate the model updates and evaluate the global model."""
    global global_weights, current_round
    
    if round_updates:
        # Perform FedAvg aggregation
        new_weights = []
        for weights_tuple in zip(*round_updates):
            new_weights.append(np.mean(weights_tuple, axis=0))
        
        global_weights = new_weights
        global_model.set_weights(global_weights)
        
        # After aggregation, update global model and evaluate
        global_model.set_weights(global_weights)
        evaluate_global_model()
        
        current_round += 1
        logger.info(f"Starting round {current_round}")
        
        # Select clients for the next round using LongFed strategy
        global selected_clients
        num_clients_to_select = max(int(num_clients*(1 - PARTICIPATION_THRESHOLD)), 1)
        selected_clients = select_clients_longfed(
            all_clients=clients,  # Assuming 'clients' is a list of all client objects
            num_selected=num_clients_to_select,  # Number of clients to select
            fairness_queues=fairness_queues,
            epsilon=0.1,  # Example value for epsilon
            delta=0.1,  # Example value for delta
            V=0.5  # Example value for V
        )

        logger.info(f"Selected clients for round {current_round}: {selected_clients}")

    # Clear updates for the next round
    round_updates.clear()

def compute_gradient_error(selected_clients, all_clients):
    """
    Compute the upper bound of gradient estimation error.
    D_UB(S_t) = sum(min(||grad_i - grad_j||^2 for j in S_t))
    """
    total_error = 0

    for client_id in all_clients:
        client_gradient = client_gradients.get(client_id, None)
        
        # Skip if the client has no recorded gradients
        if client_gradient is None or len(client_gradient) == 0:
            continue  

        # Flatten or concatenate gradients to form a single array
        client_gradient_np = np.concatenate([np.ravel(g) for g in client_gradient])

        min_error = float('inf')  # Set a high initial value

        for sel_id in selected_clients:
            sel_gradient = client_gradients.get(sel_id, None)

            if sel_gradient is not None and len(sel_gradient) > 0:
                sel_gradient_np = np.concatenate([np.ravel(g) for g in sel_gradient])

                # Ensure shapes match before subtraction
                if client_gradient_np.shape == sel_gradient_np.shape:
                    error = np.sum((client_gradient_np - sel_gradient_np) ** 2)
                    min_error = min(min_error, error)
        
        # If no valid comparison found, set a high penalty error
        total_error += min_error if min_error != float('inf') else 0

    return total_error



def compute_fairness_violation(client, selected_clients, fairness_queues, epsilon, delta):
    """
    Compute fairness violation based on selection frequencies.
    """
    if not hasattr(client, 'data_distribution') or client.data_distribution is None:
        return 0, 0, None  # Skip clients with no data

    similar_clients = [c for c in selected_clients if hasattr(c, 'data_distribution') and 
                       np.linalg.norm(client.data_distribution - c.data_distribution) <= epsilon]
    
    if not similar_clients:
        return 0, 0, None  # No similar client found, no fairness violation
    
    ref_client = min(similar_clients, key=lambda c: abs(fairness_queues.get(c.client_id, 0) - fairness_queues.get(client.client_id, 0)))
    
    m_i = fairness_queues.get(client.client_id, 0) - fairness_queues.get(ref_client.client_id, 0) - delta
    n_i = fairness_queues.get(ref_client.client_id, 0) - fairness_queues.get(client.client_id, 0) - delta
    
    return m_i, n_i, ref_client



def update_virtual_queues(fairness_queues, selected_clients, reference_clients, delta):
    """
    Update fairness virtual queues for all clients.
    """
    for client, ref_client in zip(selected_clients, reference_clients):
        if ref_client:
            fairness_queues[client] = max(fairness_queues.get(client, 0) + 1 
                                                    - fairness_queues.get(ref_client, 0) - delta, 0)


def select_clients_longfed(all_clients, num_selected, fairness_queues, epsilon, delta, V):
    """
    Select clients using LongFed strategy, balancing fairness and gradient approximation.
    """
    selected_clients = []
    remaining_clients = all_clients.copy()
    reference_clients = []

    while len(selected_clients) < num_selected:
        scores = {}
        client_references = {}

        for client_id in remaining_clients:
            # Compute fairness violation
            m_i, n_i, ref_client = compute_fairness_violation(client_id, selected_clients, fairness_queues, epsilon, delta)

            # Compute gradient estimation error
            grad_error = compute_gradient_error(selected_clients + [client_id], all_clients)

            # Ensure fairness queue entries exist
            fairness_value = fairness_queues.get(client_id, 0)

            # Compute overall score
            scores[client_id] = (1 - V) * (fairness_value * m_i + fairness_value * n_i) + V * grad_error
            client_references[client_id] = ref_client

        # Select client with minimum score
        best_client_id = min(scores, key=scores.get)
        selected_clients.append(best_client_id)
        reference_clients.append(client_references.get(best_client_id, None))
        remaining_clients.remove(best_client_id)

    # Update fairness queues for selected clients
    update_virtual_queues(fairness_queues, selected_clients, reference_clients, delta)

    return selected_clients


# Global variables for LongFed
selected_clients = []
fairness_queues = {}

# Modify should_participate to check if the client is in selected_clients

def should_participate(client_id):
    # Check if the client ID is in the selected clients list
    return int(client_id) in selected_clients

@app.route('/poll', methods=['GET'])
def poll():
    """
    Clients poll to either get the new model or are asked to wait.
    """
    global client_responses, current_round, total_rounds
    client_id = request.args.get('client_id')
    client_round = int(request.args.get('round_number', -1))
    
    logger.info(f"Client {client_id} polling (client round: {client_round}, server round: {current_round})")
    
    # Get latest accuracy metrics
    overall_accuracy, per_class_accuracy = evaluate_global_model()
    accuracy_data = {
        'overall_accuracy': float(overall_accuracy),
        'per_class_accuracy': [float(acc) for acc in per_class_accuracy]
    }
    
    # Check if we've completed all rounds
    if current_round >= total_rounds:
        logger.info(f"Client {client_id} polled - Training complete (all rounds finished)")
        return jsonify({
            'status': 'complete',
            'message': 'Training complete - all rounds finished',
            'accuracy_data': accuracy_data
        })
    
    # If client is behind current round, send them the latest model
    if client_round < current_round:
        logger.info(f"Client {client_id} is behind (client: {client_round}, server: {current_round}) - sending latest model")
        return jsonify({
            'status': 'new_model',
            'weights': [w.tolist() for w in global_weights],
            'current_round': current_round,
            'accuracy_data': accuracy_data
        })
    
    # Otherwise, tell client to wait
    logger.info(f"Client {client_id} polled - Waiting for other clients to complete round {current_round}")
    return jsonify({
        'status': 'wait', 
        'message': f'Waiting for all clients to complete round {current_round}'
    })

@app.route('/submit', methods=['POST'])
def submit():
    """
    Clients submit either their trained model or indicate non-participation.
    """
    global client_responses, waiting_clients, num_clients
    data = request.json
    client_id = data['client_id']
    participating = data['participating']

    print("Client ID: ", client_id, "Participating: ", participating, "Round Updates: ", current_round)
    
    if participating:
        # Received trained weights
        client_weights = [np.array(w) for w in data['weights']]
        round_updates.append(client_weights)

        # Store client gradients
        client_gradients[client_id] = [np.array(g) for g in data['gradients']]
    
    # Track that this client has responded
    client_responses.add(client_id)

    # If all clients have responded and round is in progress, perform aggregation
    if len(client_responses) == num_clients:
        # Trigger aggregation
        aggregate_models()
        client_responses.clear()
        waiting_clients.clear()
    
    return jsonify({'status': 'received'})

@app.route('/join', methods=['POST'])
def join():
    """Handle new client joining the federated learning system."""
    global num_clients
    client_id = len(clients) + 1
    clients.add(client_id)
    num_clients = len(clients)
    
    response = {
        'client_id': client_id,
        'weights': [w.tolist() for w in global_model.get_weights()],
        'threshold': PARTICIPATION_THRESHOLD
    }
    
    logger.info(f"New client joined with ID: {client_id}")
    return jsonify(response)

@app.route('/check_participation', methods=['GET'])
def check_participation():
    client_id = request.args.get('client_id')
    if client_id is None:
        return jsonify({'error': 'Client ID is required'}), 400

    # Determine if the client should participate
    participate = should_participate(client_id)

    print("Client ID: ", client_id, "Participate: ", participate, "Selected Clients: ", selected_clients)

    # Respond to the client with the participation decision
    return jsonify({'participate': participate})

if __name__ == '__main__':
    app.run(port=5000)
