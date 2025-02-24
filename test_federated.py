import numpy as np
from client import FederatedClient
import tensorflow as tf
import threading
import logging
import os
from dotenv import load_dotenv
from utils.email_sender import send_logs_email

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClientThread(threading.Thread):
    def __init__(self, client, client_x, client_y, num_rounds):
        threading.Thread.__init__(self)
        self.client = client
        self.client_x = client_x
        self.client_y = client_y
        self.num_rounds = num_rounds
    
    def run(self):
        for round_num in range(self.num_rounds):
            logger.info(f"Client {self.client.client_id} starting round {round_num + 1}")
            self.client.participate_in_round(self.client_x, self.client_y, round_num + 1)
            logger.info(f"Client {self.client.client_id} completed round {round_num + 1}")

def main():
    # Load config values from environment variables
    num_rounds = int(os.getenv('NUM_ROUNDS', 5))
    num_clients = int(os.getenv('NUM_CLIENTS', 3))
    server_url = os.getenv('SERVER_URL', 'http://localhost:5000')

    # Load and preprocess CIFAR-10 data
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0

    # Create multiple clients
    client_threads = []
    data_per_client = len(x_train) // num_clients

    # Create and initialize clients
    for i in range(num_clients):
        client = FederatedClient(server_url)
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        
        # Assign data to client
        client_x = x_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        
        # Create thread for this client
        thread = ClientThread(client, client_x, client_y, num_rounds)
        client_threads.append(thread)

    # Start all client threads
    logger.info(f"\nStarting federated training with {num_clients} clients for {num_rounds} rounds")
    for thread in client_threads:
        thread.start()

    # Wait for all clients to complete
    for thread in client_threads:
        thread.join()

    logger.info("All clients have completed training")

    # Send logs via email
    logger.info("Attempting to send training logs via email...")
    if send_logs_email():
        logger.info("Training logs sent successfully via email")
    else:
        logger.error("Failed to send training logs via email")

if __name__ == "__main__":
    main() 