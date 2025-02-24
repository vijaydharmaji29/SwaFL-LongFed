import numpy as np
import requests
import tensorflow as tf
import logging
import time
import psutil
import random
import os
from dotenv import load_dotenv
from monitoring import ClientMonitor
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[CLIENT] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.current_round = -1  # Initialize round number
        self.latest_accuracy_data = None  # Store latest accuracy metrics
        
        # Load configuration from environment variables
        self.alpha = float(os.getenv('DIRICHLET_ALPHA', 0.5))
        logger.info(f"Loaded Dirichlet alpha: {self.alpha}")
        
        # Join the federated learning system
        response = requests.post(f'{server_url}/join')
        logger.info(f"Joining federated learning system - POST {server_url}/join")
        
        if response.status_code == 200:
            data = response.json()
            self.client_id = data['client_id']
            self.threshold = data['threshold']  # Store threshold from server
            logger.info(f"Received participation threshold: {self.threshold}")
            
            self.model = self.create_model()
            self.model.set_weights([np.array(w) for w in data['weights']])
            self.current_round = 0  # Set initial round
            logger.info(f"Initialized client_id: {self.client_id}")
            
            # Initialize monitor
            self.monitor = ClientMonitor(self.client_id)
        else:
            logger.error(f"Failed to join. Status code: {response.status_code}")
            raise Exception("Failed to join federated learning system")

    def create_model(self):
        # Pin TensorFlow to a single CPU core
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(physical_devices[0], 'CPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def train_local_model(self, x_train, y_train, round_number, epochs=1):
        logger.info(f"Training local model for client {self.client_id}")
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = time.time()
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )
        
        # Calculate metrics
        training_time = time.time() - start_time
        accuracy = history.history['val_accuracy'][-1]
        
        # Stop monitoring and get averages
        avg_cpu, avg_memory = self.monitor.stop_monitoring()
        
        # Log summary
        logger.info(f"Round {round_number} - Training completed:")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Average CPU usage: {avg_cpu:.2f}%")
        logger.info(f"Average Memory usage: {avg_memory:.2f}%")
        
        return history, training_time, accuracy, avg_cpu, avg_memory

    def get_random_data_sample(self, x_data, y_data):
        """
        Get a random 50% sample of the data using Dirichlet distribution for labels.
        Classes are randomly ordered when applying the distribution.
        Alpha parameter is loaded from config.json.
        """
        total_samples = len(x_data)
        sample_size = total_samples // 2  # 50% of data
        num_classes = 10  # CIFAR-10 has 10 classes
        self.num_classes = num_classes
        
        # Get indices for each class
        class_indices = [np.where(y_data == i)[0] for i in range(num_classes)]
        class_sizes = [len(indices) for indices in class_indices]
        
        # Generate Dirichlet distribution for this client using alpha from config
        dirichlet_dist = np.random.dirichlet([self.alpha] * num_classes)
        
        # Randomly permute the class order
        class_order = np.random.permutation(num_classes)
        permuted_dirichlet = dirichlet_dist[class_order]
        
        # Calculate number of samples per class based on permuted Dirichlet distribution
        samples_per_class = (permuted_dirichlet * sample_size).astype(int)
        
        # Adjust samples_per_class to not exceed available samples in each class
        for i, class_idx in enumerate(class_order):
            if samples_per_class[i] > class_sizes[class_idx]:
                # If we need more samples than available, take all available
                samples_per_class[i] = class_sizes[class_idx]
        
        # Redistribute the remaining samples
        remaining = sample_size - sum(samples_per_class)
        if remaining > 0:
            # Find classes that can accept more samples
            available_classes = [
                i for i, class_idx in enumerate(class_order)
                if samples_per_class[i] < class_sizes[class_idx]
            ]
            
            while remaining > 0 and available_classes:
                # Randomly select from available classes
                idx = np.random.choice(available_classes)
                class_idx = class_order[idx]
                
                # Add one sample if possible
                if samples_per_class[idx] < class_sizes[class_idx]:
                    samples_per_class[idx] += 1
                    remaining -= 1
                else:
                    available_classes.remove(idx)
        
        # Sample indices from each class using the permuted order
        selected_indices = []
        for i, class_idx in enumerate(class_order):
            if samples_per_class[i] > 0:
                selected_indices.extend(
                    np.random.choice(
                        class_indices[class_idx],
                        size=samples_per_class[i],
                        replace=False
                    )
                )
        
        # Shuffle the selected indices
        np.random.shuffle(selected_indices)
        
        # Log distribution information with original class order
        # actual_distribution = [
        #     len(np.where(y_data[selected_indices] == i)[0]) / len(selected_indices)
        #     for i in range(num_classes)
        # ]
        logger.info(f"Client {self.client_id} Round {self.current_round} class distribution:")
        logger.info("Original Dirichlet distribution: " + 
                    " ".join([f"{x:.3f}" for x in dirichlet_dist]))
        logger.info("Class order for this round: " + 
                    " ".join([str(x) for x in class_order]))
        # logger.info("Actual class distribution:")
        # for class_idx, percentage in enumerate(actual_distribution):
        #     logger.info(f"Class {class_idx}: {percentage:.3f}")
        
        return x_data[selected_indices], y_data[selected_indices]

    def check_participation(self):
        """Check if the client should participate in the current round."""
        response = requests.get(f"{self.server_url}/check_participation", params={'client_id': self.client_id})
        if response.status_code == 200:
            data = response.json()
            return data.get('participate', False)
        else:
            logger.error(f"Error checking participation: {response.text}")
            return False

    def compute_gradients(self, x_sample, y_sample):
        """Compute gradients of the model with respect to the sample data."""
        with tf.GradientTape() as tape:
            predictions = self.model(x_sample, training=True)
            loss = self.model.compiled_loss(y_sample, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return [g.numpy() for g in gradients]

    def participate_in_round(self, x_train=None, y_train=None, round_number=0):
        # Start monitoring for this round regardless of participation
        self.monitor.start_monitoring()
        start_time = time.time()

        # Store current training data for label distribution analysis
        self.current_training_data = (x_train, y_train)

        # Get random sample for this round
        x_sample, y_sample = self.get_random_data_sample(x_train, y_train)
        self.current_training_sample = (x_sample, y_sample)
        logger.info(f"Client {self.client_id} sampled {len(x_sample)} data points for round {round_number}")
        
        # Check participation
        if round_number == 1:
            should_participate = True
            print("Round 1 Participation")
        else:
            should_participate = self.check_participation()
            print(f"Client {self.client_id} should participate: {should_participate}")
        
        if not should_participate:
            logger.info(f"Client {self.client_id} chose not to participate in round {round_number}")
            
            # Stop monitoring and get averages even when not participating
            training_time = time.time() - start_time
            avg_cpu, avg_memory = self.monitor.stop_monitoring()
            
            # Log metrics for non-participation
            self.monitor.log_metrics(
                round_number=round_number,
                training_time=training_time,
                accuracy=0,
                cpu_percent=avg_cpu,
                memory_percent=avg_memory,
                participated=False
            )
            
            response = requests.post(
                f'{self.server_url}/submit',
                json={
                    'client_id': self.client_id,
                    'participating': False
                }
            )
            return False

        # Train local model with monitoring using sampled data
        history, training_time, accuracy, avg_cpu, avg_memory = self.train_local_model(x_sample, y_sample, round_number)

        # Compute gradients
        gradients = self.compute_gradients(x_sample, y_sample)

        # Log metrics with all information
        self.monitor.log_metrics(
            round_number=round_number,
            training_time=training_time,
            accuracy=accuracy,
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            participated=True
        )
        
        # Submit updated model and gradients
        logger.info(f"Client {self.client_id} submitting updated model for round {round_number}")
        response = requests.post(
            f'{self.server_url}/submit',
            json={
                'client_id': self.client_id,
                'participating': True,
                'weights': [w.tolist() for w in self.model.get_weights()],
                'gradients': [g.tolist() for g in gradients]
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to submit model. Status: {response.status_code}")
            return False

        # Poll until we get a new model for next round
        logger.info(f"Client {self.client_id} starting to poll for aggregated model after round {self.current_round}")
        poll_count = 0
        while True:
            poll_count += 1
            logger.info(f"Client {self.client_id} poll attempt {poll_count} for round {self.current_round}")
            
            response = requests.get(
                f'{self.server_url}/poll',
                params={
                    'client_id': self.client_id,
                    'round_number': self.current_round
                }
            )

            print("Polling with: ", self.current_round)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'complete':
                    self.latest_accuracy_data = data.get('accuracy_data')
                    logger.info(f"Client {self.client_id} - Training complete after {self.current_round} rounds")
                    return False
                elif data['status'] == 'new_model':
                    self.model.set_weights([np.array(w) for w in data['weights']])
                    self.current_round = data['current_round']
                    self.latest_accuracy_data = data.get('accuracy_data')
                    logger.info(f"Client {self.client_id} received new aggregated model for round {self.current_round} after {poll_count} polls")
                    break
                elif data['status'] == 'wait':
                    logger.info(f"Client {self.client_id} waiting for aggregation... (poll attempt {poll_count})")
                    time.sleep(1)
            else:
                logger.error(f"Failed to poll server. Status: {response.status_code}")
                return False
                
        return True

    

