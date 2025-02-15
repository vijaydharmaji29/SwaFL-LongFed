import time
import psutil
import csv
from datetime import datetime
import os
import threading
import numpy as np

class ClientMonitor:
    def __init__(self, client_id):
        self.client_id = client_id
        self.monitoring = False
        self.cpu_percentages = []
        self.memory_percentages = []
        
        # Create logs directory if it doesn't exist
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set CSV file path inside logs directory
        self.csv_filename = os.path.join(self.log_dir, f'client_{client_id}_metrics.csv')
        
        # Create new CSV file with headers (overwrite if exists)
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'round_number',
                'participated',
                'training_time_seconds',
                'accuracy',
                'cpu_core_0_percent',
                'memory_percent',
                'data_score',
                'system_score',
                'overall_score',
                'adjusted_threshold'
            ])
    
    def _monitor_resources(self):
        process = psutil.Process()
        while self.monitoring:
            self.cpu_percentages.append(process.cpu_percent())
            self.memory_percentages.append(process.memory_percent())
            time.sleep(0.2)  # Collection frequency: 0.2 seconds
    
    def start_monitoring(self):
        self.monitoring = True
        self.cpu_percentages = []
        self.memory_percentages = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        self.monitor_thread.join()
        
        # Calculate averages
        avg_cpu = np.mean(self.cpu_percentages) if self.cpu_percentages else 0
        avg_memory = np.mean(self.memory_percentages) if self.memory_percentages else 0
        return avg_cpu, avg_memory
    
    def log_metrics(self, round_number, training_time, accuracy, 
                   cpu_percent, memory_percent, participated=True,
                   data_score=0.0, system_score=0.0, 
                   overall_score=0.0, adjusted_threshold=0.0):
        """Log metrics to CSV file with participation status and scores."""
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round_number,
                str(participated),
                f"{training_time:.2f}",
                f"{accuracy:.4f}",
                f"{cpu_percent:.2f}",
                f"{memory_percent:.2f}",
                f"{data_score:.4f}",
                f"{system_score:.4f}",
                f"{overall_score:.4f}",
                f"{adjusted_threshold:.4f}"
            ]) 