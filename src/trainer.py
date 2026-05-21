import torch
import threading
import time
from copy import deepcopy
from model import get_device

class AsyncTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.stop_event = threading.Event()
        self.is_training = False
        self.progress = 0.0
        self.status = "Idle"
        self.backup_state = None

    def train_step(self, user_features, item_features, feedback_list, epochs=5):
        self.is_training = True
        self.stop_event.clear()
        self.status = "Training started..."
        self.progress = 0.0
        
        # Atomic Backup
        self.backup_state = deepcopy(self.model.state_dict())
        
        device = get_device()
        self.model.to(device)
        
        # Map feedback to target values
        target_map = {"like": 1.0, "dislike": 0.1, "watched": 0.5}
        targets = [target_map.get(f, 0.5) for f in feedback_list]
        
        u_feat = torch.FloatTensor(user_features).to(device)
        i_feat = torch.FloatTensor(item_features).to(device)
        y = torch.FloatTensor(targets).to(device)
        
        try:
            for epoch in range(epochs):
                if self.stop_event.is_set():
                    self.status = "Training cancelled. Rolling back..."
                    self.model.load_state_dict(self.backup_state)
                    self.is_training = False
                    return False
                
                self.model.train()
                self.optimizer.zero_grad()
                
                outputs = self.model(u_feat, i_feat)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                self.progress = (epoch + 1) / epochs
                self.status = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}"
                time.sleep(0.5) # Simulate workload for progress visibility
                
            self.status = "Training complete."
            self.is_training = False
            return True
            
        except Exception as e:
            self.status = f"Error: {str(e)}. Rolling back..."
            self.model.load_state_dict(self.backup_state)
            self.is_training = False
            return False

    def start_training(self, user_features, item_features, ratings, epochs=5):
        thread = threading.Thread(target=self.train_step, args=(user_features, item_features, ratings, epochs))
        thread.daemon = True # Ensure thread dies when main process stops
        thread.start()
        return thread

    def cancel(self):
        self.stop_event.set()
