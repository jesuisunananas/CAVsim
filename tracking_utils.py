import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn.functional as F
import torch.nn as nn
import cv2

class AppearanceExtractor:
    """Extracts visual appearance embeddings using MobileNetV3."""
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading Appearance Extractor on {self.device}...")
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Identity()
        self.model = model.to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, frame, bbox):
        x1, y1, x2, y2 = map(int, [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
            
        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        embedding = self.model(input_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0]


class KalmanTracker:
    """
    Simple 4D Kalman filter tracking Latitude, Longitude and their velocities.
    State: [lat, lon, v_lat, v_lon]
    """
    def __init__(self, initial_lat, initial_lon):
        self.x = np.array([initial_lat, initial_lon, 0.0, 0.0])
        
        # State transition matrix F
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Covariance matrices adjusted for GPS scale (degrees are tiny)
        self.P = np.eye(4) * 1e-6
        self.Q = np.eye(4) * 1e-8
        self.R = np.eye(2) * 1e-7
        
    def get_prediction(self, dt=1.0):
        F = self.F.copy()
        F[0, 2] = dt
        F[1, 3] = dt
        pred_x = np.dot(F, self.x)
        return pred_x[:2]

    def predict(self, dt=1.0):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2]
        
    def update(self, measurement):
        z = np.array(measurement)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(4)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x[:2]
