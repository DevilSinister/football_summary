import cv2
import numpy as np
import torch
import torch.nn as nn


class EventModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        output, _ = self.lstm(inputs)
        return self.fc(output[:, -1, :])


class EventDetector:
    """Sliding-window goal/foul detector used by the canonical pipeline."""

    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.input_size = 512
        self.seq_length = 10
        self.model = EventModel(input_size=self.input_size).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.classes = ["goal", "foul", "none"]
        self.frame_buffer = []

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (64, 64)).astype(np.float32) / 255.0
        features = resized.flatten()
        if len(features) > self.input_size:
            return features[: self.input_size]
        return np.pad(features, (0, self.input_size - len(features)))

    def predict(self, frame, frame_num):
        features = self.preprocess_frame(frame)
        self.frame_buffer.append(features)
        if len(self.frame_buffer) < self.seq_length:
            return None
        if len(self.frame_buffer) > self.seq_length:
            self.frame_buffer.pop(0)

        sequence = torch.tensor(np.asarray(self.frame_buffer), dtype=torch.float32)
        sequence = sequence.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(sequence)
            probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        event_name = self.classes[prediction.item()]
        if event_name == "none":
            return None
        return {
            "frame": frame_num,
            "event": event_name,
            "confidence": float(confidence.item()),
        }
