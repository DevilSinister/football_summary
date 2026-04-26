import torch
import torch.nn as nn
import numpy as np
import cv2


class EventModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_classes=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)

        # take last timestep
        out = out[:, -1, :]

        out = self.fc(out)
        return out


class EventDetector:
    def __init__(self, model_path):
        self.device = 'cpu'

        # ⚠️ IMPORTANT: these must match training
        self.input_size = 512      # CHANGE if needed
        self.seq_length = 10       # number of frames per clip

        self.model = EventModel(input_size=self.input_size)

        # ✅ FIX WARNING + SAFE LOAD
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.classes = ['goal', 'foul', 'none']

        # buffer for sequence
        self.frame_buffer = []

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0

        # flatten → feature vector
        frame = frame.flatten()

        # reduce dimension (simple projection)
        if len(frame) > self.input_size:
            frame = frame[:self.input_size]
        else:
            frame = np.pad(frame, (0, self.input_size - len(frame)))

        return frame

    def predict(self, frame, frame_num):
        try:
            feature = self.preprocess_frame(frame)

            # add to buffer
            self.frame_buffer.append(feature)

            # keep fixed sequence length
            if len(self.frame_buffer) < self.seq_length:
                return None

            if len(self.frame_buffer) > self.seq_length:
                self.frame_buffer.pop(0)

            sequence = np.array(self.frame_buffer)

            sequence = torch.tensor(sequence, dtype=torch.float32)
            sequence = sequence.unsqueeze(0)  # (1, seq_len, features)

            with torch.no_grad():
                output = self.model(sequence)

            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

            pred = pred.item()
            conf = conf.item()

            event_name = self.classes[pred]

            if event_name != "none" and conf > 0.7:
                return {
                    "frame": frame_num,
                    "event": event_name,
                    "confidence": conf
                }

        except Exception as e:
            print("EventDetector error:", e)

        return None