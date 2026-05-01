import torch
import numpy as np
import cv2
import torchvision.models as models
import torchvision.transforms as transforms

class EventDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = device

        # ===== Load LSTM model =====
        from model import Model  # (you must save this class separately)
        self.model = Model().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # ===== ResNet feature extractor =====
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = torch.nn.Identity()
        self.resnet = self.resnet.to(device)
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        # ===== Config =====
        self.buffer = []
        self.buffer_size = 200   # sequence length
        self.frame_skip = 2
        self.frame_count = 0

        self.thresholds = [0.25, 0.30, 0.55]
        self.classes = ["goal", "card", "foul"]

        self.cooldown = 20
        self.last_event_frame = [-100, -100, -100]

    def extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.resnet(img).cpu().numpy()[0]

        return feat

    def predict(self, frame, frame_num):
        self.frame_count += 1

        # skip frames (same as training)
        if self.frame_count % self.frame_skip != 0:
            return []

        feat = self.extract_feature(frame)
        self.buffer.append(feat)

        # wait until enough sequence
        if len(self.buffer) < self.buffer_size:
            return []

        # keep sliding window
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        buffer_np = np.array(self.buffer, dtype=np.float32)
        x = torch.from_numpy(buffer_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)
            probs = torch.sigmoid(out).cpu().numpy()[0]

        events = []

        t = len(probs) - 1  # latest timestep

        for c in range(len(self.classes)):

            if probs[t, c] > self.thresholds[c]:

                # cooldown check
                if frame_num - self.last_event_frame[c] < self.cooldown:
                    continue

                # peak check
                left = max(0, t-5)
                right = min(len(probs), t+5)

                if probs[t, c] != probs[left:right, c].max():
                    continue

                events.append({
                    "frame": frame_num,
                    "event": self.classes[c],
                    "confidence": float(probs[t, c])
                })

                self.last_event_frame[c] = frame_num

        return events