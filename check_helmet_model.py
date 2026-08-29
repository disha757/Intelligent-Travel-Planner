from ultralytics import YOLO
model = YOLO("models/helmet_model.pt")
print("Helmet model loaded successfully.")
print("Classes:", model.names)
