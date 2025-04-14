import cv2
from ultralytics import YOLO
from efficientnet_predict import predict
from PIL import Image

# Load YOLO models
model_cow = YOLO('cowModel.pt')

# Open video file
video_path = 'video.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    # Detect cows
    results_cow = model_cow.predict(frame)
    
    if len(results_cow) > 0 and hasattr(results_cow[0], "boxes"):
        cow_boxes = results_cow[0].boxes.data.cpu().numpy()  # Convert to NumPy array
    else:
        cow_boxes = []

    for box in cow_boxes:
        x1, y1, x2, y2, conf, cls = box.astype(int)

        # Crop cow region
        cow_region = frame[y1:y2, x1:x2]

        # Convert OpenCV image (NumPy array) to PIL Image
        cow_region_pil = Image.fromarray(cv2.cvtColor(cow_region, cv2.COLOR_BGR2RGB))

        # Detect lumps using MobileNet
        has_lumps = predict(cow_region_pil)

        # Draw bounding box
        color = (0, 0, 255) if has_lumps else (0, 255, 0)  # Red for infected, Green otherwise
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "Infected" if has_lumps else "Healthy"
        cv2.putText(frame, f'Cow {conf:.2f} - {label}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show frame (optional)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
