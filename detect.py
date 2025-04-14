import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

class CowHealthAnalyzer:
    def __init__(self):
        # Initialize models and device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_detection_model()
        self.init_classification_model()

    def init_detection_model(self):
        """Initialize YOLO cow detection model"""
        try:
            self.model_cow = YOLO('cowModel.pt')
            if not hasattr(self.model_cow, 'predict'):
                raise RuntimeError("Invalid detection model format")
        except Exception as e:
            raise RuntimeError(f"Failed to load detection model: {str(e)}")

    def init_classification_model(self):
        """Initialize MobileNetV2 classification model with error handling"""
        try:
            # Model architecture
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.model.last_channel, 2)
            )
            
            # Load weights with checkpoint validation
            MODEL_PATH = 'training_logs/20250323_133957/best_model_epoch16_f10.926.pth'
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=True)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize classification model: {str(e)}")

    def predict_health(self, image_input):
        """Predict cow health from image region"""
        try:
            if isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
            else:
                image = Image.open(image_input).convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = outputs.max(1)
            return predicted.item() == 0
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return False

    def process_image(self, image_path):
        """Process image file with enhanced validation"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not os.access(image_path, os.R_OK):
                raise PermissionError(f"No read permissions for: {image_path}")
            
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError("Invalid image format or corrupted file")
            
            results = self.model_cow.predict(frame)
            cow_boxes = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []
            
            for box in cow_boxes:
                x1, y1, x2, y2, conf, cls = box.astype(int)
                cow_region = frame[y1:y2, x1:x2]
                cow_region_pil = Image.fromarray(cv2.cvtColor(cow_region, cv2.COLOR_BGR2RGB))
                
                has_lumps = self.predict_health(cow_region_pil)
                color = (0, 0, 255) if has_lumps else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = "Infected" if has_lumps else "Healthy"
                cv2.putText(frame, f'Cow {conf:.2f} - {label}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            output_path = os.path.splitext(image_path)[0] + '_output.jpg'
            cv2.imwrite(output_path, frame)
            print(f"Successfully processed image: {output_path}")
            return True
            
        except Exception as e:
            print(f"Image processing failed: {str(e)}")
            return False

    def process_video(self, video_path):
        """Process video file with comprehensive error handling"""
        cap = None
        out = None
        try:
            # File validation
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not os.access(video_path, os.R_OK):
                raise PermissionError(f"No read permissions for: {video_path}")
            
            # Video capture initialization
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file - possibly corrupted or unsupported format")
            
            # Video metadata validation
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30  # Default FPS
                
            # Codec validation
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            if codec not in ['avc1', 'h264', 'mp4v']:
                print(f"Warning: Unusual codec detected ({codec}), output might be unreliable")
            
            # Output setup
            output_path = os.path.splitext(video_path)[0] + '_output.mp4'
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            
            # Frame processing loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detection and processing
                results = self.model_cow.predict(frame)
                cow_boxes = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []
                
                for box in cow_boxes:
                    x1, y1, x2, y2, conf, cls = box.astype(int)
                    cow_region = frame[y1:y2, x1:x2]
                    cow_region_pil = Image.fromarray(cv2.cvtColor(cow_region, cv2.COLOR_BGR2RGB))
                    
                    has_lumps = self.predict_health(cow_region_pil)
                    color = (0, 0, 255) if has_lumps else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = "Infected" if has_lumps else "Healthy"
                    cv2.putText(frame, f'Cow {conf:.2f} - {label}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                out.write(frame)
            
            print(f"Successfully processed video: {output_path}")
            return True
            
        except Exception as e:
            print(f"Video processing failed: {str(e)}")
            return False
            
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

if __name__ == "__main__":
    try:
        analyzer = CowHealthAnalyzer()
        input_path = 'image.png'  # Replace with your input path
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png']:
            analyzer.process_image(input_path)
        elif file_ext in ['.mp4', '.avi', '.mov']:
            analyzer.process_video(input_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            print("Supported formats: Images (.jpg, .png), Videos (.mp4, .avi, .mov)")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        if 'analyzer' in locals():
            del analyzer
        torch.cuda.empty_cache()