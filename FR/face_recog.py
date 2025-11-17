import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class FaceRecognizer:
    def __init__(self):
        self.face_detector = None
        self.face_recognizer = None
        self.label_encoder = LabelEncoder()
        self.svm_classifier = None
        self.embeddings = []
        self.labels = []
        
    def setup_face_detection(self):
        """Setup face detection using OpenCV's DNN module"""
        # Load pre-trained face detection model
        model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = "deploy.prototxt"
        
        # Download files if not exists
        if not os.path.exists(model_file):
            print("Downloading face detection model...")
            # You need to download these files manually or use alternative method
            print("Please download the model files:")
            print("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
            print("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel")
            return False
            
        self.face_detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
        return True
    
    def create_face_recognition_model(self, input_shape=(160, 160, 3)):
        """Create a face recognition model using FaceNet architecture"""
        def conv_block(filters, kernel_size=3, strides=1, padding='same'):
            return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                               kernel_initializer='he_normal')
        
        inputs = keras.Input(shape=input_shape)
        
        # Initial convolution block
        x = conv_block(64)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Second convolution block
        x = conv_block(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Third convolution block
        x = conv_block(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling22)(x)
        
        # Fourth convolution block
        x = conv_block(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Global average pooling and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output embedding
        embeddings = layers.Dense(128, name='embeddings')(x)
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        model = keras.Model(inputs, embeddings)
        return model
    
    def detect_faces(self, image, confidence_threshold=0.7):
        """Detect faces in an image"""
        if self.face_detector is None:
            print("Face detector not initialized!")
            return []
        
        h, w = image.shape[:2]
        
        # Preprocess image for face detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        face_coords = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append(face)
                    face_coords.append((x1, y1, x2, y2))
        
        return faces, face_coords
    
    def preprocess_face(self, face):
        """Preprocess face for recognition model"""
        # Resize to model input size
        face = cv2.resize(face, (160, 160))
        # Convert to float32 and normalize
        face = face.astype(np.float32) / 255.0
        # Expand dimensions for batch
        face = np.expand_dims(face, axis=0)
        return face
    
    def extract_embeddings(self, faces):
        """Extract face embeddings"""
        if self.face_recognizer is None:
            print("Face recognition model not trained!")
            return []
        
        embeddings = []
        for face in faces:
            processed_face = self.preprocess_face(face)
            embedding = self.face_recognizer.predict(processed_face, verbose=0)
            embeddings.append(embedding[0])
        
        return embeddings
    
    def train(self, dataset_path):
        """Train the face recognition system"""
        print("Setting up face detection...")
        if not self.setup_face_detection():
            print("Using Haar cascade as fallback...")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        print("Creating face recognition model...")
        self.face_recognizer = self.create_face_recognition_model()
        
        # Compile model
        self.face_recognizer.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError()
        )
        
        print("Loading training data...")
        self._load_training_data(dataset_path)
        
        print("Training SVM classifier...")
        self.svm_classifier = SVC(kernel='linear', probability=True)
        self.svm_classifier.fit(self.embeddings, self.labels)
        
        print("Training completed cyka blyat!")
    
    def _load_training_data(self, dataset_path):
        """Load training data from directory structure"""
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist!")
            return
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_path):
                print(f"Processing {person_name}...")
                
                for image_file in os.listdir(person_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, image_file)
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            faces, _ = self.detect_faces(image)
                            
                            for face in faces:
                                embedding = self.extract_embeddings([face])
                                if embedding:
                                    self.embeddings.append(embedding[0])
                                    self.labels.append(person_name)
        
        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            self.labels = self.label_encoder.fit_transform(self.labels)
    
    def recognize_faces(self, image, confidence_threshold=0.6):
        """Recognize faces in an image"""
        faces, face_coords = self.detect_faces(image)
        
        if not faces:
            return [], []
        
        embeddings = self.extract_embeddings(faces)
        recognitions = []
        
        for i, embedding in enumerate(embeddings):
            # Predict using SVM
            probabilities = self.svm_classifier.predict_proba([embedding])[0]
            max_prob = np.max(probabilities)
            predicted_label = self.svm_classifier.predict([embedding])[0]
            
            if max_prob > confidence_threshold:
                person_name = self.label_encoder.inverse_transform([predicted_label])[0]
                recognitions.append({
                    'name': person_name,
                    'confidence': max_prob,
                    'coordinates': face_coords[i]
                })
            else:
                recognitions.append({
                    'name': 'Unknown',
                    'confidence': max_prob,
                    'coordinates': face_coords[i]
                })
        
        return recognitions, faces
    
    def save_model(self, model_path):
        """Save trained model"""
        if self.svm_classifier is not None:
            with open(f"{model_path}_svm.pkl", 'wb') as f:
                pickle.dump(self.svm_classifier, f)
            
            with open(f"{model_path}_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            self.face_recognizer.save(f"{model_path}_recognizer.h5")
            print("Model saved successfully!")
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            with open(f"{model_path}_svm.pkl", 'rb') as f:
                self.svm_classifier = pickle.load(f)
            
            with open(f"{model_path}_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.face_recognizer = keras.models.load_model(f"{model_path}_recognizer.h5")
            self.setup_face_detection()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def create_sample_dataset():
    """Create a sample dataset directory structure"""
    os.makedirs("dataset/person1", exist_ok=True)
    os.makedirs("dataset/person2", exist_ok=True)
    print("Sample dataset directory created!")
    print("Add face images to dataset/person1/ and dataset/person2/ folders")

def real_time_recognition(model_path="face_model"):
    """Real-time face recognition using webcam"""
    recognizer = FaceRecognizer()
    
    if not recognizer.load_model(model_path):
        print("Please train the model first!")
        return
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Recognize faces
        recognitions, faces = recognizer.recognize_faces(frame)
        
        # Draw results on frame
        for recognition in recognitions:
            x1, y1, x2, y2 = recognition['coordinates']
            name = recognition['name']
            confidence = recognition['confidence']
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display name and confidence
            label = f"{name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition - Press Q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    
    # Create sample dataset structure
    create_sample_dataset()
    
    print("Face Recognition System")
    print("1. Train model")
    print("2. Real-time recognition")
    print("3. Test on image")
    
    choice = input("Choose option (1-3): ")
    
    if choice == "1":
        # Train the model
        dataset_path = "dataset"  # Change this to your dataset path
        recognizer.train(dataset_path)
        recognizer.save_model("face_model")
        
    elif choice == "2":
        # Real-time recognition
        real_time_recognition()
        
    elif choice == "3":
        # Test on single image
        image_path = input("Enter image path: ")
        image = cv2.imread(image_path)
        
        if image is not None:
            recognizer.load_model("face_model")
            recognitions, faces = recognizer.recognize_faces(image)
            
            # Display results
            for i, recognition in enumerate(recognitions):
                print(f"Face {i+1}: {recognition['name']} (Confidence: {recognition['confidence']:.2f})")
            
            # Draw results on image
            for recognition in recognitions:
                x1, y1, x2, y2 = recognition['coordinates']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, recognition['name'], (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Could not load image!")
    
    else:
        print("Invalid choice!")