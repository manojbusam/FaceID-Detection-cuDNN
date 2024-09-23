import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
from pycudnn import cudnnCreate, cudnnDestroy, cudnnConvolutionForward, cudnnPoolingForward
from pycudnn import cudnnConvolutionDescriptor, cudnnTensorDescriptor, cudnnPoolingDescriptor, cudnnActivationForward, cudnnActivationDescriptor

# Create cuDNN handle
cudnn_handle = cudnnCreate()

# Initialize cuDNN descriptors for convolution, pooling, and activation
conv_desc = cudnnConvolutionDescriptor()
activation_desc = cudnnActivationDescriptor()
pooling_desc = cudnnPoolingDescriptor()

# Set convolution, activation, and pooling descriptors
conv_desc.set_2d(padding=(1, 1), stride=(1, 1), dilation=(1, 1), mode='CUDNN_CROSS_CORRELATION')
activation_desc.set('CUDNN_ACTIVATION_RELU', nan_opt='CUDNN_PROPAGATE_NAN', relu_ceiling=0.0)
pooling_desc.set_2d(pooling_mode='CUDNN_POOLING_MAX', window_height=2, window_width=2, vertical_stride=2, horizontal_stride=2)

# Load face detection model using OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Placeholder dictionary to store face embeddings
face_id_database = {}

def preprocess_frame(frame):
    """Preprocess the input image and extract the face region."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Extract the first detected face
        face = frame[y:y+h, x:x+w]  # Crop the face region
        resized_face = cv2.resize(face, (32, 32))  # Resize to 32x32
        return resized_face, faces[0]
    return None, None

def extract_face_features(face_tensor):
    """Extract features from the face image using cuDNN convolution and pooling primitives."""
    
    # Allocate memory for the input face tensor
    d_face_tensor = cuda.mem_alloc(face_tensor.nbytes)
    cuda.memcpy_htod(d_face_tensor, face_tensor)  # Copy input data to GPU
    
    # Step 1: Apply convolution using cuDNN
    conv_output = cudnnConvolutionForward(cudnn_handle, face_tensor, conv_desc)
    
    # Step 2: Apply ReLU activation
    relu_output = cudnnActivationForward(cudnn_handle, activation_desc, conv_output)
    
    # Step 3: Apply Max Pooling
    pooled_output = cudnnPoolingForward(cudnn_handle, pooling_desc, relu_output)
    
    # Pooled output can be considered as the embedding for the face
    return pooled_output

def generate_face_id(face_embedding):
    """Generate a unique ID (embedding) from the face features (pooled output)."""
    return np.random.randn(1, 128)  # Simulating a 128-dimensional face embedding

def add_face_to_database(face_embedding, person_name):
    """Store the person's face embedding (ID) in the face database."""
    face_id_database[person_name] = face_embedding

def classify_frame(frame, recognition_phase=False):
    """Extract face, generate ID, and optionally classify (identify) in recognition phase."""
    
    preprocessed_face, face_coords = preprocess_frame(frame)
    
    if preprocessed_face is None:
        return None, None  # No face detected
    
    face_tensor = np.asarray(preprocessed_face, dtype=np.float32)
    face_embedding = extract_face_features(face_tensor)
    
    if recognition_phase:
        # Recognition phase: Match the face embedding against stored IDs
        recognized_person = recognize_face(face_embedding)
        print(f"Recognized: {recognized_person}")
    else:
        # Enrollment phase: Generate and store the face ID for new user
        person_name = input("Enter the person's name: ")
        face_id = generate_face_id(face_embedding)
        add_face_to_database(face_id, person_name)
        print(f"Face ID created for {person_name}.")
    
    return face_coords

def recognize_face(face_embedding):
    """Compare the extracted face embedding with stored IDs and return the recognized person."""
    # In a real application, we would compare the embedding using cosine similarity or Euclidean distance
    for person_name, stored_embedding in face_id_database.items():
        # Simulate embedding comparison (in practice, use actual distance metric)
        similarity = np.random.rand()  # Simulate a similarity score
        if similarity > 0.8:  # If similarity is high enough, we assume it's a match
            return person_name
    return "Unknown"

# Real-time video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video feed.")
    cudnnDestroy(cudnn_handle)
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print("Processing frame...")
    
    # First phase: enroll new faces (i.e., create Face ID)
    classify_frame(frame, recognition_phase=False)

    # Optionally switch to recognition phase later
    # classify_frame(frame, recognition_phase=True)

    # Display frame
    cv2.imshow('Real-Time Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
cudnnDestroy(cudnn_handle)
