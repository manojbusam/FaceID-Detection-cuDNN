# Face ID Detection Using cuDNN

This repository implements a **real-time Face ID detection system** using **NVIDIA's cuDNN** primitives for fast and efficient neural network operations, including convolution, ReLU, and pooling. The system detects a person’s face, extracts unique features, creates a face embedding (ID), and uses it for subsequent face recognition.

## Features:
- **Real-Time Face Detection**: Captures video feed and detects faces using OpenCV.
- **Face Feature Extraction**: Uses cuDNN convolution, ReLU activation, and max pooling to extract key features from the face.
- **Face ID Creation**: Generates a unique embedding (vector) for each detected face.
- **Face Recognition**: Compares real-time face embeddings with stored embeddings to recognize individuals.
- **Highly Optimized**: Leveraging cuDNN for GPU-accelerated computation.

## Prerequisites

1. **Python 3.x**
2. **NVIDIA GPU** with CUDA support.
3. **cuDNN** library installed.
4. **OpenCV**: For real-time video capture and face detection.
5. **PyCUDA**: To interface with CUDA for GPU computations.
6. **cuDNN Python bindings**: Install using `pip` or compile the bindings manually.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/face-id-cudnn.git
   cd face-id-cudnn
   ```

2. Install dependencies:

   ```bash
   pip install numpy opencv-python pycuda pycudnn
   ```

3. Ensure that CUDA and cuDNN are properly installed on your system.

4. Verify your GPU setup:

   ```bash
   nvidia-smi
   ```

## How It Works

### 1. **Face Detection**:
   - Uses OpenCV’s Haar Cascade classifier to detect faces in a video frame.
   - The face is cropped and resized to 32x32 pixels for feature extraction.

### 2. **Face Feature Extraction**:
   - The face image is passed through cuDNN primitives:
     - **Convolution**: Extracts feature maps from the input image using a set of filters.
     - **ReLU Activation**: Introduces non-linearity to the feature maps.
     - **Max Pooling**: Reduces the spatial size of the feature maps, keeping essential information.

### 3. **Face ID Creation**:
   - The output of the pooling layer is transformed into a lower-dimensional **face embedding** (128 dimensions), representing a unique identifier for the person.
   - This face ID is stored in a dictionary for later recognition.

### 4. **Face Recognition**:
   - In the recognition phase, the face embedding of the current frame is compared with the stored embeddings.
   - A similarity metric is applied to match the current face to a stored face ID.

## Running the Program

1. Run the script to capture the video feed and start face detection and ID creation:

   ```bash
   python face_id_detection.py
   ```

2. In the **enrollment phase**, the system will prompt for the person's name after detecting a face and will store the generated Face ID.

3. In the **recognition phase**, switch the program to compare live faces with stored face IDs to recognize known individuals.

4. Press `q` to exit the video feed.

## Example Output

- In the terminal, you will see outputs such as:
  - `Processing frame...`
  - `Face ID created for <person_name>`
  - `Recognized: <person_name>` when recognizing faces from the database.

## Real-World Use Case

This Face ID detection system can be integrated into real-world applications, such as:
- **Access Control Systems**: Unlock doors based on facial recognition.
- **Device Authentication**: Use face recognition to unlock mobile devices or computers.
- **Attendance Systems**: Automatically recognize and log attendees in meetings or classrooms.

---

## Acknowledgements

- **cuDNN**: Accelerating deep learning applications with highly optimized GPU routines.
- **OpenCV**: Simplifying computer vision tasks like face detection.

