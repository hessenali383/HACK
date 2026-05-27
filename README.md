# FaceID — Real-Time Face Recognition System

A lightweight, modern, and high-performance web-based face recognition application. This system captures video from the client's webcam, processes frames via a Flask backend using deep learning models to detect and recognize faces in real-time, and manages a local database of known individuals.

---

## 🚀 Key Features

* **Real-Time Multi-Face Tracking:** Detects, crops, and processes multiple faces simultaneously directly from the live camera stream.
* **AI-Powered Accuracy:** Powered by **MTCNN** for robust face detection/bounding box localization, and **FaceNet (InceptionResnetV1)** pretrained on `vggface2` for generating high-quality face embeddings.
* **On-the-Fly Registration:** An intuitive pop-up interface automatically detects unknown faces, allowing you to register and name them instantly without restarting the server.
* **Persistent Local Storage:** Utilizes **SQLite** to securely store face embeddings and names, ensuring quick lookups and easy management.
* **Cyberpunk UI Dashboard:** Features a responsive frontend with interactive controls to adjust frame verification intervals, change cosine similarity confidence thresholds, and view live performance metrics like FPS and Latency.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask, Flask-CORS
* **AI/ML:** PyTorch, Facenet-PyTorch, Scikit-Learn (Cosine Similarity), NumPy
* **Database:** SQLite3
* **Frontend:** HTML5 (WebRTC MediaDevices & Canvas API), CSS3 (Modern Grid/Flexbox), JavaScript (Vanilla ES6)

---

## ⚙️ Installation & Setup

### 1. Clone or Create the Files
Ensure your project directory structure looks exactly like this:
```text
├── app.py
├── database/        # Created automatically upon startup
└── templates/
    └── index.html
