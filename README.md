(note the file has a gitignore as we have api key for email notification that sends absent and late users email as shown  <img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/6d437adf-3d35-437f-a2bc-bd13bc749515" />

# Smart Attendance System — Complete Documentation

A real-time face recognition attendance system that identifies people via webcam and logs check-ins at designated times throughout the day.

---

## Table of Contents
- [System Overview](#system-overview)
- [Architecture](#architecture)
- [File-by-File Explanation](#file-by-file-explanation)
  - [db.py — Database Layer](#dbpy--database-layer)
  - [attendance_logic.py — Business Rules](#attendance_logicpy--business-rules)
  - [recognition.py — Face Recognition Engine](#recognitionpy--face-recognition-engine)
  - [seed_users.py — Initial Enrollment](#seed_userspy--initial-enrollment)
  - [enroll_webcam.py — Live Enrollment](#enroll_webcampy--live-enrollment)
  - [live_engine.py — Real-Time Engine](#live_enginepy--real-time-engine)
- [Key Thresholds](#key-thresholds)
- [Configuration Variables](#configuration-variables)
  - [Speed vs Accuracy](#speed-vs-accuracy)
  - [Model Selection](#model-selection)
- [Flask Dashboard](#flask-dashboard)
- [Main Entry Point (CLI)](#main-entry-point-cli)
- [Installation & Setup](#installation--setup)
- [Security & Privacy](#security--privacy)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## System Overview
This system detects faces, creates embeddings, compares them to enrolled users, and logs attendance when a match is confirmed in an active time window (“gate”).

---

## Architecture



flowchart LR
    CAM[Camera Input] --> DET{Face Detect\n(MediaPipe or Haar)}
    DET --> CROP[Crop Face\n+ Preprocess]
    CROP --> EMB[DeepFace Embedding\n(128/512-dim)]
    EMB --> CMP[Compare with DB Embeddings\n(cosine distance)]
    CMP --> VOTE[Voting System\n(require 2/2)]
    VOTE --> LOG[Log to DB\n(5s cooldown)]


db.py — Database Layer

SQLAlchemy ORM models and session management.

User

user_id (PK) | name (unique) | face_embedding (JSON string of floats)
email (optional) | external_id (optional)


Attendance (composite key: user_id + date)

user_id (PK/FK) | date (PK, YYYY-MM-DD) | quarter_id
checkin1_time | checkin2_time | checkin3_time | checkin4_time

attendance_logic.py — Business Rules

Defines four daily windows and helpers.

Windows (default):

Morning Start: 07:00–09:00

Lunch Out: 12:00–13:30

Lunch In: 13:00–14:30

Evening End: 17:00–19:30

Functions:

gate_for_now() → returns which gate (1–4) is open

write_gate_timestamp() → writes time to correct column

compute_lateness_minutes() → lateness calculation

recognition.py — Face Recognition Engine

Core ML utilities.

Loads DeepFace model (Facenet / Facenet512)

Converts face → embedding vector

Compares via cosine distance

Key methods:

embed_face(image)        # -> np.ndarray vector
identify(face, known)    # -> matches with distance/confidence
cosine_distances(A, B)   # -> 1 - dot(A, B)

seed_users.py — Initial Enrollment

Batch-enroll from seed_images/:

Detect face(s) → compute embedding → store name + vector in DB.

Usage:

python seed_users.py

enroll_webcam.py — Live Enrollment

Interactive capture via webcam:

Press SPACE to take multiple samples

Average embeddings for robustness

Save to DB

Usage:

python enroll_webcam.py --name "John Doe" --samples 5

live_engine.py — Real-Time Engine
flowchart TD
    A[Capture Frame] --> B{Face Detection\n(every Nth frame)}
    B -->|MediaPipe ok| C[Largest Face]
    B -->|Fallback Haar| C
    C --> D[Validate size ≥ 80x80]
    D --> E[Crop + 30% margin]
    E --> F[Preprocess\nBGR→RGB, CLAHE, normalize]
    F --> G[Embedding\n(Facenet/512)]
    G --> H[Compare vs Known\n(cosine)]
    H --> I{Thresholds\n(dist ≤ 0.50\nconf ≥ 0.70\nquality ≥ 25)}
    I -->|pass| J[Voting 2/2]
    I -->|fail| K[Analyzing / Unknown]
    J --> L{Cooldown ≥ 5s?}
    L -->|yes| M[Log to DB\n(gate 1–4)]
    L -->|no| N[Wait & show cooldown]


Key Thresholds

DIST_THRESHOLD = 0.50 — cosine distance (lower = stricter)

MIN_CONFIDENCE = 0.70 — derived from distance (≈ 1 - dist/2)

MIN_FACE_QUALITY = 25.0 — blur/brightness quality gate

Voting: VOTE_WINDOW = 2, VOTE_MIN_SAME = 2, COOLDOWN_SEC = 5

Configuration Variables
Speed vs Accuracy
# FASTER (less accurate)
DETECTION_SCALE = 0.5
PROCESS_EVERY_N_FRAMES = 2
VOTE_WINDOW = 2

# MORE ACCURATE (slower)
DETECTION_SCALE = 1.0
PROCESS_EVERY_N_FRAMES = 1
VOTE_WINDOW = 5

Model Selection
MODEL_NAME = "Facenet"     # 128-dim, faster
# or
MODEL_NAME = "Facenet512"  # 512-dim, more accurate

Flask Dashboard

Live camera frame preview (saved JPEG)

Live status via JSON

User management (add / edit / delete)

Attendance reports, exports

Email notifications

Charts & analytics

Admin auth

Main Entry Point (CLI)
# Start live recognition
python main.py engine --camera 0

# View attendance
python main.py check --date today
python main.py check --days 7 --user "John"

# List users
python main.py users

# Launch web dashboard
python main.py dashboard

Installation & Setup
# 1) Install dependencies
pip install -r requirements.txt

# 2) Initialize database
python -c "from db import init_db; init_db()"

# 3) Enroll first user
python enroll_webcam.py --name "YourName" --samples 5

# 4) Start recognition engine
python main.py engine

# 5) (Optional) Start web dashboard
python main.py dashboard

Security & Privacy

Face embeddings are vectors, not photos; can’t be reversed.

Admin password & rate limiting on sensitive endpoints.

Keep secrets in .env (never hardcode). Use:

from dotenv import load_dotenv; load_dotenv()


Use HTTPS in production.

Troubleshooting

No face detected

Improve lighting; face camera directly; move closer (≥ 80×80 px).

Unknown person

Re-enroll with better photos; adjust DIST_THRESHOLD; verify DB entry.

Slow performance

Lower DETECTION_SCALE (e.g., 0.4); increase PROCESS_EVERY_N_FRAMES (e.g., 3); use Facenet instead of Facenet512.

Technical Details

Embedding: 128/512-dimensional DeepFace (Facenet/Facenet512).

Cosine Distance: 0.0 identical → 2.0 opposite; we use 0.50.

Voting: Multiple consecutive confirmations to avoid single-frame errors.

With good enrollment & lighting, targets ~90% accuracy.
