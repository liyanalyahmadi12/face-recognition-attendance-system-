(note the file has a gitignore as we have api key for email notification that sends absent and late users email as shown  <img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/6d437adf-3d35-437f-a2bc-bd13bc749515" />

Smart Attendance System — Complete Documentation

A real-time face recognition attendance system that identifies people via webcam and logs check-ins at designated times throughout the day.

Table of Contents

System Overview

Architecture

File-by-File Explanation

1. db.py — Database Layer

2. attendance_logic.py — Business Rules

3. recognition.py — Face Recognition Engine

4. seed_users.py — Initial Enrollment

5. enroll_webcam.py — Live Enrollment

6. live_engine.py — Real-Time Engine

Key Thresholds

Configuration Variables

Speed vs Accuracy

Model Selection

Flask Dashboard

Main Entry Point (CLI)

Installation & Setup

Security & Privacy

Troubleshooting

Technical Details

System Overview

This system performs face detection, creates embeddings, compares them to a database of enrolled users, and logs attendance to the database when a match is confirmed in an active time window (“gate”).

Architecture
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   Camera     │─────▶│ Face Detect  │                     │
│  │   Input      │      │ (MediaPipe/  │                     │
│  └──────────────┘      │   Haar)      │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │  Crop Face   │                     │
│                        │  + Preproc   │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │  DeepFace    │                     │
│                        │  Embedding   │                     │
│                        │  (512-dim)   │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │   Compare    │                     │
│                        │  w/ Database │                     │
│                        │  Embeddings  │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │  Voting      │                     │
│                        │  System      │                     │
│                        │  (2/2 votes) │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │  Log to DB   │                     │
│                        │ (5s cooldown)│                     │
│                        └──────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

File-by-File Explanation
1. db.py — Database Layer

Defines the database schema and connection (SQLAlchemy ORM).

User Table

user_id (primary key)

name (unique)

face_embedding (JSON string of 512 numbers)

Attendance Table

record_id (primary key)

user_id (foreign key)

date (YYYY-MM-DD)

checkin1_time, checkin2_time, checkin3_time, checkin4_time

quarter_id (for reporting)

Includes session management utilities for DB operations.

2. attendance_logic.py — Business Rules

Manages the 4 daily check-in windows:

Morning Start: 07:00–09:00

Lunch Out: 12:00–13:30

Lunch In: 13:00–14:30

Evening End: 17:00–19:30

Functions

gate_for_now() → returns which gate is currently open (1–4)

write_gate_timestamp() → writes time to correct column

compute_lateness_minutes() → calculates lateness

3. recognition.py — Face Recognition Engine

The core ML module.

class FaceRecognitionEngine:
    # Loads DeepFace model (Facenet/Facenet512)
    # Converts face images → 512-dim vectors
    # Compares vectors via cosine distance
    # Returns match results with confidence scores


Key functions

embed_face() → convert face image to embedding vector

identify() → compare query face against known database

cosine_distances() → similarity via 1 - dot_product

4. seed_users.py — Initial Enrollment

Batch-enroll users from photos:

Reads images from seed_images/

Detects face using Haar cascades + fallbacks

Generates embeddings via DeepFace

Stores in database with person’s name

Usage

python seed_users.py

5. enroll_webcam.py — Live Enrollment

Interactive webcam enrollment:

Opens camera feed

Press SPACE to capture multiple samples

Averages embeddings for robustness

Saves to database

Usage

python enroll_webcam.py --name "John Doe" --samples 5

6. live_engine.py — Real-Time Engine

The real-time recognition engine that runs continuously.

Detection & Recognition Flow
┌─────────────────────────────────────────────────────────────┐
│               LIVE_ENGINE.PY PROCESS FLOW                   │
└─────────────────────────────────────────────────────────────┘

1) CAPTURE FRAME
2) FACE DETECTION (every 2nd frame for speed)
   - Try MediaPipe (fast, accurate)
   - Fallback to Haar Cascades
3) FIND LARGEST FACE
4) VALIDATE SIZE (≥ 80x80 px)
5) CROP WITH 30% MARGIN
6) PREPROCESSING
   - BGR → RGB
   - CLAHE (contrast normalization)
   - Standardize lighting
7) EMBEDDING GENERATION (skip detector, already cropped)
   - DeepFace.represent()
   - Model: Facenet (128-dim) or Facenet512 (512-dim)
   - Output: normalized vector
8) COMPARISON WITH DATABASE
   - Load known embeddings
   - Cosine distance = 1 − dot(query, known)
9) THRESHOLDING & CONFIDENCE
   - Distance ≤ 0.50
   - Confidence ≥ 0.70
   - Quality ≥ 25.0
10) VOTING SYSTEM
   - Window: 2 consecutive frames
   - Required: 2 matching votes
11) DECISION
   - MATCHED → Check cooldown (5s)
   - UNKNOWN → Red box + alert
   - NO MATCH → Keep scanning
12) DATABASE LOGGING
   - Determine open gate (1–4)
   - Write timestamp to correct column
   - Console log; clear votes to avoid duplicates


How Unknown Detection Works

if distance > DIST_THRESHOLD:
    # Person not in database
    voted_name = "UNKNOWN"
    status = "REJECTED - Not in database"
    box_color = RED

elif matched and quality >= 25.0 and confidence >= 0.70:
    # Valid match
    voted_name = person.name
    status = "ACCEPTED - Check-in recorded"
    box_color = GREEN

else:
    # Still analyzing (low quality/confidence)
    status = "Hold still... Analyzing"
    box_color = YELLOW


Example Console Output

======================================================================
PERSON DETECTED: Liyan
Check-in Time: 2025-10-05 08:15:32
Gate: Check-in 1
Match Accuracy: 87.3%
Distance Score: 0.253
Status: ACCEPTED - Check-in recorded
======================================================================

======================================================================
UNKNOWN PERSON DETECTED
Time: 2025-10-05 08:16:45
Gate: 1
Confidence: 45.2%
Distance: 0.658
Status: REJECTED - Not in database
======================================================================

Key Thresholds

DIST_THRESHOLD = 0.50
Cosine distance between embeddings (0.0=identical, 2.0=opposite).

Lower → stricter (fewer false positives, more false negatives)

Higher → more lenient (risk of wrong person)

MIN_CONFIDENCE = 0.70
Derived as confidence = 1 - (distance / 2).

70% = reasonable certainty requirement

MIN_FACE_QUALITY = 25.0
Quality from:

Blur detection (Laplacian variance)

Brightness (distance from ideal gray level ≈128)

Rejects blurry or dark images

Voting Parameters

VOTE_WINDOW = 2 → requires 2 consecutive frames

VOTE_MIN_SAME = 2 → both must agree on same person

COOLDOWN_SEC = 5 → prevents duplicate logs

Configuration Variables
Speed vs Accuracy
# FASTER (less accurate)
DETECTION_SCALE = 0.5        # Downscale before detection
PROCESS_EVERY_N_FRAMES = 2   # Skip every other frame
VOTE_WINDOW = 2              # Quick decision

# MORE ACCURATE (slower)
DETECTION_SCALE = 1.0        # Full-resolution detection
PROCESS_EVERY_N_FRAMES = 1   # Process every frame
VOTE_WINDOW = 5              # Longer voting period

Model Selection
MODEL_NAME = "Facenet"      # Fast, 128-dim
# or
MODEL_NAME = "Facenet512"   # Slower, 512-dim (more accurate)

Flask Dashboard (flask_dashboard.py)

Features:

Real-time camera feed display

Live status updates (JSON polling)

User management (add/edit/delete)

Attendance reports with photos

Excel export with embedded images

Email notifications for late/absent

Charts and analytics

Admin authentication

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

Face embeddings are mathematical representations, not photos.

Embeddings cannot be reversed to reconstruct faces.

Admin password protection (configurable via .env).

Rate-limiting on sensitive endpoints.

HTTPS recommended for production.

Troubleshooting

“No face detected”

Ensure adequate lighting

Face the camera directly

Move closer (face must be ≥ 80×80 px)

“Unknown person”

Re-enroll with better-quality photos

Lower DIST_THRESHOLD if too strict

Check that an embedding exists in the database

Slow performance

Reduce DETECTION_SCALE to 0.4

Increase PROCESS_EVERY_N_FRAMES to 3

Use Facenet instead of Facenet512

Technical Details

Face Embedding: 512-dimensional vector (DeepFace Facenet512) capturing unique facial features.

Cosine Distance: Measures angle between vectors (0.0 perfect match, 2.0 opposite). Threshold used: 0.50 (≈ 60° tolerance).

Voting System: Prevents single-frame errors by requiring consistent detections across multiple frames before logging.

With proper enrollment and lighting, the system targets ~90% accuracy.
