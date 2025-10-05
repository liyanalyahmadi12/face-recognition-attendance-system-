(note the file has a gitignore as we have api key for email notification that sends absent and late users email as shown  <img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/6d437adf-3d35-437f-a2bc-bd13bc749515" />

Smart Attendance System - Complete DocumentationSystem OverviewThis is a real-time face recognition attendance system that automatically 
identifies people via webcam and logs their check-ins at designated times throughout the day.Core Components Architecture
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   Camera     │─────▶│ Face Detect  │                     │
│  │   Input      │      │ (MediaPipe/  │                     │
│  └──────────────┘      │   Haar)      │                     │
│                        └──────┬───────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                     │
│                        │  Crop Face   │                     │
│                        │  + Preproc   │                     │
│                        └──────┬───────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                     │
│                        │  DeepFace    │                     │
│                        │  Embedding   │                     │
│                        │  (512-dim)   │                     │
│                        └──────┬───────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                     │
│                        │   Compare    │                     │
│                        │  w/ Database │                     │
│                        │  Embeddings  │                     │
│                        └──────┬───────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                     │
│                        │  Voting      │                     │
│                        │  System      │                     │
│                        │  (2/2 votes) │                     │
│                        └──────┬───────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                     │
│                        │  Log to DB   │                     │
│                        │  (5s cooldown)│                    │
│                        └──────────────┘                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
File-by-File Explanation
1. db.py - Database Layer
Defines the database schema and connection:

User Table: Stores person info and face embeddings
Attendance Table: Logs check-ins with 4 daily gates
Session Management: SQLAlchemy ORM for database operations

pythonUser:
  - user_id (primary key)
  - name (unique)
  - face_embedding (JSON string of 512 numbers)

Attendance:
  - record_id (primary key)
  - user_id (foreign key)
  - date (YYYY-MM-DD)
  - checkin1_time, checkin2_time, checkin3_time, checkin4_time
  - quarter_id (for reporting)
2. attendance_logic.py - Business Rules
Manages the 4 daily check-in windows:

Morning Start: 07:00-09:00
Lunch Out: 12:00-13:30
Lunch In: 13:00-14:30
Evening End: 17:00-19:30

Functions:

gate_for_now() - Returns which gate is currently open
write_gate_timestamp() - Writes time to correct column
compute_lateness_minutes() - Calculates if person is late

3. recognition.py - Face Recognition Engine
The core ML module:
pythonclass FaceRecognitionEngine:
    - Loads DeepFace model (Facenet/Facenet512)
    - Converts face images → 512-dim vectors
    - Compares vectors via cosine distance
    - Returns match results with confidence scores
Key functions:

embed_face() - Converts face image to embedding vector
identify() - Compares query face against known database
cosine_distances() - Measures similarity (1 - dot product)

4. seed_users.py - Initial Enrollment
Batch enrolls users from photos:

Reads images from seed_images/ folder
Detects face using Haar cascades + fallbacks
Generates embedding via DeepFace
Stores in database with person's name

Usage: python seed_users.py
5. enroll_webcam.py - Live Enrollment
Interactive enrollment via webcam:

Opens camera feed
User presses SPACE to capture multiple samples
Averages embeddings for robustness
Saves to database

Usage: python enroll_webcam.py --name "John Doe" --samples 5

live_engine.py - The Heart of the System
This is the real-time recognition engine that runs continuously. Let me explain in detail:
Detection & Recognition Flow
┌─────────────────────────────────────────────────────────────┐
│               LIVE_ENGINE.PY PROCESS FLOW                    │
└─────────────────────────────────────────────────────────────┘

1. CAPTURE FRAME
   │
   ▼
2. FACE DETECTION (every 2nd frame for speed)
   ├─ Try MediaPipe (fast, accurate)
   └─ Fallback to Haar Cascades
   │
   ▼
3. FIND LARGEST FACE
   │
   ▼
4. VALIDATE SIZE (must be ≥80x80 pixels)
   │
   ▼
5. CROP WITH 30% MARGIN
   │
   ▼
6. PREPROCESSING
   ├─ Convert BGR → RGB
   ├─ Apply CLAHE (contrast normalization)
   └─ Standardize lighting
   │
   ▼
7. EMBEDDING GENERATION (skip detector, we already cropped)
   ├─ DeepFace.represent()
   ├─ Model: Facenet (128-dim) or Facenet512 (512-dim)
   └─ Output: Normalized 512-float vector
   │
   ▼
8. COMPARISON WITH DATABASE
   ├─ Load all known embeddings
   ├─ Compute cosine distances
   └─ Distance = 1 - dot_product(query, known)
   │
   ▼
9. THRESHOLDING & CONFIDENCE
   ├─ Distance threshold: 0.50 (lower = stricter)
   ├─ Confidence threshold: 0.70 (70% minimum)
   └─ Quality check: ≥25.0 (blur + brightness)
   │
   ▼
10. VOTING SYSTEM (prevent false positives)
    ├─ Window: 2 consecutive frames
    ├─ Required: 2 matching votes
    └─ Same person must appear in both
    │
    ▼
11. DECISION
    ├─ MATCHED → Check cooldown (5 seconds)
    ├─ UNKNOWN → Show red box + alert
    └─ NO MATCH → Keep scanning
    │
    ▼
12. DATABASE LOGGING
    ├─ Check which gate is open (1-4)
    ├─ Write timestamp to correct column
    ├─ Print detailed console log
    └─ Clear votes to prevent duplicate logs
Key Thresholds Explained
DIST_THRESHOLD = 0.50

Cosine distance between embeddings (0.0 = identical, 2.0 = opposite)
0.50 = strict matching (recommended for security)
Lower values = fewer false positives, more false negatives
Higher values = more lenient, risk of wrong person

MIN_CONFIDENCE = 0.70

Derived from distance: confidence = 1 - (distance / 2)
70% = reasonable certainty required
Filters out poor quality detections

MIN_FACE_QUALITY = 25.0

Combined score from:

Blur detection (Laplacian variance)
Brightness (distance from ideal 128 gray level)


Rejects blurry or dark images

Voting Parameters

VOTE_WINDOW = 2: Requires 2 consecutive frames
VOTE_MIN_SAME = 2: Both must agree on same person
COOLDOWN_SEC = 5: Prevents duplicate logs

How Unknown Detection Works
pythonif distance > DIST_THRESHOLD:
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
    # Analyzing (low quality/confidence)
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

Configuration Variables
Speed vs Accuracy Trade-offs
python# FASTER (less accurate)
DETECTION_SCALE = 0.5      # Downscale image before detection
PROCESS_EVERY_N_FRAMES = 2 # Skip every other frame
VOTE_WINDOW = 2            # Quick decision (2 frames)

# MORE ACCURATE (slower)
DETECTION_SCALE = 1.0      # Full resolution detection
PROCESS_EVERY_N_FRAMES = 1 # Process every frame
VOTE_WINDOW = 5            # Longer voting period
Model Selection
pythonMODEL_NAME = "Facenet"      # Fast, 128-dim embedding
MODEL_NAME = "Facenet512"   # Slower, 512-dim (more accurate)

Flask Dashboard (flask_dashboard.py)
Web interface features:

Real-time camera feed display
Live status updates (JSON polling)
User management (add/edit/delete)
Attendance reports with photos
Excel export with embedded images
Email notifications for late/absent
Charts and analytics
Admin authentication


Main Entry Point (main.py)
Command-line interface:
bash# Start live recognition
python main.py engine --camera 0

# View attendance
python main.py check --date today
python main.py check --days 7 --user "John"

# List users
python main.py users

# Launch web dashboard
python main.py dashboard

Installation & Setup
bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database
python -c "from db import init_db; init_db()"

# 3. Enroll first user
python enroll_webcam.py --name "YourName" --samples 5

# 4. Start recognition engine
python main.py engine

# 5. (Optional) Start web dashboard
python main.py dashboard

Security & Privacy

Face embeddings are mathematical representations, not photos
Embeddings cannot be reversed to reconstruct faces
Admin password protection (configurable via .env)
Rate limiting on sensitive endpoints
HTTPS recommended for production


Troubleshooting
"No face detected"

Ensure adequate lighting
Face camera directly
Move closer (face must be ≥80x80 pixels)

"Unknown person"

Re-enroll with better quality photos
Lower DIST_THRESHOLD if too strict
Check if embedding exists in database

Slow performance

Reduce DETECTION_SCALE to 0.4
Increase PROCESS_EVERY_N_FRAMES to 3
Use Facenet instead of Facenet512


Technical Details
Face Embedding: 512-dimensional vector representation extracted by DeepFace's Facenet512 model, trained on millions of faces to capture unique facial features.
Cosine Distance: Measures angle between vectors. Perfect match = 0.0, opposite = 2.0. We use 0.50 as threshold (≈60° angle tolerance).
Voting System: Prevents single-frame errors by requiring consistent detection across multiple frames before logging attendance.
This system achieves ~90% accuracy with proper enrollment and lighting conditions.
