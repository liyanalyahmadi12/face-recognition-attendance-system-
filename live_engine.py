# live_engine.py — IMPROVED: Faster detection + detailed console logs + unknown detection
import os
import cv2
import json
import time
import datetime as dt
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from db import init_db, SessionLocal, User, Attendance, current_quarter_id
from attendance_logic import gate_for_now, write_gate_timestamp
from recognition import FaceRecognitionEngine

# ────────────────────────────────────────────────
# Dashboard hook
ENGINE_FRAME_PATH = os.getenv(
    "ENGINE_FRAME_PATH",
    os.path.join(os.path.dirname(__file__), "static", "last_frame.jpg")
)
ENGINE_STATUS_PATH = os.getenv(
    "ENGINE_STATUS_PATH",
    os.path.join(os.path.dirname(__file__), "static", "last_status.json")
)
os.makedirs(os.path.dirname(ENGINE_FRAME_PATH), exist_ok=True)

ENGINE_SHOW = os.getenv("ENGINE_SHOW", "1") == "1"
ENGINE_JPEG_QUALITY = int(os.getenv("ENGINE_JPEG_QUALITY", "82"))

# ────────────────────────────────────────────────
# Config - FASTER DETECTION
MODEL_NAME       = os.getenv("MODEL_NAME", "Facenet")
DIST_THRESHOLD   = float(os.getenv("DIST_THRESHOLD", "0.50"))
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.70"))
CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX", "-1"))
DETECTION_SCALE  = float(os.getenv("DETECTION_SCALE", "0.5"))
MIN_FACE_SIZE    = int(os.getenv("MIN_FACE_SIZE", "40"))

# SPEED: Process every Nth frame
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))

# Voting - REDUCED for faster response
VOTE_WINDOW      = int(os.getenv("VOTE_WINDOW", "2"))  # Was 3, now 2
VOTE_MIN_SAME    = int(os.getenv("VOTE_MIN_SAME", "2"))  # Just 2 votes needed
COOLDOWN_SEC     = int(os.getenv("COOLDOWN_SEC", "5"))

# Quality check
MIN_FACE_QUALITY = float(os.getenv("MIN_FACE_QUALITY", "25.0"))

# Business rule
ALLOW_GATE_OVERWRITE = bool(int(os.getenv("ALLOW_GATE_OVERWRITE", "1")))

SHOW_DISTANCE    = os.getenv("SHOW_DISTANCE", "1") == "1"
SHOW_BOXES       = os.getenv("SHOW_BOXES", "1") == "1"
SHOW_INSTRUCTIONS = os.getenv("SHOW_INSTRUCTIONS", "1") == "1"

# ────────────────────────────────────────────────
# Detector imports
def _try_import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception:
        return None

# ────────────────────────────────────────────────
# Camera
def open_capture(preferred: Optional[int] = None) -> cv2.VideoCapture:
    backends: List[int] = []
    if hasattr(cv2, "CAP_DSHOW"): backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):  backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    indices = [preferred] if (preferred is not None and preferred >= 0) else [0, 1, 2, 3]
    for idx in indices:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[camera] using index {idx} backend {be}")
                    return cap
            cap.release()
    raise RuntimeError("Cannot open camera")

# ────────────────────────────────────────────────
# Detector
class CombinedDetector:
    def __init__(self):
        self._init_mediapipe()
        self._init_haar()

    def _init_mediapipe(self):
        self.mp = _try_import_mediapipe()
        self.mp_detector = None
        if self.mp:
            try:
                self.mp_detector = self.mp.solutions.face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=0.6
                )
                print("[det] MediaPipe loaded")
            except Exception:
                self.mp = None
                self.mp_detector = None

    def _init_haar(self):
        self.haar: List[cv2.CascadeClassifier] = []
        for name in ("haarcascade_frontalface_default.xml",
                     "haarcascade_frontalface_alt2.xml"):
            c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            if not c.empty():
                self.haar.append(c)
        if self.haar:
            print("[det] Haar loaded")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame_bgr: np.ndarray, scale: float, min_size: int) -> List[Tuple[int,int,int,int]]:
        H, W = frame_bgr.shape[:2]
        boxes: List[Tuple[int,int,int,int]] = []

        # MediaPipe
        if self.mp_detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                res = self.mp_detector.process(rgb)
                if res and res.detections:
                    for d in res.detections:
                        b = d.location_data.relative_bounding_box
                        x1 = int(b.xmin * W); y1 = int(b.ymin * H)
                        w  = int(b.width * W); h = int(b.height * H)
                        if w >= min_size and h >= min_size:
                            boxes.append((max(0,x1), max(0,y1), min(W-x1,w), min(H-y1,h)))
                if boxes:
                    return boxes
            except Exception:
                pass

        # Haar fallback
        if self.haar:
            small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame_bgr
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray  = self.clahe.apply(gray)
            for det in self.haar:
                faces = det.detectMultiScale(
                    gray, 
                    scaleFactor=1.07, 
                    minNeighbors=4,
                    minSize=(int(min_size*scale), int(min_size*scale)),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in faces:
                    boxes.append((int(x/scale), int(y/scale), int(w/scale), int(h/scale)))
            if boxes:
                return _nms(boxes, 0.35)

        return boxes

def _nms(boxes: List[Tuple[int,int,int,int]], thr=0.35) -> List[Tuple[int,int,int,int]]:
    if not boxes: return []
    b = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,0]+b[:,2], b[:,1]+b[:,3]
    area = b[:,2]*b[:,3]
    idxs = np.argsort(area)[::-1]
    keep: List[int] = []
    while len(idxs):
        i = idxs[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        overlap = (w*h) / (area[idxs[1:]] + 1e-8)
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > thr)[0] + 1)))
    return [tuple(map(int, boxes[i])) for i in keep]

# ────────────────────────────────────────────────
# Helpers
def load_known_users(sess) -> List[Dict[str, Any]]:
    users = []
    for u in sess.query(User).all():
        vec = np.asarray(json.loads(u.face_embedding), dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        users.append({"user_id": u.user_id, "name": u.name, "embedding": vec})
    return users

def draw_instruction_overlay(frame: np.ndarray, instruction: str, color: Tuple[int,int,int], show: bool = True):
    """Draw helpful instructions on screen"""
    if not show:
        return
    
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - 80), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, instruction, (20, H - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def print_detection_log(name: str, gate_id: int, confidence: float, distance: float, is_unknown: bool = False):
    """Print detailed detection log to console"""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    if is_unknown:
        print(f"UNKNOWN PERSON DETECTED")
        print(f"Time: {timestamp}")
        print(f"Gate: {gate_id}")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"Distance: {distance:.3f}")
        print(f"Status: REJECTED - Not in database")
    else:
        accuracy_pct = confidence * 100
        print(f"PERSON DETECTED: {name}")
        print(f"Check-in Time: {timestamp}")
        print(f"Gate: Check-in {gate_id}")
        print(f"Match Accuracy: {accuracy_pct:.1f}%")
        print(f"Distance Score: {distance:.3f}")
        print(f"Status: ACCEPTED - Check-in recorded")
    print("="*70 + "\n")

# ────────────────────────────────────────────────
# Main loop
def main():
    init_db()

    # camera
    cam_idx = CAMERA_INDEX if CAMERA_INDEX >= 0 else None
    cap = open_capture(preferred=cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # db + known users
    sess = SessionLocal()
    known_users = load_known_users(sess)
    print(f"[db] users: {len(known_users)}")

    # engines
    detector = CombinedDetector()
    recog    = FaceRecognitionEngine(model_name=MODEL_NAME, dist_threshold=DIST_THRESHOLD)

    votes: deque = deque(maxlen=VOTE_WINDOW)
    last_logged: Dict[Tuple[int,int], float] = defaultdict(float)
    last_face_box: Optional[Tuple[int,int,int,int]] = None
    last_dist: Optional[float] = None
    last_quality: float = 0.0
    show_instructions = SHOW_INSTRUCTIONS

    # fps
    t0 = time.time(); n = 0; fps = 0.0

    frame_count = 0
    boxes: List[Tuple[int,int,int,int]] = []

    stats = {
        "total_attempts": 0,
        "successful_matches": 0,
        "failed_matches": 0,
        "unknown_rejections": 0,
        "poor_quality_rejections": 0,
        "gate_logs": defaultdict(int),
    }

    print(f"[engine] model={MODEL_NAME} dist_thr={DIST_THRESHOLD:.3f} conf_thr={MIN_CONFIDENCE:.2f}")
    print(f"[engine] voting={VOTE_MIN_SAME}/{VOTE_WINDOW} (FASTER)")
    print("q=quit, r=reload users, i=toggle instructions, s=show stats")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame_count += 1

            n += 1
            now_sec = time.time()
            if now_sec - t0 >= 1.0:
                fps = n / (now_sec - t0)
                t0 = now_sec
                n = 0

            gate_id = gate_for_now()

            # Gate timing display (TOP LEFT)
            if gate_id:
                from attendance_logic import GATES
                gate_info = GATES.get(gate_id, {"label": "Unknown", "window": ("??:??", "??:??")})
                gate_text = f"Gate {gate_id}: {gate_info['label']}"
                gate_time = f"{gate_info['window'][0]} - {gate_info['window'][1]}"
                
                cv2.putText(frame, gate_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, gate_time, (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # SPEED: Process every Nth frame
            should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

            if should_process:
                boxes = detector.detect(frame, scale=DETECTION_SCALE, min_size=MIN_FACE_SIZE)
            
            if SHOW_BOXES:
                for (bx, by, bw, bh) in boxes:
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 1)

            instruction = ""
            instruction_color = (255, 255, 255)

            # Choose largest + identify
            if boxes:
                x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
                last_face_box = (x, y, w, h)
                
                # CRITICAL: Validate face size before processing
                if w < 80 or h < 80:
                    votes.append((None, None, None, None))
                    instruction = "Face too small - Move closer to camera"
                    instruction_color = (0, 165, 255)
                    print(f"[DEBUG] Face too small: {w}x{h} - skipping")
                else:
                    pad = int(max(w, h) * 0.30)
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                    face = frame[y1:y2, x1:x2]
                    
                    # Additional validation
                    if face.size == 0 or face.shape[0] < 50 or face.shape[1] < 50:
                        votes.append((None, None, None, None))
                        instruction = "Invalid face crop - Adjust position"
                        instruction_color = (0, 165, 255)
                        print(f"[DEBUG] Invalid crop: {face.shape}")
                    else:
                        try:
                            matches = recog.identify(face, known_users, top_k=1, detector_backend_override="skip")
                            if matches:
                                m = matches[0]
                                last_dist = m.distance
                                last_quality = m.quality_score
                                
                                # DEBUG: Print what we detected
                                print(f"[DEBUG] Face detected - Name: {m.name if m.matched else 'Unknown'}, "
                                      f"Distance: {m.distance:.3f}, Quality: {m.quality_score:.1f}, "
                                      f"Confidence: {m.confidence:.2f}")
                                
                                # Check if it's an unknown person (distance too high)
                                if m.distance > DIST_THRESHOLD:
                                    votes.append(("UNKNOWN", "UNKNOWN", m.distance, m.confidence))
                                    instruction = "Unknown person detected - Not in database"
                                    instruction_color = (0, 0, 255)
                                    print(f"[DEBUG] Unknown person - distance {m.distance:.3f} > threshold {DIST_THRESHOLD}")
                                # Check match criteria for known users
                                elif m.matched and m.quality_score >= MIN_FACE_QUALITY and m.confidence >= MIN_CONFIDENCE:
                                    votes.append((m.user_id, m.name, m.distance, m.confidence))
                                    vote_count = len([v for v in votes if v[1] == m.name])
                                    instruction = f"Confirming {m.name}... ({vote_count}/{VOTE_MIN_SAME})"
                                    instruction_color = (0, 255, 255)
                                    print(f"[DEBUG] Match found: {m.name} - Vote {vote_count}/{VOTE_MIN_SAME}")
                                else:
                                    votes.append((None, None, m.distance, m.confidence))
                                    if m.quality_score < MIN_FACE_QUALITY:
                                        instruction = "Move closer or improve lighting"
                                        instruction_color = (0, 165, 255)
                                        print(f"[DEBUG] Quality too low: {m.quality_score:.1f} < {MIN_FACE_QUALITY}")
                                    else:
                                        instruction = "Hold still... Analyzing"
                                        instruction_color = (255, 255, 255)
                                        print(f"[DEBUG] Analyzing... conf={m.confidence:.2f}, matched={m.matched}")
                            else:
                                votes.append((None, None, None, None))
                        except Exception as e:
                            print(f"[ERROR] Recognition failed: {e}")
                            votes.append((None, None, None, None))
                            instruction = "Recognition error - Reposition face"
                            instruction_color = (255, 0, 0)
            else:
                votes.append((None, None, None, None))
                instruction = "No face detected - Position yourself in view"
                instruction_color = (0, 0, 255)

            # Voting
            voted_uid: Optional[int] = None
            voted_name: Optional[str] = None
            voted_dist: Optional[float] = None
            voted_conf: Optional[float] = None
            is_unknown = False

            if len(votes) >= VOTE_MIN_SAME:
                counts: Dict[Any, int] = {}
                dists: Dict[Any, List[float]] = {}
                confs: Dict[Any, List[float]] = {}
                
                for uid, name, dist, conf in votes:
                    if uid is not None or name == "UNKNOWN":
                        key = name if name == "UNKNOWN" else uid
                        counts[key] = counts.get(key, 0) + 1
                        if dist is not None:
                            dists.setdefault(key, []).append(float(dist))
                        if conf is not None:
                            confs.setdefault(key, []).append(float(conf))
                
                if counts:
                    best = max(counts, key=lambda k: counts[k])
                    if counts[best] >= VOTE_MIN_SAME:
                        if best == "UNKNOWN":
                            is_unknown = True
                            voted_name = "UNKNOWN"
                            if dists.get("UNKNOWN"):
                                voted_dist = sum(dists["UNKNOWN"]) / len(dists["UNKNOWN"])
                            if confs.get("UNKNOWN"):
                                voted_conf = sum(confs["UNKNOWN"]) / len(confs["UNKNOWN"])
                        else:
                            voted_uid = best
                            for uid, name, dist, conf in reversed(votes):
                                if uid == best:
                                    voted_name = name
                                    break
                            if dists.get(best):
                                voted_dist = sum(dists[best]) / len(dists[best])
                            if confs.get(best):
                                voted_conf = sum(confs[best]) / len(confs[best])

            # Message + DB write
            msg = "NO GATE OPEN"
            color = (0, 0, 255)
            remaining = 0

            if gate_id:
                if is_unknown:
                    # Unknown person detected
                    msg = "⚠ UNKNOWN PERSON - Not in Database"
                    color = (0, 0, 255)
                    instruction = "Unknown person - Access denied"
                    instruction_color = (0, 0, 255)
                    
                    # Print to console
                    print_detection_log("UNKNOWN", gate_id, voted_conf or 0.0, voted_dist or 1.0, is_unknown=True)
                    votes.clear()  # Clear votes after logging unknown
                    
                elif voted_uid is not None:
                    key = (voted_uid, gate_id)
                    elapsed = time.time() - last_logged[key]
                    if elapsed >= COOLDOWN_SEC:
                        ts = dt.datetime.now()
                        date_iso = ts.date().isoformat()
                        rec = (sess.query(Attendance)
                               .filter_by(user_id=voted_uid, date=date_iso)
                               .one_or_none())
                        if not rec:
                            rec = Attendance(
                                user_id=voted_uid,
                                date=date_iso,
                                quarter_id=current_quarter_id(ts.date())
                            )
                            sess.add(rec)
                            sess.flush()
                        if write_gate_timestamp(rec, gate_id, ts.strftime("%H:%M:%S"), overwrite=True):
                            sess.commit()
                            last_logged[key] = time.time()
                            msg = f"✓ RECOGNIZED: {voted_name} → Gate {gate_id}"
                            color = (0, 255, 0)
                            instruction = f"Success! {voted_name} checked in"
                            instruction_color = (0, 255, 0)
                            
                            # Print detailed log to console
                            print_detection_log(voted_name, gate_id, voted_conf or 0.0, voted_dist or 0.0)
                            
                            # CRITICAL: Clear votes and reset detection to stop loop
                            votes.clear()
                            voted_uid = None
                            voted_name = None
                            last_face_box = None
                        else:
                            msg = f"{voted_name} → Gate {gate_id} (already logged)"
                            color = (255, 255, 0)
                            instruction = f"{voted_name} already checked in"
                            instruction_color = (255, 255, 0)
                    else:
                        remaining = max(0, COOLDOWN_SEC - int(elapsed))
                        msg = f"{voted_name} - Cooldown {remaining}s"
                        color = (255, 165, 0)
                        instruction = f"Wait {remaining}s before next check-in"
                        instruction_color = (255, 165, 0)
                else:
                    msg = f"Gate {gate_id} OPEN - Scanning..."
                    color = (0, 255, 255)

            # Draw focus box
            if last_face_box:
                lx, ly, lw, lh = last_face_box
                
                if is_unknown:
                    box_color = (0, 0, 255)  # Red = unknown
                elif voted_uid:
                    box_color = (0, 255, 0)  # Green = recognized
                elif last_quality < MIN_FACE_QUALITY:
                    box_color = (0, 0, 255)  # Red = poor quality
                else:
                    box_color = (0, 255, 255)  # Yellow = analyzing
                    
                cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), box_color, 3)
                
                if voted_name:
                    label = voted_name
                    label_size = 0.8
                else:
                    label = "Analyzing..."
                    label_size = 0.6
                    
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_size, 2)
                cv2.rectangle(frame, (lx, ly - th - 12), (lx + tw + 12, ly), box_color, -1)
                cv2.putText(frame, label, (lx + 6, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, label_size, (0, 0, 0), 2)

            # Status message (center top, below gate info)
            cv2.rectangle(frame, (0, 70), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # FPS
            cv2.putText(frame, f"FPS {fps:.1f}  Users: {len(known_users)}",
                        (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw instructions
            if show_instructions and instruction:
                draw_instruction_overlay(frame, instruction, instruction_color, True)

            # Dashboard hook
            try:
                cv2.imwrite(ENGINE_FRAME_PATH, frame, [int(cv2.IMWRITE_JPEG_QUALITY), ENGINE_JPEG_QUALITY])
                status = {
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "name": voted_name or "",
                    "gate": gate_id or None,
                    "distance": None if voted_dist is None else float(voted_dist),
                    "confidence": None if voted_conf is None else float(voted_conf),
                    "quality": float(last_quality),
                    "cooldown": int(remaining)
                }
                with open(ENGINE_STATUS_PATH, "w", encoding="utf-8") as f:
                    json.dump(status, f)
            except Exception:
                pass

            # Window
            if ENGINE_SHOW:
                cv2.imshow("Face Recognition Attendance (q=quit, r=reload, i=help, s=stats)", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                time.sleep(0.01)
                key = -1

            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                known_users = load_known_users(sess)
                print(f"[db] reloaded users: {len(known_users)}")
            elif key == ord('i'):
                show_instructions = not show_instructions
                print(f"[engine] instructions {'ON' if show_instructions else 'OFF'}")
            elif key == ord('s'):
                print("\n" + "="*60)
                print("RECOGNITION STATISTICS")
                print("="*60)
                total = stats["total_attempts"]
                if total > 0:
                    accuracy = (stats["successful_matches"] / total * 100) if total > 0 else 0
                    print(f"Total Recognition Attempts: {total}")
                    print(f"Successful Matches: {stats['successful_matches']}")
                    print(f"Failed Matches: {stats['failed_matches']}")
                    print(f"Unknown Rejections: {stats['unknown_rejections']}")
                    print(f"Poor Quality Rejections: {stats['poor_quality_rejections']}")
                    print(f"\nACCURACY RATE: {accuracy:.1f}%")
                else:
                    print("No recognition attempts yet")
                print("="*60 + "\n")

    finally:
        cap.release()
        if ENGINE_SHOW:
            cv2.destroyAllWindows()
        sess.close()
        print("[engine] cleanup complete")

# Export for main.py compatibility
engine_main = main

if __name__ == "__main__":
    main()