import cv2
import numpy as np
import csv
import time
import os
import requests
import subprocess
from collections import defaultdict
from pytube import YouTube
from filterpy.kalman import KalmanFilter

# ======================
# Configuration Settings
# ======================
VIDEO_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
OUTPUT_VIDEO = "processed_video.mp4"
OUTPUT_CSV = "traffic_data.csv"
VEHICLE_CLASSES = [2, 5, 7]  # COCO: car=2, bus=5, truck=7
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
TRACKING_PARAMS = {"max_age": 20, "min_hits": 3, "iou_threshold": 0.3}

# Lane definitions (adjust based on video perspective)
LANES = [
    {"id": 1, "polygon": np.array([[200, 720], [400, 500], [600, 500], [400, 720]]), "color": (0, 255, 0)},
    {"id": 2, "polygon": np.array([[600, 720], [800, 500], [1000, 500], [800, 720]]), "color": (0, 255, 255)},
    {"id": 3, "polygon": np.array([[1000, 720], [1200, 500], [1400, 500], [1200, 720]]), "color": (0, 0, 255)}
]

# ======================
# Utility Functions
# ======================
def download_file(url, filename):
    """Download a file from a URL with progress indicator"""
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = int(100 * downloaded / total_size) if total_size > 0 else 0
                        print(f"\rProgress: {percent}%", end='', flush=True)
            print("\nDownload complete!")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False
    return True

def download_youtube_video(url, filename):
    """Download a YouTube video using pytube"""
    try:
        print(f"Downloading YouTube video: {url}")
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension='mp4', res='720p').first()
        stream.download(filename=filename)
        print(f"Video downloaded to {filename}")
        return True
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return False

def load_yolo():
    """Load YOLOv4 model with automatic weight downloading"""
    # Create directory for YOLO files
    os.makedirs("yolov4", exist_ok=True)
    
    # Define paths
    cfg_path = "yolov4/yolov4.cfg"
    weights_path = "yolov4/yolov4.weights"
    names_path = "yolov4/coco.names"
    
    # Download weights if missing
    if not os.path.exists(weights_path):
        if not download_file(
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            weights_path
        ):
            return None, None, None
    
    # Download cfg if missing
    if not os.path.exists(cfg_path):
        if not download_file(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            cfg_path
        ):
            return None, None, None
    
    # Download coco.names if missing
    if not os.path.exists(names_path):
        if not download_file(
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            names_path
        ):
            return None, None, None
    
    # Load YOLO model
    try:
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        
        print("Using CPU for YOLOv4")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layers
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Load class names
        with open(names_path, 'r') as f:
            classes = f.read().strip().split('\n')
        
        print("YOLOv4 model loaded successfully")
        return net, ln, classes
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None, None

def detect_vehicles(frame, net, ln, vehicle_classes):
    """Detect vehicles in a frame using YOLOv4"""
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONF_THRESHOLD and class_id in vehicle_classes:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    if len(idxs) > 0:
        boxes = [boxes[i] for i in idxs.flatten()]
        confidences = [confidences[i] for i in idxs.flatten()]
        class_ids = [class_ids[i] for i in idxs.flatten()]
        return boxes, confidences, class_ids
    
    return [], [], []

def draw_lanes(frame, lanes, counts):
    """Draw lane boundaries and counts on the frame"""
    for lane in lanes:
        pts = lane["polygon"].reshape((-1,1,2))
        cv2.polylines(frame, [pts], True, lane["color"], 2)
        cv2.putText(frame, f'Lane {lane["id"]}: {counts[lane["id"]]}', 
                   tuple(pts[0][0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane["color"], 2)
        
        # Draw lane label at top
        top_center = np.mean(pts, axis=0)[0]
        cv2.putText(frame, f'L{lane["id"]}', 
                   (int(top_center[0]) - 20, int(top_center[1]) - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, lane["color"], 2)

# ======================
# SORT Tracker Implementation
# ======================
def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)

def convert_bbox_to_z(bbox):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if score is None:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
            
    if len(matches) == 0:
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class SORT:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

# ======================
# Main Application
# ======================
def main():
    # Download YouTube video
    video_path = "traffic_video.mp4"
    if not os.path.exists(video_path):
        if not download_youtube_video(VIDEO_URL, video_path):
            print("Failed to download video. Exiting.")
            return
    
    # Load YOLO model
    net, ln, classes = load_yolo()
    if net is None:
        print("Failed to load YOLO model. Exiting.")
        return
    
    # Initialize tracker
    tracker = SORT(**TRACKING_PARAMS)
    
    # Initialize counters
    lane_counts = {1: 0, 2: 0, 3: 0}
    counted_ids = set()
    tracking_history = defaultdict(list)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Vehicle_ID', 'Lane', 'Frame', 'Timestamp', 'Entry_Time', 'Exit_Time'])
    
    frame_count = 0
    start_time = time.time()
    vehicle_data = {}
    
    print("Starting processing...")
    print("Press 'q' to quit processing early")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        timestamp = frame_count / fps
        
        # Detect vehicles
        boxes, confidences, class_ids = detect_vehicles(frame, net, ln, VEHICLE_CLASSES)
        
        # Prepare detections for SORT
        detections = []
        for box, conf in zip(boxes, confidences):
            x, y, w, h = box
            detections.append([x, y, x+w, y+h, conf])
        detections = np.array(detections)
        
        # Update tracker
        tracked_objects = tracker.update(detections) if len(detections) > 0 else np.empty((0, 5))
        
        current_ids = set()
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            obj_id = int(obj_id)
            current_ids.add(obj_id)
            
            # Calculate center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Store tracking history
            tracking_history[obj_id].append((cx, cy, frame_count, timestamp))
            
            # Check lane assignment
            if obj_id not in counted_ids:
                for lane in LANES:
                    if cv2.pointPolygonTest(lane["polygon"], (cx, cy), False) >= 0:
                        counted_ids.add(obj_id)
                        lane_counts[lane["id"]] += 1
                        
                        # Record entry time
                        entry_time = timestamp
                        
                        # Create vehicle record
                        vehicle_data[obj_id] = {
                            "lane": lane["id"],
                            "entry_frame": frame_count,
                            "entry_time": entry_time,
                            "exit_frame": None,
                            "exit_time": None
                        }
                        
                        csv_writer.writerow([obj_id, lane["id"], frame_count, timestamp, entry_time, None])
                        break
        
        # Update exit time for vehicles that left
        for obj_id in list(vehicle_data.keys()):
            if obj_id not in current_ids and vehicle_data[obj_id]["exit_time"] is None:
                last_seen = tracking_history[obj_id][-1]
                exit_time = last_seen[3]
                vehicle_data[obj_id]["exit_frame"] = last_seen[2]
                vehicle_data[obj_id]["exit_time"] = exit_time
                
                # Update CSV with exit time
                csv_writer.writerow([
                    obj_id, 
                    vehicle_data[obj_id]["lane"], 
                    last_seen[2], 
                    exit_time,
                    vehicle_data[obj_id]["entry_time"],
                    exit_time
                ])
        
        # Draw UI elements
        draw_lanes(frame, LANES, lane_counts)
        
        # Display lane counts
        cv2.putText(frame, f"Lane 1: {lane_counts[1]}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Lane 2: {lane_counts[2]}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Lane 3: {lane_counts[3]}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (width-300, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (width-300, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Display frame
        cv2.imshow('Traffic Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print progress
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}/{total_frames} - FPS: {fps:.1f}")
    
    # After processing all frames, handle vehicles that never exited
    for obj_id, data in vehicle_data.items():
        if data["exit_time"] is None:
            if obj_id in tracking_history and tracking_history[obj_id]:
                last_seen = tracking_history[obj_id][-1]
                exit_frame = last_seen[2]
                exit_time = last_seen[3]
                data["exit_frame"] = exit_frame
                data["exit_time"] = exit_time
                csv_writer.writerow([
                    obj_id, 
                    data["lane"], 
                    exit_frame, 
                    exit_time,
                    data["entry_time"],
                    exit_time
                ])
    
    # Cleanup
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n--- Traffic Summary ---")
    print(f"Lane 1: {lane_counts[1]} vehicles")
    print(f"Lane 2: {lane_counts[2]} vehicles")
    print(f"Lane 3: {lane_counts[3]} vehicles")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"Average FPS: {frame_count/(time.time() - start_time):.1f}")
    print(f"Output video saved to: {OUTPUT_VIDEO}")
    print(f"CSV data saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    # Install required packages
    print("Checking required packages...")
    required_packages = [
        "opencv-python",
        "numpy",
        "requests",
        "pytube",
        "filterpy",
        "lap"
    ]
    
    for package in required_packages:
        try:
            __import__(package.split("==")[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call(["pip", "install", package])
    
    # Run main application
    main()
