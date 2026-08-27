import cv2
from pathlib import Path

def sample_full_roi_frames():
    video_path = Path("data/annotated/bidirectional_full_roi_test.mp4")
    cap = cv2.VideoCapture(str(video_path))
    
    out_dir = Path("C:/Users/piyus/.gemini/antigravity-ide/brain/44cf297f-c842-4786-98c9-ee1a7dccb5c8/.tempmediaStorage")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sample_indices = [30, 100, 200, 300]
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx in sample_indices:
            out_file = out_dir / f"bidirectional_full_frame_{frame_idx}.jpg"
            cv2.imwrite(str(out_file), frame)
            print(f"Saved {out_file}")
    cap.release()

if __name__ == "__main__":
    sample_full_roi_frames()
