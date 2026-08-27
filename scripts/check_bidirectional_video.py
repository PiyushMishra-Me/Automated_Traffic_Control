import cv2
from pathlib import Path

video_path = Path("data/uploads/bidirectional.mp4")
if not video_path.exists():
    print(f"File not found: {video_path}")
else:
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {video_path}")
    print(f"Width: {w}, Height: {h}, FPS: {fps:.2f}, Total Frames: {frames}")
    
    # Save a first frame to inspect
    ret, frame = cap.read()
    if ret:
        out_path = Path("C:/Users/piyus/.gemini/antigravity-ide/brain/44cf297f-c842-4786-98c9-ee1a7dccb5c8/.tempmediaStorage/bidirectional_frame_1.jpg")
        cv2.imwrite(str(out_path), frame)
        print(f"Saved initial frame to {out_path}")
    cap.release()
