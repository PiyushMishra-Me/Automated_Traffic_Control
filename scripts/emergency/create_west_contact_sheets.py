import cv2
import numpy as np
from pathlib import Path

def create_contact_sheets():
    project_root = Path(__file__).resolve().parent.parent.parent
    video_path = project_root / "data" / "uploads" / "west.mp4"
    
    out_dir = project_root / "data" / "emergency_vehicle_dataset" / "demo_ambulance" / "west_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    
    print(f"Loading west.mp4 ({total_frames} frames, {fps:.1f} fps)...")
    
    # -------------------------------------------------------------
    # 1. Contact Sheet 1: ~30 evenly distributed full-video frames
    # -------------------------------------------------------------
    full_indices = np.linspace(1, total_frames - 1, 30, dtype=int)
    frames_sheet1 = []
    
    for f_idx in full_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Burn in frame number clearly
        vis = frame.copy()
        cv2.rectangle(vis, (10, 10), (220, 50), (0, 0, 0), -1)
        cv2.putText(vis, f"Frame {f_idx:04d} ({f_idx/fps:.1f}s)", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Resize for grid
        resized = cv2.resize(vis, (256, 144), interpolation=cv2.INTER_AREA)
        frames_sheet1.append(resized)
        
    cols1 = 5
    rows1 = int(np.ceil(len(frames_sheet1) / cols1))
    grid1 = np.zeros((rows1 * 144, cols1 * 256, 3), dtype=np.uint8)
    
    for idx, f_img in enumerate(frames_sheet1):
        r = idx // cols1
        c = idx % cols1
        grid1[r * 144:(r + 1) * 144, c * 256:(c + 1) * 256] = f_img
        
    sheet1_path = out_dir / "contact_sheet_full_video_30_frames.jpg"
    cv2.imwrite(str(sheet1_path), grid1)
    print(f"Full video contact sheet saved: {sheet1_path}")
    
    # -------------------------------------------------------------
    # 2. Contact Sheet 2: Focused region frames 450–750 (step 10)
    # -------------------------------------------------------------
    focused_indices = list(range(450, 751, 10))
    frames_sheet2 = []
    
    for f_idx in focused_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        vis = frame.copy()
        cv2.rectangle(vis, (10, 10), (220, 50), (0, 0, 0), -1)
        cv2.putText(vis, f"Frame {f_idx:04d} ({f_idx/fps:.1f}s)", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        resized = cv2.resize(vis, (256, 144), interpolation=cv2.INTER_AREA)
        frames_sheet2.append(resized)
        
    cols2 = 5
    rows2 = int(np.ceil(len(frames_sheet2) / cols2))
    grid2 = np.zeros((rows2 * 144, cols2 * 256, 3), dtype=np.uint8)
    
    for idx, f_img in enumerate(frames_sheet2):
        r = idx // cols2
        c = idx % cols2
        grid2[r * 144:(r + 1) * 144, c * 256:(c + 1) * 256] = f_img
        
    sheet2_path = out_dir / "contact_sheet_focused_frames_450_750.jpg"
    cv2.imwrite(str(sheet2_path), grid2)
    print(f"Focused contact sheet saved: {sheet2_path}")
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("CONTACT SHEETS GENERATED:")
    print("=" * 60)
    print(f"1. Full-Video Contact Sheet: {sheet1_path}")
    print(f"   Covered: Frame {full_indices[0]} – Frame {full_indices[-1]} ({len(frames_sheet1)} frames)")
    print(f"2. Focused Region Contact Sheet: {sheet2_path}")
    print(f"   Covered: Frame {focused_indices[0]} – Frame {focused_indices[-1]} ({len(frames_sheet2)} frames, step=10)")

if __name__ == "__main__":
    create_contact_sheets()
