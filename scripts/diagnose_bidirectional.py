import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.vision.tracker import VehicleTracker
from backend.core.vision.video_processor import VideoProcessor
from backend.models.traffic_schemas import ApproachEnum, MovementStateEnum
from backend.config import settings

def main():
    video_path = settings.UPLOAD_DIR / "bidirectional.mp4"
    output_video_path = settings.ANNOTATED_DIR / "bidirectional_direction_test.mp4"
    report_path = settings.ANNOTATED_DIR / "bidirectional_direction_diagnostic.txt"

    print("==================================================")
    print("RUNNING BIDIRECTIONAL DIAGNOSTIC TEST")
    print(f"Input Video:  {video_path}")
    print(f"Output Video: {output_video_path}")
    print(f"Report:       {report_path}")
    print("==================================================")

    # Use current production VehicleTracker exactly as it is (ApproachEnum.NORTH default)
    tracker = VehicleTracker(approach=ApproachEnum.NORTH)
    processor = VideoProcessor(tracker=tracker)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Track diagnostics collection
    tracks_info = {}
    
    # Process video and annotate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    cap = cv2.VideoCapture(str(video_path))
    tracker.set_approach(approach=ApproachEnum.NORTH, fps=fps)
    tracker.reset()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        tracked_vehicles = tracker.track(frame, fps=fps)

        # Collect track stats
        for v in tracked_vehicles:
            tid = v.track_id
            if tid not in tracks_info:
                tracks_info[tid] = {
                    "track_id": tid,
                    "stable_class": v.class_name,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "frames_count": 0,
                    "first_center": v.center,
                    "last_center": v.center,
                    "final_state": v.direction,
                    "last_moving_direction": v.last_moving_direction,
                    "all_states_seen": set(),
                    "centers": []
                }
            
            t = tracks_info[tid]
            t["last_frame"] = frame_idx
            t["frames_count"] += 1
            t["last_center"] = v.center
            t["stable_class"] = v.class_name
            t["final_state"] = v.direction
            t["last_moving_direction"] = v.last_moving_direction
            t["all_states_seen"].add(v.direction)
            t["centers"].append(v.center)

        # Draw frame annotation matching production
        annotated = frame.copy()
        for v in tracked_vehicles:
            x1, y1, x2, y2 = [int(c) for c in v.xyxy]
            color = (245, 130, 49) if v.class_name == "car" else (60, 180, 75) if v.class_name == "motorcycle" else (230, 25, 75)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            dir_str = v.direction.value if hasattr(v.direction, 'value') else str(v.direction)
            label = f"#{v.track_id} {v.class_name.upper()} | {dir_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cx, cy = int(v.center[0]), int(v.center[1])
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        out.write(annotated)
        if frame_idx % 25 == 0 or frame_idx == total_frames:
            print(f"Processed frame {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)")

    cap.release()
    out.release()

    print("\nAnnotation complete. Generating diagnostic breakdown...")

    # Calculate displacements
    for tid, info in tracks_info.items():
        fc = info["first_center"]
        lc = info["last_center"]
        info["dx"] = lc[0] - fc[0]
        info["dy"] = lc[1] - fc[1]
        info["displacement"] = np.hypot(info["dx"], info["dy"])
        info["avg_x"] = np.mean([c[0] for c in info["centers"]])

    # Group counts
    incoming_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.INCOMING]
    outgoing_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.OUTGOING]
    stopped_incoming = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.STOPPED_INCOMING]
    stopped_outgoing = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.STOPPED_OUTGOING]
    parked_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.PARKED]
    unknown_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.UNKNOWN]

    # Analyze left side (x < 380) vs right side (x >= 380)
    left_side_tracks = [t for t in tracks_info.values() if t["avg_x"] < 380.0]
    right_side_tracks = [t for t in tracks_info.values() if t["avg_x"] >= 380.0]

    # Check upward vs downward moving tracks
    upward_moving = [t for t in tracks_info.values() if t["dy"] < -10.0 and t["displacement"] > 15.0]
    downward_moving = [t for t in tracks_info.values() if t["dy"] > 10.0 and t["displacement"] > 15.0]

    # False INCOMING classifications: tracks that moved clearly upward (away from junction) but got classified as INCOMING / STOPPED_INCOMING
    false_incoming = [t for t in upward_moving if t["final_state"] in (MovementStateEnum.INCOMING, MovementStateEnum.STOPPED_INCOMING) or t["last_moving_direction"] == MovementStateEnum.INCOMING]
    # False OUTGOING classifications: tracks that moved clearly downward (toward junction) but got classified as OUTGOING / STOPPED_OUTGOING
    false_outgoing = [t for t in downward_moving if t["final_state"] in (MovementStateEnum.OUTGOING, MovementStateEnum.STOPPED_OUTGOING) or t["last_moving_direction"] == MovementStateEnum.OUTGOING]

    report = []
    report.append("====================================================================================================")
    report.append("                   BIDIRECTIONAL CCTV DIRECTIONAL DIAGNOSTIC REPORT                                 ")
    report.append("====================================================================================================")
    report.append(f"Input Video:                {video_path}")
    report.append(f"Annotated Output Video:     {output_video_path}")
    report.append(f"Active Tracker Approach:    {tracker.approach.value if hasattr(tracker.approach, 'value') else tracker.approach}")
    report.append(f"Active Junction Vector:     {tracker.junction_vector.tolist()} (Normalized [dx, dy] toward junction)")
    report.append(f"Active Tracker ROI:         {tracker.roi}")
    report.append(f"Frames Processed:           {total_frames} frames ({fps:.2f} fps)")
    report.append(f"Total Unique Track IDs:     {len(tracks_info)}")
    report.append("====================================================================================================\n")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("1. DIRECTIONAL STATE BREAKDOWN")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f" • Total Unique Tracks:     {len(tracks_info)}")
    report.append(f" • INCOMING:                {len(incoming_tracks)}")
    report.append(f" • OUTGOING:                {len(outgoing_tracks)}")
    report.append(f" • STOPPED_INCOMING:        {len(stopped_incoming)}")
    report.append(f" • STOPPED_OUTGOING:        {len(stopped_outgoing)}")
    report.append(f" • PARKED:                  {len(parked_tracks)}")
    report.append(f" • UNKNOWN:                 {len(unknown_tracks)}")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("2. ROADWAY SPATIAL & TRAJECTORY ANALYSIS (LEFT VS RIGHT LANES)")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f" • Left Side Lanes  (avg_x < 380px, Northbound / Moving Away from Camera): {len(left_side_tracks)} tracks")
    report.append(f" • Right Side Lanes (avg_x >= 380px, Southbound / Moving Toward Camera):   {len(right_side_tracks)} tracks")
    report.append(f" • Net Upward Motion   (dy < -10px, moving away towards horizon):          {len(upward_moving)} tracks")
    report.append(f" • Net Downward Motion (dy > +10px, moving towards camera foreground):      {len(downward_moving)} tracks")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("3. MISCLASSIFICATION ANALYSIS")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f" • Obvious False INCOMING (Upward dy < -10px classified as INCOMING):      {len(false_incoming)}")
    report.append(f" • Obvious False OUTGOING (Downward dy > +10px classified as OUTGOING):    {len(false_outgoing)}")
    report.append("")
    if false_incoming:
        report.append("Details of False INCOMING tracks:")
        for t in false_incoming:
            report.append(f"   Track #{t['track_id']} ({t['stable_class']}): avg_x={t['avg_x']:.1f}px, dy={t['dy']:.1f}px (f{t['first_frame']}->f{t['last_frame']}) | Classified: {t['final_state'].value}")
    if false_outgoing:
        report.append("Details of False OUTGOING tracks:")
        for t in false_outgoing:
            report.append(f"   Track #{t['track_id']} ({t['stable_class']}): avg_x={t['avg_x']:.1f}px, dy={t['dy']:.1f}px (f{t['first_frame']}->f{t['last_frame']}) | Classified: {t['final_state'].value}")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("4. COMPLETE TRACK-BY-TRACK INVENTORY")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f"{'ID':<5} | {'Class':<11} | {'Span':<11} | {'First Center':<15} | {'Last Center':<15} | {'Displacement (dx, dy)':<24} | {'State':<18} | {'Last Moving'}")
    report.append("-" * 120)
    for tid, t in sorted(tracks_info.items()):
        span_str = f"f{t['first_frame']:03d}-f{t['last_frame']:03d}"
        fc_str = f"({t['first_center'][0]:.1f}, {t['first_center'][1]:.1f})"
        lc_str = f"({t['last_center'][0]:.1f}, {t['last_center'][1]:.1f})"
        disp_str = f"dx={t['dx']:+6.1f}, dy={t['dy']:+6.1f}"
        state_str = t['final_state'].value if hasattr(t['final_state'], 'value') else str(t['final_state'])
        last_dir_str = t['last_moving_direction'].value if (t['last_moving_direction'] and hasattr(t['last_moving_direction'], 'value')) else str(t['last_moving_direction'])
        report.append(f"#{t['track_id']:<4} | {t['stable_class']:<11} | {span_str:<11} | {fc_str:<15} | {lc_str:<15} | {disp_str:<24} | {state_str:<18} | {last_dir_str}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\nDiagnostic report saved to: {report_path}")
    print(f"Annotated video saved to:   {output_video_path}")

if __name__ == "__main__":
    main()
