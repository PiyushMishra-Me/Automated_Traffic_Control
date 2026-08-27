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
    output_video_path = settings.ANNOTATED_DIR / "bidirectional_full_roi_test.mp4"
    report_path = settings.ANNOTATED_DIR / "bidirectional_full_roi_diagnostic.txt"

    print("==================================================")
    print("RUNNING BIDIRECTIONAL FULL-FRAME ROI TEST")
    print(f"Input:  {video_path}")
    print(f"Output: {output_video_path}")
    print(f"Report: {report_path}")
    print("==================================================")

    # Use FULL-FRAME ROI (roi=None)
    tracker = VehicleTracker(roi=None, approach=ApproachEnum.NORTH, junction_vector=[0.0, 1.0])
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    cap = cv2.VideoCapture(str(video_path))
    tracker.set_approach(approach=ApproachEnum.NORTH, junction_vector=[0.0, 1.0], fps=fps, roi=None)
    tracker.reset()

    tracks_info = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        tracked_vehicles = tracker.track(frame, fps=fps)

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

        # Draw frame annotation matching requirement:
        # #TRACK_ID CLASS | INCOMING / OUTGOING / STOPPED_INCOMING / STOPPED_OUTGOING / UNKNOWN
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

    print("\nAnnotation complete. Generating full diagnostic breakdown...")

    # Calculate displacements
    for tid, info in tracks_info.items():
        fc = info["first_center"]
        lc = info["last_center"]
        info["dx"] = lc[0] - fc[0]
        info["dy"] = lc[1] - fc[1]
        info["displacement"] = np.hypot(info["dx"], info["dy"])
        info["avg_x"] = np.mean([c[0] for c in info["centers"]])
        info["avg_y"] = np.mean([c[1] for c in info["centers"]])

    # Group counts
    incoming_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.INCOMING]
    outgoing_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.OUTGOING]
    stopped_incoming = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.STOPPED_INCOMING]
    stopped_outgoing = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.STOPPED_OUTGOING]
    parked_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.PARKED]
    unknown_tracks = [t for t in tracks_info.values() if t["final_state"] == MovementStateEnum.UNKNOWN]

    # Analyze UNKNOWN tracks reasons
    # Reasons:
    # 1. too few trajectory points (< 3 frames)
    # 2. insufficient displacement (< 2.0 px)
    # 3. brief track / detection lost (3 to 5 frames)
    # 4. horizon/distant vehicle (avg_y < 80px)
    # 5. other / stationary queue with no prior direction
    unknown_reasons = {
        "too_few_trajectory_points": 0,
        "insufficient_displacement": 0,
        "brief_track_lost": 0,
        "horizon_small_vehicle": 0,
        "stationary_uninitialized": 0
    }
    for t in unknown_tracks:
        if t["frames_count"] < 3:
            unknown_reasons["too_few_trajectory_points"] += 1
            t["unknown_reason"] = "Too few trajectory points (< 3 frames)"
        elif t["displacement"] < 2.0:
            unknown_reasons["insufficient_displacement"] += 1
            t["unknown_reason"] = "Insufficient displacement (< 2.0px)"
        elif t["frames_count"] <= 5:
            unknown_reasons["brief_track_lost"] += 1
            t["unknown_reason"] = "Brief track lost (< 5 frames)"
        elif t["avg_y"] < 80.0:
            unknown_reasons["horizon_small_vehicle"] += 1
            t["unknown_reason"] = "Horizon distant vehicle (y < 80px)"
        else:
            unknown_reasons["stationary_uninitialized"] += 1
            t["unknown_reason"] = "Stationary queue with no prior movement history"

    # Select representative 10 Left Carriageway and 10 Right Carriageway tracks
    left_side = sorted([t for t in tracks_info.values() if t["avg_x"] < 380.0 and t["frames_count"] >= 10], key=lambda x: -x["frames_count"])
    right_side = sorted([t for t in tracks_info.values() if t["avg_x"] >= 380.0 and t["frames_count"] >= 10], key=lambda x: -x["frames_count"])

    rep_left = left_side[:10]
    rep_right = right_side[:10]

    # Check for mismatches
    upward_moving = [t for t in tracks_info.values() if t["dy"] < -10.0 and t["displacement"] > 15.0]
    downward_moving = [t for t in tracks_info.values() if t["dy"] > 10.0 and t["displacement"] > 15.0]
    false_incoming = [t for t in upward_moving if t["final_state"] in (MovementStateEnum.INCOMING, MovementStateEnum.STOPPED_INCOMING) or t["last_moving_direction"] == MovementStateEnum.INCOMING]
    false_outgoing = [t for t in downward_moving if t["final_state"] in (MovementStateEnum.OUTGOING, MovementStateEnum.STOPPED_OUTGOING) or t["last_moving_direction"] == MovementStateEnum.OUTGOING]

    report = []
    report.append("====================================================================================================")
    report.append("                   BIDIRECTIONAL CCTV FULL-FRAME ROI DIAGNOSTIC REPORT                              ")
    report.append("====================================================================================================")
    report.append(f"Input Video:                {video_path}")
    report.append(f"Annotated Output Video:     {output_video_path}")
    report.append(f"Active Tracker Approach:    {tracker.approach.value if hasattr(tracker.approach, 'value') else tracker.approach}")
    report.append(f"Active Junction Vector:     {tracker.junction_vector.tolist()} (Normalized [dx, dy] toward junction)")
    report.append(f"Active Tracker ROI:         FULL FRAME [0, 0, {width}, {height}] (None)")
    report.append(f"Frames Processed:           {total_frames} frames ({fps:.2f} fps)")
    report.append(f"Total Unique Track IDs:     {len(tracks_info)}")
    report.append("====================================================================================================\n")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("1. DIRECTIONAL STATE SUMMARY")
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
    report.append("2. COMPARISON: FULL-FRAME RUN vs PREVIOUS CROPPED ROI RUN")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f"{'Metric':<25} | {'Previous (Cropped ROI)':<24} | {'New (Full-Frame ROI)':<24} | {'Difference / Reason'}")
    report.append("-" * 105)
    report.append(f"{'Total Unique Tracks':<25} | {'255':<24} | {len(tracks_info):<24} | +{len(tracks_info)-255} (Captured entire left outgoing carriageway x=0..220)")
    report.append(f"{'INCOMING (Active)':<25} | {'14':<24} | {len(incoming_tracks):<24} | Active Southbound vehicles moving towards camera")
    report.append(f"{'OUTGOING (Active)':<25} | {'31':<24} | {len(outgoing_tracks):<24} | +{len(outgoing_tracks)-31} (Newly detected Northbound vehicles in left lanes)")
    report.append(f"{'STOPPED_INCOMING':<25} | {'42':<24} | {len(stopped_incoming):<24} | Queued traffic on Right Carriageway")
    report.append(f"{'STOPPED_OUTGOING':<25} | {'26':<24} | {len(stopped_outgoing):<24} | +{len(stopped_outgoing)-26} (Queued/slow traffic on Left Carriageway)")
    report.append(f"{'UNKNOWN':<25} | {'142':<24} | {len(unknown_tracks):<24} | Horizon & brief transient detections across entire frame width")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("3. DETAILED ROOT-CAUSE BREAKDOWN OF ALL UNKNOWN TRACKS")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f" • Too Few Trajectory Points (< 3 frames):            {unknown_reasons['too_few_trajectory_points']} tracks")
    report.append(f" • Insufficient Displacement (< 2.0px total motion):  {unknown_reasons['insufficient_displacement']} tracks")
    report.append(f" • Brief Track Lost (3 to 5 frames lifespan):         {unknown_reasons['brief_track_lost']} tracks")
    report.append(f" • Horizon Distant Vehicles (avg_y < 80px):           {unknown_reasons['horizon_small_vehicle']} tracks")
    report.append(f" • Stationary Queue (uninitialized movement history): {unknown_reasons['stationary_uninitialized']} tracks")
    report.append(f" • Total UNKNOWN tracks:                              {len(unknown_tracks)} tracks")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("4. REPRESENTATIVE 10 TRACKS FROM LEFT CARRIAGEWAY (NORTHBOUND / OUTGOING)")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f"{'Track ID':<9} | {'Class':<11} | {'Span':<11} | {'First Center':<15} | {'Last Center':<15} | {'Displacement':<22} | {'State':<18} | {'Physical Movement'}")
    report.append("-" * 125)
    for t in rep_left:
        span_str = f"f{t['first_frame']:03d}-f{t['last_frame']:03d}"
        fc_str = f"({t['first_center'][0]:.1f}, {t['first_center'][1]:.1f})"
        lc_str = f"({t['last_center'][0]:.1f}, {t['last_center'][1]:.1f})"
        disp_str = f"dx={t['dx']:+6.1f}, dy={t['dy']:+6.1f}"
        state_str = t['final_state'].value if hasattr(t['final_state'], 'value') else str(t['final_state'])
        phys = "UPWARD / AWAY (Expected: OUTGOING)" if t['dy'] <= 0 else "DOWNWARD / TOWARD"
        report.append(f"#{t['track_id']:<8} | {t['stable_class']:<11} | {span_str:<11} | {fc_str:<15} | {lc_str:<15} | {disp_str:<22} | {state_str:<18} | {phys}")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("5. REPRESENTATIVE 10 TRACKS FROM RIGHT CARRIAGEWAY (SOUTHBOUND / INCOMING)")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f"{'Track ID':<9} | {'Class':<11} | {'Span':<11} | {'First Center':<15} | {'Last Center':<15} | {'Displacement':<22} | {'State':<18} | {'Physical Movement'}")
    report.append("-" * 125)
    for t in rep_right:
        span_str = f"f{t['first_frame']:03d}-f{t['last_frame']:03d}"
        fc_str = f"({t['first_center'][0]:.1f}, {t['first_center'][1]:.1f})"
        lc_str = f"({t['last_center'][0]:.1f}, {t['last_center'][1]:.1f})"
        disp_str = f"dx={t['dx']:+6.1f}, dy={t['dy']:+6.1f}"
        state_str = t['final_state'].value if hasattr(t['final_state'], 'value') else str(t['final_state'])
        phys = "DOWNWARD / TOWARD (Expected: INCOMING)" if t['dy'] >= 0 else "UPWARD / AWAY"
        report.append(f"#{t['track_id']:<8} | {t['stable_class']:<11} | {span_str:<11} | {fc_str:<15} | {lc_str:<15} | {disp_str:<22} | {state_str:<18} | {phys}")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("6. MISCLASSIFICATION ANALYSIS")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f" • Obvious False INCOMING: {len(false_incoming)}")
    report.append(f" • Obvious False OUTGOING: {len(false_outgoing)}")
    report.append("")

    report.append("----------------------------------------------------------------------------------------------------")
    report.append("7. COMPLETE TRACK-BY-TRACK INVENTORY (ALL UNIQUE TRACKS)")
    report.append("----------------------------------------------------------------------------------------------------")
    report.append(f"{'ID':<5} | {'Class':<11} | {'Span':<11} | {'Frames':<6} | {'First Center':<15} | {'Last Center':<15} | {'Displacement (dx, dy)':<24} | {'Direction':<18} | {'State'}")
    report.append("-" * 125)
    for tid, t in sorted(tracks_info.items()):
        span_str = f"f{t['first_frame']:03d}-f{t['last_frame']:03d}"
        fc_str = f"({t['first_center'][0]:.1f}, {t['first_center'][1]:.1f})"
        lc_str = f"({t['last_center'][0]:.1f}, {t['last_center'][1]:.1f})"
        disp_str = f"dx={t['dx']:+6.1f}, dy={t['dy']:+6.1f}"
        state_str = t['final_state'].value if hasattr(t['final_state'], 'value') else str(t['final_state'])
        last_dir_str = t['last_moving_direction'].value if (t['last_moving_direction'] and hasattr(t['last_moving_direction'], 'value')) else str(t['last_moving_direction'])
        report.append(f"#{t['track_id']:<4} | {t['stable_class']:<11} | {span_str:<11} | {t['frames_count']:<6} | {fc_str:<15} | {lc_str:<15} | {disp_str:<24} | {last_dir_str:<18} | {state_str}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\nDiagnostic report saved to: {report_path}")
    print(f"Annotated video saved to:   {output_video_path}")

if __name__ == "__main__":
    main()
