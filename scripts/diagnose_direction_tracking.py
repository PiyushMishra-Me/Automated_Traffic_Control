import sys
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.vision.tracker import VehicleTracker
from backend.models.traffic_schemas import ApproachEnum, MovementStateEnum
from backend.config import settings

def main():
    video_path = settings.UPLOAD_DIR / "my_traffic.mp4"
    report_path = settings.ANNOTATED_DIR / "my_traffic_direction_diagnostic.txt"

    print("==================================================")
    print("RUNNING DIRECTIONAL STATE DIAGNOSTIC")
    print(f"Video:  {video_path}")
    print(f"Report: {report_path}")
    print("==================================================")

    # Initialize tracker with default settings and ApproachEnum.NORTH
    tracker = VehicleTracker(roi=settings.DETECTION_ROI, approach=ApproachEnum.NORTH)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Track diagnostics data
    # track_id -> dict
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
                    "all_centers": []
                }
            
            t = tracks_info[tid]
            t["last_frame"] = frame_idx
            t["frames_count"] += 1
            t["last_center"] = v.center
            t["stable_class"] = v.class_name
            t["final_state"] = v.direction
            t["last_moving_direction"] = v.last_moving_direction
            t["all_states_seen"].add(v.direction)
            t["all_centers"].append(v.center)

    cap.release()

    print(f"Processed {frame_idx} frames. Total unique tracks observed: {len(tracks_info)}")

    # Analyze trajectories
    incoming_tracks = []
    outgoing_tracks = []
    stopped_incoming_tracks = []
    stopped_outgoing_tracks = []
    unknown_tracks = []
    parked_tracks = []

    mismatched_tracks = []

    for tid, info in sorted(tracks_info.items()):
        fc = info["first_center"]
        lc = info["last_center"]
        dx = lc[0] - fc[0]
        dy = lc[1] - fc[1]
        info["dx"] = dx
        info["dy"] = dy
        info["displacement"] = np.hypot(dx, dy)

        # Categorize by final state (or if ever active state)
        fstate = info["final_state"]
        if fstate == MovementStateEnum.INCOMING:
            incoming_tracks.append(info)
        elif fstate == MovementStateEnum.OUTGOING:
            outgoing_tracks.append(info)
        elif fstate == MovementStateEnum.STOPPED_INCOMING:
            stopped_incoming_tracks.append(info)
        elif fstate == MovementStateEnum.STOPPED_OUTGOING:
            stopped_outgoing_tracks.append(info)
        elif fstate == MovementStateEnum.PARKED:
            parked_tracks.append(info)
        else:
            unknown_tracks.append(info)

        # Check for trajectory vs classification mismatch:
        # Junction vector for NORTH is [0, 1] (moving downward = positive dy = toward junction = INCOMING)
        # If a track clearly moved UPWARD (dy < -10px) with significant displacement (> 15px),
        # but was classified as INCOMING or STOPPED_INCOMING:
        if dy < -10.0 and info["displacement"] > 15.0 and (fstate in (MovementStateEnum.INCOMING, MovementStateEnum.STOPPED_INCOMING) or info["last_moving_direction"] == MovementStateEnum.INCOMING):
            mismatched_tracks.append((info, "Moved UPWARD (dy < -10) but classified as INCOMING"))
        # If a track clearly moved DOWNWARD (dy > 10px) but was classified as OUTGOING:
        elif dy > 10.0 and info["displacement"] > 15.0 and (fstate in (MovementStateEnum.OUTGOING, MovementStateEnum.STOPPED_OUTGOING) or info["last_moving_direction"] == MovementStateEnum.OUTGOING):
            mismatched_tracks.append((info, "Moved DOWNWARD (dy > 10) but classified as OUTGOING"))

    # Also check if any track ever had OUTGOING in its lifetime
    ever_outgoing = [info for info in tracks_info.values() if MovementStateEnum.OUTGOING in info["all_states_seen"] or MovementStateEnum.STOPPED_OUTGOING in info["all_states_seen"]]
    ever_incoming = [info for info in tracks_info.values() if MovementStateEnum.INCOMING in info["all_states_seen"] or MovementStateEnum.STOPPED_INCOMING in info["all_states_seen"]]

    # Let's inspect video lane physical direction:
    # Look at all 164 tracks dy distribution:
    positive_dy_tracks = [info for info in tracks_info.values() if info["dy"] > 5.0 and info["displacement"] > 10.0]
    negative_dy_tracks = [info for info in tracks_info.values() if info["dy"] < -5.0 and info["displacement"] > 10.0]
    stationary_tracks = [info for info in tracks_info.values() if info["displacement"] <= 10.0]

    report_lines = []
    report_lines.append("====================================================================================================")
    report_lines.append("                         VEHICLE TRACKER DIRECTIONAL DIAGNOSTIC REPORT                              ")
    report_lines.append("====================================================================================================")
    report_lines.append(f"Input Video:                  {video_path}")
    report_lines.append(f"Configured Approach:          {tracker.approach.value if hasattr(tracker.approach, 'value') else tracker.approach}")
    report_lines.append(f"Configured Junction Vector:   {tracker.junction_vector.tolist()} (Normalized [dx, dy] toward junction)")
    report_lines.append(f"Video Frame Count:            {total_frames} frames ({fps:.2f} fps)")
    report_lines.append(f"Total Unique Track IDs:       {len(tracks_info)}")
    report_lines.append("====================================================================================================\n")

    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append("1. STATE BREAKDOWN SUMMARY (FINAL STATE OF ALL 164 TRACKS)")
    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append(f" • INCOMING:                  {len(incoming_tracks)}")
    report_lines.append(f" • OUTGOING:                  {len(outgoing_tracks)}")
    report_lines.append(f" • STOPPED_INCOMING:          {len(stopped_incoming_tracks)}")
    report_lines.append(f" • STOPPED_OUTGOING:          {len(stopped_outgoing_tracks)}")
    report_lines.append(f" • PARKED:                    {len(parked_tracks)}")
    report_lines.append(f" • UNKNOWN:                   {len(unknown_tracks)}")
    report_lines.append(f" • Ever Classified OUTGOING:  {len(ever_outgoing)}")
    report_lines.append(f" • Ever Classified INCOMING:  {len(ever_incoming)}\n")

    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append("2. PHYSICAL TRAJECTORY DISPLACEMENT (dx, dy) ANALYSIS")
    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append(f" • Net Downward Motion (dy > +5px, toward bottom/junction): {len(positive_dy_tracks)} tracks")
    report_lines.append(f" • Net Upward Motion   (dy < -5px, toward top/away):        {len(negative_dy_tracks)} tracks")
    report_lines.append(f" • Stationary/Jitter   (total displacement <= 10px):        {len(stationary_tracks)} tracks\n")

    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append("3. ALL OUTGOING TRACKS REPORT (Lifetime or Final OUTGOING)")
    report_lines.append("----------------------------------------------------------------------------------------------------")
    if not ever_outgoing:
        report_lines.append(" [NONE] No tracks were classified as OUTGOING or STOPPED_OUTGOING at any point in the video.")
    else:
        for t in ever_outgoing:
            report_lines.append(f" Track #{t['track_id']:<3} | Class: {t['stable_class']:<10} | Frames: f{t['first_frame']:03d}->f{t['last_frame']:03d} ({t['frames_count']:<3}f) | First: ({t['first_center'][0]:.1f}, {t['first_center'][1]:.1f}) -> Last: ({t['last_center'][0]:.1f}, {t['last_center'][1]:.1f}) | Displacement: dx={t['dx']:+.1f}, dy={t['dy']:+.1f} | Final State: {t['final_state'].value}")
    report_lines.append("")

    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append("4. UPWARD-MOVING TRACKS (dy < -5.0px) INSPECTION")
    report_lines.append("----------------------------------------------------------------------------------------------------")
    if not negative_dy_tracks:
        report_lines.append(" [NONE] No tracks exhibited net upward displacement (dy < -5px).")
    else:
        for t in negative_dy_tracks:
            report_lines.append(f" Track #{t['track_id']:<3} | Class: {t['stable_class']:<10} | Frames: f{t['first_frame']:03d}->f{t['last_frame']:03d} ({t['frames_count']:<3}f) | First: ({t['first_center'][0]:.1f}, {t['first_center'][1]:.1f}) -> Last: ({t['last_center'][0]:.1f}, {t['last_center'][1]:.1f}) | Displacement: dx={t['dx']:+.1f}, dy={t['dy']:+.1f} | State: {t['final_state'].value} | Last Moving: {t['last_moving_direction']}")
    report_lines.append("")

    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append("5. MISMATCH / OPPOSITE-DIRECTION ANOMALIES INSPECTION")
    report_lines.append("----------------------------------------------------------------------------------------------------")
    if not mismatched_tracks:
        report_lines.append(" [NONE] Zero mismatched tracks detected. Every classified movement corresponds with geometric motion.")
    else:
        for t, reason in mismatched_tracks:
            report_lines.append(f" Track #{t['track_id']} ({t['stable_class']}): {reason} | dx={t['dx']:+.1f}, dy={t['dy']:+.1f}")
    report_lines.append("")

    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append("6. COMPLETE TRACK-BY-TRACK TABLE FOR ALL 164 UNIQUE TRACKS")
    report_lines.append("----------------------------------------------------------------------------------------------------")
    report_lines.append(f"{'ID':<5} | {'Class':<11} | {'Span':<11} | {'Frames':<6} | {'First Center':<15} | {'Last Center':<15} | {'Displacement (dx, dy)':<24} | {'Movement State':<18} | {'Last Moving'}")
    report_lines.append("-" * 125)
    for tid, t in sorted(tracks_info.items()):
        span_str = f"f{t['first_frame']:03d}-f{t['last_frame']:03d}"
        fc_str = f"({t['first_center'][0]:.1f}, {t['first_center'][1]:.1f})"
        lc_str = f"({t['last_center'][0]:.1f}, {t['last_center'][1]:.1f})"
        disp_str = f"dx={t['dx']:+6.1f}, dy={t['dy']:+6.1f}"
        state_str = t['final_state'].value if hasattr(t['final_state'], 'value') else str(t['final_state'])
        last_dir_str = t['last_moving_direction'].value if (t['last_moving_direction'] and hasattr(t['last_moving_direction'], 'value')) else str(t['last_moving_direction'])
        report_lines.append(f"#{t['track_id']:<4} | {t['stable_class']:<11} | {span_str:<11} | {t['frames_count']:<6} | {fc_str:<15} | {lc_str:<15} | {disp_str:<24} | {state_str:<18} | {last_dir_str}")

    report_content = "\n".join(report_lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nDiagnostic report successfully written to: {report_path}")
    print("\nSummary:")
    print(f"Total Unique Tracks:          {len(tracks_info)}")
    print(f"INCOMING (Final):             {len(incoming_tracks)}")
    print(f"OUTGOING (Final):             {len(outgoing_tracks)}")
    print(f"STOPPED_INCOMING (Final):     {len(stopped_incoming_tracks)}")
    print(f"STOPPED_OUTGOING (Final):     {len(stopped_outgoing_tracks)}")
    print(f"UNKNOWN (Final):              {len(unknown_tracks)}")
    print(f"Ever Classified OUTGOING:     {len(ever_outgoing)}")
    print(f"Ever Classified INCOMING:     {len(ever_incoming)}")
    print(f"Tracks with Net dy > +5px:    {len(positive_dy_tracks)}")
    print(f"Tracks with Net dy < -5px:    {len(negative_dy_tracks)}")

if __name__ == "__main__":
    main()
