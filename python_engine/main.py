"""
AI-Powered Real-Time Object Detection & Tracking System
Main CLI Video Processing Pipeline
"""

import sys
import os
import argparse
import logging
import cv2

from python_engine.config import VisionConfig
from python_engine.detector import YOLODetector
from python_engine.tracker import ByteTracker
from python_engine.analytics import AnalyticsManager
from python_engine.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MainPipeline")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI-Powered Real-Time Object Detection & Tracking System"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Input video source: webcam index (e.g. '0'), RTSP URL, or video file path"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="YOLO model checkpoint (yolov8n.pt, yolov8s.pt, yolov8m.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.40,
        help="Detection confidence threshold [0.0 - 1.0]"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold [0.0 - 1.0]"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Inference device: 'auto', 'cuda:0', 'mps', or 'cpu'"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save processed output video with visual overlays"
    )
    parser.add_argument(
        "--output", type=str, default="output_tracked.mp4",
        help="Path for saved video file"
    )
    parser.add_argument(
        "--no-line", action="store_true",
        help="Disable virtual line-crossing detection"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Initialize Configuration
    config = VisionConfig(
        model_name=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        enable_line_crossing=not args.no_line,
        save_output_video=args.save,
        output_video_path=args.output
    )

    # 2. Open Video Stream (Webcam or File)
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)
        logger.info(f"Connecting to live webcam device index {source_val}...")
    else:
        if not os.path.exists(source_val) and not source_val.startswith("rtsp://") and not source_val.startswith("http"):
            logger.error(f"Specified input video file does not exist: {source_val}")
            sys.exit(1)
        logger.info(f"Opening video file source: {source_val}")

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        logger.error(f"Failed to open video source '{args.source}'. Please verify camera permissions or file integrity.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"Input Stream Info: {width}x{height} @ {fps:.1f} FPS")

    # Adapt tripwire line to resolution
    config.line_start = (int(width * 0.05), int(height * 0.55))
    config.line_end = (int(width * 0.95), int(height * 0.55))

    # 3. Video Writer if saving requested
    video_writer = None
    if config.save_output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(config.output_video_path, fourcc, fps, (width, height))
        logger.info(f"Recording processed video to {config.output_video_path}")

    # 4. Initialize Core AI & Tracking Components
    logger.info("Initializing YOLO Detector, ByteTracker, Analytics, and Visualizer...")
    detector = YOLODetector(config)
    tracker = ByteTracker(config)
    analytics = AnalyticsManager(config)
    visualizer = Visualizer(config)

    logger.info("=" * 60)
    logger.info("LIVE TRACKING STARTED. Controls:")
    logger.info(" [q / ESC] : Quit application")
    logger.info(" [p / SPACE]: Pause / Resume playback")
    logger.info(" [s]       : Capture high-resolution screenshot")
    logger.info(" [r]       : Reset tracker IDs and crossing counters")
    logger.info("=" * 60)

    paused = False
    screenshot_idx = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream reached.")
                    break

                # Step 1: Detect objects
                detections, inference_ms = detector.detect(frame)

                # Step 2: Track objects with ByteTrack
                tracks = tracker.update(detections)

                # Step 3: Analytics, counts & line crossing
                analytics.process_frame(tracks, inference_ms)

                # Step 4: Visual rendering
                vis_frame = visualizer.render(frame, tracks, analytics)

                # Step 5: Video saving
                if video_writer is not None:
                    video_writer.write(vis_frame)
            else:
                # When paused, keep displaying same frame
                vis_frame = frame.copy()
                cv2.putText(vis_frame, "[PAUSED] Press 'p' to Resume", (int(width / 2) - 180, int(height / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)

            # Display window
            cv2.imshow("AI Real-Time Object Detection & Tracking", vis_frame)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in [ord('q'), 27]:  # 'q' or ESC
                break
            elif key in [ord('p'), 32]:  # 'p' or SPACE
                paused = not paused
                logger.info("Playback Paused" if paused else "Playback Resumed")
            elif key == ord('s'):
                screenshot_idx += 1
                fname = f"screenshot_{screenshot_idx}.png"
                cv2.imwrite(fname, vis_frame)
                logger.info(f"Saved screenshot: {fname}")
            elif key == ord('r'):
                tracker.reset()
                analytics.reset()
                logger.info("Tracker IDs and analytics counters reset.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        # Save detection logs to CSV
        analytics.export_csv(config.log_csv_path)
        logger.info(f"Exported detection logs to {config.log_csv_path}")

        summary = analytics.get_summary()
        logger.info("-" * 40)
        logger.info("FINAL SESSION SUMMARY:")
        logger.info(f" Total Unique Objects Tracked: {summary['total_unique']}")
        logger.info(f" Virtual Line Crossing IN:     {summary['count_in']}")
        logger.info(f" Virtual Line Crossing OUT:    {summary['count_out']}")
        logger.info(f" Average Processing FPS:       {summary['current_fps']}")
        logger.info("-" * 40)


if __name__ == "__main__":
    main()
