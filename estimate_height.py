from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

import dlib
import settings
from tracker_utils import CentroidTracker, TrackedObject
from utils import (CircularBuffer, add_bounding_box, add_text,
                   configure_stream, get_3d_translation,
                   get_closest_point_distance, get_speed, preprocess_image,
                   process_detections, run_stream, secs_diff)


def main():
    # Get settings parameters
    expected_size = settings.EXPECTED_SIZE
    confidence_threshold = settings.CONFIDENCE_THRESHOLD
    in_scale_factor = settings.IN_SCALE_FACTOR
    mean_val = settings.MEAN_VAL
    focal_y = settings.FOCAL_Y
    x_res = settings.X_RES
    y_res = settings.Y_RES
    fps = settings.FPS
    max_distance = settings.MAX_DISTANCE
    prototxt_path = settings.PROTOTXT_PATH
    model_path = settings.MODEL_PATH
    class_names = settings.CLASS_NAMES
    searched_classes = settings.SEARCHED_CLASSES
    skip_frames = settings.SKIP_FRAMES
    max_disappeared = settings.MAX_DISAPPEARED
    max_object_distance = settings.MAX_OBJECT_DISTANCE
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output.mp4', fourcc, fps, (x_res, y_res))

    centroid_tracker = CentroidTracker(max_disappeared=max_disappeared,
                                       max_distance=max_object_distance)
    trackers = []
    heights = []
    trackable_objects = {}
    total_frames = 0

    # Read neural network model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    height_buffer = CircularBuffer(size=15)
    pipeline, rs_config = configure_stream(x_res, y_res, fps)

    # Run stream from realsense camera
    profile = run_stream(pipeline, rs_config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    depth_intrin = None

    try:
        while True:
            speed_str = ''
            frames = pipeline.wait_for_frames()
            # Align RGB and depth frames to match corresponding points
            aligned_frames = align.process(frames)
            # Get frames
            depth_frame = aligned_frames.get_depth_frame()
            # depth_frame2 = frames.get_depth_frame()
            color_frame = aligned_frames.first(rs.stream.color)
            if not depth_intrin:
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # Get color image
            color_image = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # Get depth image
            depth_image = np.asanyarray(depth_frame.get_data())
            if not depth_image.size or not color_image.size:
                continue

            rects = []
            if total_frames % skip_frames == 0:
                trackers = []
                heights = []
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                blob = preprocess_image(color_image, expected_size, in_scale_factor, mean_val)
                net.setInput(blob, "data")
                # Detect objects on RGB image
                detections = net.forward("detection_out")
                # Get label, position, distance, height and other objects information
                objects_parameters = process_detections(detections, depth_image, height_buffer,
                                                        depth_scale, confidence_threshold, x_res,
                                                        y_res, class_names, searched_classes,
                                                        max_distance, focal_y)
                # Put information on image
                for (label, idx, confidence, x_min, x_max, y_min, y_max, distance,
                     height) in objects_parameters:
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x_min, y_min, x_max, y_max)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    heights.append(height)
            else:
                for tracker in trackers:
                    tracker.update(rgb)
                    pos = tracker.get_position()
                    x_min = int(pos.left())
                    y_min = int(pos.top())
                    x_max = int(pos.right())
                    y_max = int(pos.bottom())
                    rects.append((x_min, y_min, x_max, y_max))

            objects, objects_heights = centroid_tracker.update(rects, heights)

            for (object_id, centroid), (height) in zip(objects.items(), objects_heights.values()):
                tracked_object = trackable_objects.get(object_id, None)
                actual_distance, closest_point = get_closest_point_distance(centroid, depth_frame)
                if tracked_object is None:
                    tracked_object = TrackedObject(object_id, 15, timestamp=datetime.now(),
                                                   depth=actual_distance,
                                                   closest_point=closest_point)
                else:
                    tracked_object.counter += 1
                    tracked_object.timestamps.append(datetime.now())
                    tracked_object.closest_points.append(closest_point)
                    tracked_object.depths.append(actual_distance)
                    recent_timestamps = tracked_object.timestamps.data
                    add_bounding_box(color_image, centroid[0] - 30, centroid[1] - 30,
                                     centroid[0] + 30, centroid[1] + 30, colors[4])
                    shift_vec = get_3d_translation(depth_intrin,
                                                   tracked_object.closest_points.data[0],
                                                   tracked_object.closest_points.data[1],
                                                   tracked_object.depths.data[0],
                                                   tracked_object.depths.data[1])
                    if shift_vec:
                        secs = secs_diff(*recent_timestamps)
                        speed_vec = get_speed(shift_vec, secs)
                        tracked_object.x_speed_buffer.append(speed_vec[0])
                        tracked_object.y_speed_buffer.append(speed_vec[1])
                        tracked_object.z_speed_buffer.append(speed_vec[2])
                        speed_x = tracked_object.x_speed_buffer.smooth_average
                        speed_y = tracked_object.y_speed_buffer.smooth_average
                        speed_z = tracked_object.z_speed_buffer.smooth_average
                        speed_str = "{:.2f}\n{:.2f}\n{:.2f} m/s".format(speed_x,
                                                                        speed_y,
                                                                        speed_z)
                        cv2.circle(color_image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        y0, dy = centroid[1], 20
                        # height_str = "Height: {:.1f} cm".format(height)
                        # cv2.putText(color_image, height_str, (centroid[0], centroid[1] - dy),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        for i, line in enumerate(speed_str.split('\n')):
                            y = y0 + i * dy
                            cv2.putText(color_image, line, (centroid[0], y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # out.write(color_image)
                trackable_objects[object_id] = tracked_object

            # Apply colormap to depth image
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                            cv2.COLORMAP_JET)
            # Display RGB and depth frames
            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", depth_image)
            cv2.waitKey(1)
            total_frames += 1

    finally:
        # Stop streaming
        pipeline.stop()
        # out.release()


if __name__ == "__main__":
    main()
