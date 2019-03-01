from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs


class CircularBuffer(deque):
    def __init__(self, size=0):
        super(CircularBuffer, self).__init__(maxlen=size)

    @property
    def average(self):
        return sum(self) / len(self)

    @property
    def median(self):
        return np.median(self)

    @property
    def data(self):
        return list(self)

    @property
    def smooth_average(self):
        data = self.reject_outliers(self)
        if data.size:
            return np.mean(data)
        return np.mean(self)

    @staticmethod
    def reject_outliers(data, m=2.):
        data = np.asarray(data)
        d = np.abs(data - np.median(data))
        stdev = np.std(d)
        s = d / (stdev if stdev else 1.)
        return data[s < m]


def configure_stream(x_res, y_res, fps):
    """
    Configure depth and color streams.
    """
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, x_res, y_res, rs.format.z16, fps)
    rs_config.enable_stream(rs.stream.color, x_res, y_res, rs.format.bgr8, fps)
    return pipeline, rs_config


def run_stream(pipeline, config):
    profile = pipeline.start(config)
    return profile


def add_bounding_box(image, x_min, y_min, x_max, y_max, color):
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)


def add_text(image, x_min, y_min, color, label):
    """
    Add text to image.
    """
    y = y_min - 15 if y_min - 15 > 15 else y_min + 15
    cv2.putText(image, label, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)


def get_distance_to_object(depth_image, x_min, y_min, x_max, y_max, depth_scale):
    """
    Use more points to calculate reliable distance from person.
    """
    px_width = x_max - x_min
    px_height = y_max - y_min
    x_min = int(x_min + px_width / 4)
    x_max = int(x_max - px_width / 4)
    y_min = int(y_min + px_height / 3)
    y_max = int(y_max - px_height / 6)
    depth = depth_image[y_min:y_max, x_min:x_max].astype(float)
    depth = depth * depth_scale
    depth_slice = depth[(depth < np.quantile(depth, 0.7)) & (depth > np.quantile(depth, 0.3))]
    if depth_slice.size:
        distance = np.mean(depth_slice)
        distance *= 100  # from m to cm
        return distance
    return None


def preprocess_image(image, expected_size, in_scale_factor, mean_val):
    """
    Create blob from image as an input to neural network.
    """
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (expected_size, expected_size)),
                                 in_scale_factor, (expected_size, expected_size), mean_val)
    return blob


def calculate_height(distance, y_max, y_min, focal_y):
    """
    Calculate real person height in centimeters.
    """
    px_height = y_max - y_min
    person_height = distance * px_height / focal_y
    return person_height


def process_detections(detections, depth_image, height_buffer, depth_scale, confidence_threshold,
                       x_res, y_res, class_names, searched_classes, max_distance, focal_y):
    """
    Get information about all objects.
    :return:
    """
    detected_objects_parameters = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7]
            box[box < 0] = 0
            box = box * np.array([x_res, y_res, x_res, y_res])
            (x_min, y_min, x_max, y_max) = box.astype("int")
            label = class_names[idx]

            if label in searched_classes:
                distance = get_distance_to_object(depth_image, x_min, y_min, x_max, y_max,
                                                  depth_scale)
                if not distance or distance > max_distance:
                    continue
                person_height = calculate_height(distance, y_max, y_min, focal_y)
                height_buffer.append(person_height)
                person_averaged_height = height_buffer.median
                detected_objects_parameters.append([label, idx, confidence, x_min, x_max, y_min,
                                                    y_max, distance, person_averaged_height])
    return detected_objects_parameters


def get_3d_point(intr, point, distance):
    depth_point = rs.rs2_deproject_pixel_to_point(intr, [point[0], point[1]], distance)
    return depth_point


def get_3d_translation(intr, point1, point2, previous_distance, actual_distance):
    dist1 = previous_distance
    dist2 = actual_distance
    if dist1 == 0 or dist2 == 0:
        return None
    depth_point1 = get_3d_point(intr, point1, dist1)
    depth_point2 = get_3d_point(intr, point2, dist2)
    x_translation = depth_point2[0] - depth_point1[0]
    y_translation = depth_point2[1] - depth_point1[1]
    z_translation = depth_point2[2] - depth_point1[2]
    if x_translation == 0 and y_translation == 0 and z_translation == 0:
        return None
    return (x_translation, y_translation, z_translation)


def get_closest_point_distance(point, depth_frame):
    """Get closest distance to object near given point."""
    min_distance = depth_frame.get_distance(point[0], point[1])
    closest_point = None
    for i in range(-30, 30, 10):
        for j in range(-30, 30, 10):
            distance = depth_frame.get_distance(point[0] + i, point[1] + j)
            if (distance < min_distance and distance > 0) or distance == 0:
                min_distance = distance
                closest_point = (point[0] + i, point[1] + j)
    if closest_point:
        return min_distance, closest_point
    return min_distance, point


def get_speed(shift_vec, seconds):
    if seconds > 0.0:
        return [shift / seconds for shift in shift_vec]
    else:
        return [0, 0, 0]


def secs_diff(start_time, end_time):
    diff = (end_time - start_time).total_seconds()
    return diff
