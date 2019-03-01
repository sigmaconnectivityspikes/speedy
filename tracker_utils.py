from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist

from utils import CircularBuffer


class TrackedObject():
    def __init__(self, object_id, speed_bufer_length, timestamp, depth, closest_point, label=0,
                 idx=0, confidence=0, x_min=0, x_max=0, y_min=0, y_max=0, height=0):
        self.object_id = object_id
        self.depths = CircularBuffer(2)
        self.closest_points = CircularBuffer(2)
        self.timestamps = CircularBuffer(2)  # keeps actual and previous object detection time
        self.x_speed_buffer = CircularBuffer(speed_bufer_length)
        self.y_speed_buffer = CircularBuffer(speed_bufer_length)
        self.z_speed_buffer = CircularBuffer(speed_bufer_length)
        self.counter = 0
        self.timestamps.append(timestamp)
        self.depths.append(depth)
        self.closest_points.append(closest_point)
    #  add depth information
    # self.label = label
    # self.idx = idx
    # self.confidence = confidence
    # self.x_min = x_min
    # self.x_max = x_max
    # self.y_min = y_min
    # self.y_max = y_max
    # self.distance = distance
    # self.height = height


class CentroidTracker():
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.heights = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, height):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.heights[self.next_object_id] = height
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects, heights):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.heights

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x_min, y_min, x_max, y_max)) in enumerate(rects):
            c_x = int((x_min + x_max) / 2.0)
            c_y = int((y_min + y_max) / 2.0)
            input_centroids[i] = (c_x, c_y)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], heights[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the distance between each pair of object
            distance = dist.cdist(np.array(object_centroids), input_centroids)

            # sort by distances
            rows = distance.min(axis=1).argsort()
            cols = distance.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distance[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, distance.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance.shape[1])).difference(used_cols)

            if distance.shape[0] >= distance.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], heights[col])

        # return the set of trackable objects
        return self.objects, self.heights
