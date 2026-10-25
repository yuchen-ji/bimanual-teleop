"""Shared live circle positions and ordering on a display copy."""

import cv2
import numpy as np


def detection_overlay(rgb, centers):
    """Annotate a display copy; the sensor image is never modified."""
    display = rgb.copy()
    count = 0 if centers is None else len(centers)
    if count:
        points = np.round(centers).astype(np.int32)
        cv2.polylines(display, [points.reshape(-1, 1, 2)], False, (40, 200, 255), 1, cv2.LINE_AA)
        for index, pixel in enumerate(points):
            color = (255, 70, 40) if index == 0 else (40, 255, 80)
            cv2.circle(display, tuple(pixel), 6 if index == 0 else 4, color, 2, cv2.LINE_AA)
            label = str(index)
            (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)
            x = max(0, min(int(pixel[0]) + 8, display.shape[1] - width - 1))
            y = max(height + 1, min(int(pixel[1]) - 8, display.shape[0] - 2))
            cv2.putText(display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
    label = f"{count}/20 circles | order 0-19" if count else "Board not detected (0/20)"
    cv2.putText(display, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(display, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (40, 255, 80) if count else (255, 180, 40), 1, cv2.LINE_AA)
    return display
