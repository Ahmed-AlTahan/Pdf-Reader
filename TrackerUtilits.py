import cv2
def create_tracker(tracker_name):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_name
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker