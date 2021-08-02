# SleepyDriver

SleepyDriver uses OpenCV image processing to detect whether or not a driver is paying attention and allows the car to respond appropriately.

Current Functionality includes:
- If the driver is looking at something in the distance, their pupils will be in an irregular quadrant of the eye. Selective areas of the eye are determined as attentive regions.
- If the driver is drifting asleep, their eyelids will render the pupils undetectable, which is another case of unattentiveness.
- If the driver is looking at their phone or at the backseats, their face will be undetectable; another case of unattentiveness.

Packages Used:
- Ros2 (for additional functionality on AWS Deepracer Robot)
- OpenCV

COSMOS UCSD 2021 - Cluster 11 - Group 8
