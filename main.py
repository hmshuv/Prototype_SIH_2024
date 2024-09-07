import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import os

# Load YOLO v3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()

if len(unconnected_out_layers.shape) > 1:
    unconnected_out_layers = unconnected_out_layers.flatten()

output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load YOLO classes (coco names)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Unit assignments for different vehicles
vehicle_units = {
    'bicycle': 1, 'car': 2, 'motorbike': 1, 'bus': 4,
    'truck': 4, 'fire truck': 100
}

# Parameters for traffic density calculation
lane_length_km = 1.0  # Adjust this as needed

# Initialize video capture for four lanes
cap1 = cv2.VideoCapture("lane1.mp4")
cap2 = cv2.VideoCapture("lane2.mp4")
cap3 = cv2.VideoCapture("lane3.mp4")
cap4 = cv2.VideoCapture("lane4.mp4")

# Ensure videos are opened successfully
if not cap1.isOpened():
    print("Error: Could not open video lane1_video.mp4")
if not cap2.isOpened():
    print("Error: Could not open video lane2_video.mp4")
if not cap3.isOpened():
    print("Error: Could not open video lane3_video.mp4")
if not cap4.isOpened():
    print("Error: Could not open video lane4_video.mp4")

# Initialize previous positions for tracking direction
previous_positions = {
    'lane1': {},
    'lane2': {},
    'lane3': {},
    'lane4': {}
}

def track_direction(previous_positions, current_positions, lane_name):
    """Determine if vehicles are incoming (moving towards the camera)."""
    incoming_vehicles = {}
    
    for vehicle_id, current_pos in current_positions.items():
        prev_pos = previous_positions.get(vehicle_id)
        if prev_pos:
            # Calculate movement direction: if y-coordinate decreases, vehicle is moving towards the camera
            if current_pos[1] < prev_pos[1]:
                incoming_vehicles[vehicle_id] = current_pos
    
    # Update previous positions
    previous_positions.update(current_positions)
    return incoming_vehicles

# Function to process each lane's video feed
def process_lane(cap, lane_name):
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    unit_count = 0
    counts = {key: 0 for key in vehicle_units.keys()}  # Reset counts
    current_positions = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                current_positions[len(current_positions)] = (center_x, center_y)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Track incoming vehicles based on direction
    incoming_vehicles = track_direction(previous_positions[lane_name], current_positions, lane_name)

    # Count vehicles that are incoming only
    if len(indices) > 0:
        for i in indices.flatten():
            class_name = classes[class_ids[i]]
            vehicle_id = i  # Use the index 'i' as vehicle_id
            if class_name in vehicle_units and vehicle_id in incoming_vehicles:
                unit_count += vehicle_units[class_name]
                counts[class_name] += 1
                # Draw bounding box (optional)
                x, y, w, h = boxes[i]
                label = f"{class_name} {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calculate traffic density
    traffic_density = unit_count / lane_length_km
    return counts, traffic_density, frame

# Start processing for all four lanes
frame_count = 0
while cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened():
    lane_data1 = []
    lane_data2 = []
    lane_data3 = []
    lane_data4 = []

    counts1, traffic_density1, frame1 = process_lane(cap1, "lane1")
    counts2, traffic_density2, frame2 = process_lane(cap2, "lane2")
    counts3, traffic_density3, frame3 = process_lane(cap3, "lane3")
    counts4, traffic_density4, frame4 = process_lane(cap4, "lane4")

    # If any of the videos finish, break the loop
    if counts1 is None or counts2 is None or counts3 is None or counts4 is None:
        break

    # Store data for each lane separately
    lane_data1.append({
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': traffic_density1
    })
    lane_data2.append({
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': traffic_density2
    })
    lane_data3.append({
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': traffic_density3
    })
    lane_data4.append({
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': traffic_density4
    })

    # Create individual DataFrames for each lane
    df_data1 = {
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': []
    }
    df_data2 = {
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': []
    }
    df_data3 = {
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': []
    }
    df_data4 = {
        'Vehicle Type': [],
        'Count': [],
        'Traffic Density (units/km)': []
    }

    for vehicle_type, count in counts1.items():
        df_data1['Vehicle Type'].append(vehicle_type)
        df_data1['Count'].append(count)
        df_data1['Traffic Density (units/km)'].append(traffic_density1)

    for vehicle_type, count in counts2.items():
        df_data2['Vehicle Type'].append(vehicle_type)
        df_data2['Count'].append(count)
        df_data2['Traffic Density (units/km)'].append(traffic_density2)

    for vehicle_type, count in counts3.items():
        df_data3['Vehicle Type'].append(vehicle_type)
        df_data3['Count'].append(count)
        df_data3['Traffic Density (units/km)'].append(traffic_density3)

    for vehicle_type, count in counts4.items():
        df_data4['Vehicle Type'].append(vehicle_type)
        df_data4['Count'].append(count)
        df_data4['Traffic Density (units/km)'].append(traffic_density4)

    df1 = pd.DataFrame(df_data1)
    df2 = pd.DataFrame(df_data2)
    df3 = pd.DataFrame(df_data3)
    df4 = pd.DataFrame(df_data4)

    # Clear console and print updated data for each lane separately
    os.system('clear' if os.name == 'posix' else 'cls')  # Clear console for Unix or Windows
    print(f"Frame: {frame_count}")
    print("\nLane 1 Data:")
    print(tabulate(df1, headers='keys', tablefmt='grid'))
    print("\nLane 2 Data:")
    print(tabulate(df2, headers='keys', tablefmt='grid'))
    print("\nLane 3 Data:")
    print(tabulate(df3, headers='keys', tablefmt='grid'))
    print("\nLane 4 Data:")
    print(tabulate(df4, headers='keys', tablefmt='grid'))

    # Create a dictionary for traffic densities
    lane_densities = {
        'Lane 1': traffic_density1,
        'Lane 2': traffic_density2,
        'Lane 3': traffic_density3,
        'Lane 4': traffic_density4
    }

    # Sort lanes by traffic density in descending order (highest priority first)
    sorted_lanes = sorted(lane_densities.items(), key=lambda item: item[1], reverse=True)

    # Create a DataFrame for the priority table
    df_priority = pd.DataFrame(sorted_lanes, columns=["Lane", "Traffic Density (units/km)"])

    # Print the priority table
    print("\nPriority Table:")
    print(tabulate(df_priority, headers='keys', tablefmt='grid'))

    frame_count += 1

    # Display each lane's frame (optional, for visual feedback)
    cv2.imshow("Lane 1", frame1)
    cv2.imshow("Lane 2", frame2)
    cv2.imshow("Lane 3", frame3)
    cv2.imshow("Lane 4", frame4)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture objects and close all windows
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
