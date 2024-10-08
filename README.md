# **Traffic Management System with YOLO**

This repository contains a project developed as part of the **Smart India Hackathon 2024**, by the team **DATA PIRATES**. The project aims to implement real-time traffic management using vehicle detection and traffic density analysis across a four-lane cross section. We utilize **YOLO (You Only Look Once)** for vehicle detection and **OpenCV** for image processing, displaying real-time traffic data to optimize traffic flow.

## **Project Overview**

In urban areas, traffic congestion is a significant issue. This project uses machine learning and computer vision techniques to detect vehicles on multiple lanes, calculate traffic density, and provide insights for optimizing traffic signals.

### **Features:**
- Real-time vehicle detection using YOLOv3.
- Traffic density calculation for four lanes.
- Graphical representation of traffic data for better understanding.
- Scalable solution for multiple cross-sections in a city.
- Provides data on average traffic density, helping to optimize signal times dynamically.

### **Technology Stack:**
- **YOLOv3**: Used for object (vehicle) detection.
- **OpenCV**: For real-time video processing.
- **NumPy & Pandas**: For data handling and calculations.
- **Matplotlib**: To visualize traffic density on different lanes.

## **Installation**

### **Prerequisites:**
- Python 3.x
- `numpy`, `pandas`, `opencv-python`, `matplotlib`, `torch` (for YOLO)

### **Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hmshuv/Prototype_SIH_2024.git
   cd Prototype_SIH_2024
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the YOLOv3 weights:**
   Download the `yolov3.weights` file from [this link](https://pjreddie.com/media/files/yolov3.weights) and place it in the root directory of the project.

4. **Run the program:**
   ```bash
   python main.py
   ```

## **Usage**

The program processes video frames to detect vehicles and calculate the traffic density on four lanes. The output includes:
- Number of vehicles per lane.
- Real-time graphical representation of traffic density.
  
## **Screenshots**

Here are some example screenshots of the output generated by the system:

### **Vehicle Detection on Lanes**
<img width="1470" alt="Vehicle_Detection" src="https://github.com/user-attachments/assets/d436aed0-6739-4152-afbd-f374105dd117">




### **Traffic Density Graph**
![Taffic_Density_detection](https://github.com/user-attachments/assets/9ef8358c-e472-41b0-b9f2-7cf30b51dfca)


---

To add your screenshots:

1. Create a `screenshots` folder in the repository root.
2. Place your images (e.g., `vehicle_detection.png`, `traffic_density.png`) inside the `screenshots` folder.
3. Ensure the image paths in the README file match the location of the images in the folder.

## **Contributing**

If you wish to contribute, feel free to fork the repository and submit a pull request.

---

Let me know if you need more adjustments or further sections added to this!
