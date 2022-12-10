from Utils import open_masks
import numpy as np
from matplotlib import pyplot as plt
import cv2
# Calcualte the rotation of a polygon represented in the .roi file format
folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION"
mask_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION"

masks = open_masks(mask_folder)

def calculate_rotation(polygon):
    # Iterate through polygon dict.
    rotations = {}
    idx = 0
    for image_name, value in polygon.items():
        # Iterate through each polygon in the image
        if image_name not in rotations:
            rotations[image_name] = {}
        for key, value in value.items():
            idx += 1
            polygon = list(zip(value["x"],value["y"]))
            # Find centroid of polygon.
            centroid = np.mean(polygon, axis=0)

            angles = []
            for vertex in polygon:
                dx = vertex[0] - centroid[0]
                dy = vertex[1] - centroid[1]
                angle = np.arctan2(dy, dx)
                angles.append(angle)
            # print("Vertex angles:", angles)

            # Calculate the average angle to approximate the polygon's rotation
            rotation_raw = np.mean(angles)

            # Find rotated bounding box of polygon using numpy min area rectangle.
            rect = cv2.minAreaRect(np.array([polygon], dtype=np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            rotations[image_name]["rotation_raw"] = rotation_raw
            rotations[image_name]["rotation_box"] = rect[-1]
            rotations[image_name]["polygon"] = polygon
            rotations[image_name]["centroid"] = centroid
            
            # Four corners of the bounding box.
            rotations[image_name]["box"] = box
    return rotations

def plot_rotations(rotations):
    idx = 0
    for image_name, info in rotations.items():
        idx += 1
        print(image_name, info["rotation_raw"])
        # Plot rotations.
        plt.plot(idx, info["rotation_raw"], 'ro', color='b')
        plt.plot(idx, info["rotation_box"], 'ro', color='r')
    
    plt.title("Rotation of polygons")
    plt.xlabel("Polygon")
    plt.ylabel("Rotation")
    plt.legend(["Raw", "Box"])
    plt.show()

def plot_polygon(rotation_info):
    polygon = rotation_info["polygon"]
    # Draw line using rotation at centroid.
    centroid = rotation_info["centroid"]
    # For loop that loops through every angle and splits it with a line at the centroid. and finds the orientation with equal mass on both sides.
    # Find the angle that has the least difference between the two sides.
    angles = []
    right_areas = []
    left_areas = []
    area_differences = []
    # Calculate inertial axis of polygon using its moments.
    moments = cv2.moments(np.array([polygon], dtype=np.float32))
    # Find the centroid of the polygon.
    centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    # Find inertial axis of polygon.
    inertial_axis = np.arctan2(moments["mu11"], moments["mu20"] - moments["mu02"]) / 2
    # Plot inertial axis.
    plt.fill(*zip(*polygon), color='r')
    plt.plot([centroid[0], centroid[0] + 1000*np.cos(inertial_axis)], [centroid[1], centroid[1] + 1000*np.sin(inertial_axis)], color='g')
    plt.show()

    # for angle in np.arange(0, 2*np.pi, 0.1):
    #     angles.append(angle)
    #     # Find the points on each side of the line.

    #     # left_points = []
    #     # right_points = []
    #     # for point in polygon:
    #     #     dx = point[0] - centroid[0]
    #     #     dy = point[1] - centroid[1]
    #     #     point_angle = np.arctan2(dy, dx)
    #     #     if point_angle < angle:
    #     #         left_points.append(point)
    #     #     else:
    #     #         right_points.append(point)
        
    #     # Need to fix this code that find all the points on each side of the line.

    #     # Calculate the centroid of each side.
    #     left_centroid = np.mean(left_points, axis=0)
    #     right_centroid = np.mean(right_points, axis=0)
    #     # Find area of each side.
    #     if(len(left_points) == 0):
    #         left_area = 0
    #     else:
    #         left_area = cv2.contourArea(np.array([left_points], dtype=np.float32))
    #     if(len(right_points) == 0):
    #         right_area = 0
    #     else:
    #         right_area = cv2.contourArea(np.array([right_points], dtype=np.float32))
    #     right_areas.append(right_area)
    #     left_areas.append(left_area)
    #     # Find the difference between the two areas.
    #     area_difference = abs(left_area - right_area)
    #     area_differences.append(area_difference)

    #     # Plot current angle with a long line.
    #     plt.fill(*zip(*polygon), color='r')
    #     plt.plot([centroid[0], centroid[0] + 1000*np.cos(angle)], [centroid[1], centroid[1] + 1000*np.sin(angle)], color='g')
    #     plt.show()
    #     # Calculate the distance between the centroids.
    #     distance = np.linalg.norm(left_centroid - right_centroid)
    #     # print("Angle:", angle, "Distance:", distance)
    # Plot the area differences.
    # plt.plot(angles, area_differences, color='b')
    # plt.show()

    # Plot the areas.
    plt.plot(angles, left_areas, color='r')
    plt.plot(angles, right_areas, color='b')
    plt.show()

    # Plot centroid.
    plt.plot(*centroid, marker='o', color='g', ls='-')
    # Plot the polygon.
    plt.fill(*zip(*polygon), color='r')


    # plt.plot(*zip(*polygon), marker='o', color='r', ls='-')
    plt.title("Polygon")
    plt.show()


rotations = calculate_rotation(masks)
plot_rotations(rotations)
for image_name, info in rotations.items():
    plot_polygon(info)
