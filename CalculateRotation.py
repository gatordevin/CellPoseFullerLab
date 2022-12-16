from Utils import open_masks, get_gray_matter_section_polygons
import numpy as np
from matplotlib import pyplot as plt
from shapely.ops import split
import cv2
from shapely.geometry import Point, LineString, Polygon
from shapely import affinity
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
        # plt.plot(idx, info["rotation_raw"], 'ro', color='b')
        # plt.plot(idx, info["rotation_box"], 'ro', color='r')
    
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

    shapely_polygon = Polygon(polygon)
    convex_hull = shapely_polygon.convex_hull
    # Simplify convex hull to 4 edges.
    # convex_hull = convex_hull.simplify(20, preserve_topology=False)

    # Rotated bounding box from the shapely_polygon.
    # rect = cv2.minAreaRect(np.array([polygon], dtype=np.float32))
    # # Shapely box.
    # shapely_box = Polygon(rect[0])
    # plt.plot(shapely_box.exterior.xy[0], shapely_box.exterior.xy[1], color='g')


    # plt.plot(convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], color='r')
    # Find the 4th longest line in convex hull.
    lines = []
    for i in range(len(convex_hull.exterior.xy[0])-1):
        lines.append(LineString([Point(convex_hull.exterior.xy[0][i], convex_hull.exterior.xy[1][i]), Point(convex_hull.exterior.xy[0][i+1], convex_hull.exterior.xy[1][i+1])]))
    lines.sort(key=lambda x: x.length, reverse=True)
    if(len(lines) > 3):
        longest_line = lines[0]
    else:
        return

    # Get all lines longer than 100 pixels.
    lines = [line for line in lines if line.length > 150]
    # Find the two lines that hav the most similiar angle.
    smallest_angle_lines = []
    angle_diff = []
    for i in range(len(lines)):
        angle1 = np.arctan2(lines[i].coords[1][1] - lines[i].coords[0][1], lines[i].coords[1][0] - lines[i].coords[0][0])
        for j in range(i+1, len(lines)):
            # Check if the two lines share a point.
            if(lines[i].coords[0] == lines[j].coords[0] or lines[i].coords[0] == lines[j].coords[1] or lines[i].coords[1] == lines[j].coords[0] or lines[i].coords[1] == lines[j].coords[1]):
                continue
            angle2 = np.arctan2(lines[j].coords[1][1] - lines[j].coords[0][1], lines[j].coords[1][0] - lines[j].coords[0][0])
            # print(angle1, angle2)
            dff = abs(angle1 - angle2)
            # Minimize the difference between the two angles.
            if(dff > np.pi/2):
                dff = np.pi - dff
            # to degrees.
            dff = abs(np.degrees(dff))
            # print(dff)
            angle_diff.append(dff)
            # check if dff is the smallest in angle diff and if so add the two lines to the smallest_angle_lines.
            if(dff == min(angle_diff)):
                smallest_angle_lines = [lines[i], lines[j]]
            if(len(smallest_angle_lines) == 0):
                smallest_angle_lines = [lines[i], lines[j]]

    if(len(smallest_angle_lines) == 0):
        # Display shapely_polygon.
        print("Invalid polygon moving on to next polygon.")
        return

    # Get average angle of smallest angle lines.
    # print(smallest_angle_lines)
    angle1 = np.arctan2(smallest_angle_lines[0].coords[1][1] - smallest_angle_lines[0].coords[0][1], smallest_angle_lines[0].coords[1][0] - smallest_angle_lines[0].coords[0][0])
    angle2 = np.arctan2(smallest_angle_lines[1].coords[1][1] - smallest_angle_lines[1].coords[0][1], smallest_angle_lines[1].coords[1][0] - smallest_angle_lines[1].coords[0][0])
    angle = (angle1 + angle2)/2
    angle = np.degrees(angle)
    print("Average angle of smallest angle lines:", angle)

    # Find distance two smallest lines are apart.
    distance = smallest_angle_lines[0].distance(smallest_angle_lines[1])
    print("Distance between smallest angle lines:", distance)

    # Get line between midpoints of smallest angle lines.
    midpoint1 = smallest_angle_lines[0].interpolate(0.5, normalized=True)
    midpoint2 = smallest_angle_lines[1].interpolate(0.5, normalized=True)
    midpoint_line = LineString([midpoint1, midpoint2])
    # plt.plot(midpoint_line.coords.xy[0], midpoint_line.coords.xy[1], color='g')

    # Rotate shapely polygon around center of midpointline by the midpint line angle.
    rotated_shapely_polygon = affinity.rotate(shapely_polygon, angle, origin=midpoint_line.centroid)
    plt.plot(rotated_shapely_polygon.exterior.xy[0], rotated_shapely_polygon.exterior.xy[1], color='b')
    plt.plot(rotated_shapely_polygon.exterior.xy[0], rotated_shapely_polygon.exterior.xy[1], color='b')

    # Plot two lines one at the same angle and one perpendicular to the angle.
    line1 = affinity.rotate(midpoint_line, 90, "centroid")

    # Plot two other lines that are the same anlge but shifted by distance/6.
    # Translate line 1 along line 2 a distance of distance/6.
    xoff = distance/6 * np.cos(np.radians(angle))
    yoff = distance/6 * np.sin(np.radians(angle))
    line3 = affinity.translate(line1, xoff, yoff)
    line4 = affinity.translate(line1, -xoff, -yoff)
    # plt.plot(line3.coords.xy[0], line3.coords.xy[1], color='g')
    # plt.plot(line4.coords.xy[0], line4.coords.xy[1], color='g')


    # for line in smallest_angle_lines:
    #     plt.plot(line.coords.xy[0], line.coords.xy[1], color='g')

    # Rotate polygon by angle of longest line.

    # Plot horizontal and vertical line going through centroid.
    # horizontal_line = LineString([(centroid[0]-1000, centroid[1]), (centroid[0]+1000, centroid[1])])
    # vertical_line = LineString([(centroid[0], centroid[1]-1000), (centroid[0], centroid[1]+1000)])
    # plt.plot(horizontal_line.coords.xy[0], horizontal_line.coords.xy[1], color='g')
    # plt.plot(vertical_line.coords.xy[0], vertical_line.coords.xy[1], color='g')
    # plot centroid
    # plt.plot(centroid[0], centroid[1], 'ro')
    plt.show()



# rotations = calculate_rotation(masks)
# plot_rotations(rotations)
# for image_name, info in rotations.items():
#     plot_polygon(info)

for image_name, value in masks.items():
    # Iterate through each polygon in the image
    for key, value in value.items():
        polygon = list(zip(value["x"],value["y"]))
        get_gray_matter_section_polygons(polygon)
