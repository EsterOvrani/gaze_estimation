import copy

import cv2
import mediapipe as mp
import numpy as np
from mediapipeArrays import VERTISES
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:\\Users\\Esti\\PycharmProjects\\Gaze_estimation-master\\face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE, output_face_blendshapes=True)

face_mesh = FaceLandmarker.create_from_options(options)

# Assume very little distortion
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Example 3D model points of the face (You need to define this according to your 3D model)
model_points = VERTISES

camera_matrix = None

def load_image(image_path):
    return cv2.imread(image_path)

def get_face_keypoints(image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = face_mesh.detect(mp_image)
    if not results.face_landmarks:
        return None, None
    H, W, _ = image.shape

    print(results.face_landmarks[0][0].x)
    return (np.array([(landmark.x, landmark.y) for landmark in results.face_landmarks[0]]) * np.array([W, H]).reshape(1,2), results.face_blendshapes)
def estimate_pose(image_points):
    image_points = np.array(image_points, dtype=np.float32)
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points[:468], camera_matrix, dist_coeffs)
    return rotation_vector, translation_vector

def project_points(rotation_vector, translation_vector):
    # Project the 3D points back onto the image plane
    projected_points, _ = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    return projected_points

def c_matrix(image_width, image_height):
    global camera_matrix
    # Assuming fx and fy are half the image dimensions
    fx = fy = max(image_width, image_height) / 2

    # Optical center is typically at the image center
    cx = image_width / 2
    cy = image_height / 2

    # Constructing the camera matrix
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
def rotation_vector_to_euler_angles(rvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Calculate Euler angles from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])  # Convert radians to degrees

def draw_look_vector(image, nose_point, rotation_vector):
    # Assuming the rotation vector is in radians and is (pitch, yaw, roll)
    # Convert rotation vector to degrees for easier trigonometry
    pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector)
    # Calculate the end point of the vector
    length = 100  # Length of the look vector
    x_end = int(nose_point[0] + length * -np.sin(np.radians(yaw)))
    y_end = int(nose_point[1] - length * np.sin(np.radians(pitch)))

    # Convert points to integer tuple
    start_point = (int(nose_point[0]), int(nose_point[1]))
    end_point = (x_end, y_end)

    # Draw the vector
    cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2)

def draw_look_vector1(image, nose_point, rotation_vector):
    # Assuming the rotation vector is in radians and is (pitch, yaw, roll)
    # Convert rotation vector to degrees for easier trigonometry
    pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector)
    #pitch, yaw, roll = rotation_vector * (180 / np.pi)
    # Calculate the end point of the vector
    length = 100  # Length of the look vector
    x_end = int(nose_point[0] + length * -np.sin(np.radians(yaw)))
    y_end = int(nose_point[1] - length * np.sin(np.radians(pitch)))
    z_end = int(nose_point[2] + length * np.sin(np.radians(roll)))


    # Convert points to integer tuple
    start_point = (int(nose_point[0]), int(nose_point[1]))
    end_point = (x_end, y_end)

    # Draw the vector
    cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2)
    return [int(nose_point[0]) - x_end, int(nose_point[1]) - y_end, int(nose_point[2]) - z_end]

def simulation(t_vec, r_vec):

    # PLANE 1 #################################################################################################

    point = np.array([0, 0, 0]) # change from 0,0,1
    vector1 = np.array([0, 1, 0]) # change from 0,1,0
    vector2 = np.array([1, 0, 0]) # change from 1,0,0
    # create the size of the Coordinates
    s, t = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    # canonical face modle plane
    plane1 = point + s[:, :, np.newaxis] * vector1 + t[:, :, np.newaxis] * vector2

    # # canonical face modle points
    # points = VERTISES
    # # Extracting x, y, z coordinates from the points
    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]

    # face_diraction = np.array([0, 0, -1])

    # PLANE 2 #################################################################################################

    # create rotation object
    rotation = R.from_rotvec(r_vec.flatten())
    # rotate the vectors ant translate
    v1_rotated = rotation.apply(vector1)
    v2_rotated = rotation.apply(vector2)
    point_rotated = rotation.apply(point - t_vec.flatten())  # TODO: to check if it's fine to mines the t_vec instead of to add t_vec.
    # image plane
    plane2 = point_rotated + s[:, :, np.newaxis] * v1_rotated + t[:, :, np.newaxis] * v2_rotated

    # PLANE 3 ##################################################################################################

    # rotate the plane to the original plane
    rotation_inv = rotation.inv()
    v1_inverse = rotation_inv.apply(v1_rotated)
    v2_inverse = rotation_inv.apply(v2_rotated)
    point_inverse = rotation_inv.apply(point_rotated) + t_vec.flatten()
    # inverse plane
    plane3 = point_inverse + s[:, :, np.newaxis] * v1_inverse + t[:, :, np.newaxis] * v2_inverse

    # PLANE 4 - LIKE PLANE 2 BESIDE #############################################################################

    # canonical face modle points
    points = copy.deepcopy(VERTISES)
    for i in range(points.__len__()):
        points[i] = rotation.apply(points[i] - t_vec.flatten())

    # Extracting x, y, z coordinates from the points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    face_diraction = rotation.apply(np.array([0, 0, -1]))

    #  SOLVE - FIND THE CUT POINT ################################################################################

    # when the camera rotate
    # a = np.vstack([v1_rotated, v2_rotated, face_diraction*-1]).T # TODO: if i need to enter the point that's start face_diraction
    # b = point_rotated * -1
    # p = np.linalg.solve(a, b)

    # when the face rotate
    a = np.vstack([v1_inverse, v2_inverse, face_diraction*-1]).T # TODO: if i need to enter the point that's start face_diraction
    b = point_inverse * -1 + point_rotated
    p = np.linalg.solve(a, b)

    print("p: ", p)

    # CORNERS #####################################################################################################
    # corners of the 4 vectors of canonical face plan
    corner1 = np.array([-10,-10,0])
    corner2 = np.array([10,10,0])
    # corners of the 4 vectors of image plan
    corner1_rotated = rotation.apply(corner1 - t_vec.flatten())
    corner2_rotated = rotation.apply(corner2 - t_vec.flatten())

    # DISPLAY ####################################################################################################
    # create the Coordinate system
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # add the planes to the Coordinate system
    ax.plot_surface(plane1[:, :, 0], plane1[:, :, 1], plane1[:, :, 2], alpha=0.5, rstride=1, cstride=1, color='blue')
    ax.plot_surface(plane2[:, :, 0], plane2[:, :, 1], plane2[:, :, 2], alpha=0.5, rstride=1, cstride=1, color='red')
    ax.plot_surface(plane3[:, :, 0], plane3[:, :, 1], plane3[:, :, 2], alpha=0.5, rstride=1, cstride=1, color='green')

    # add the points of the canonical face model
    ax.scatter(x, y, z, color='red', s=1)

    # middle point plane1, plane2, plane3
    ax.scatter(point_rotated[0], point_rotated[1], point_rotated[2], color='green', s=1)
    ax.scatter(point[0], point[1], point[2], color='green', s=1)

    # vectors : v1, v2, v1_rotate, v2_rotate, v1_inverse, v2_inverse
    ax.plot([point[0], point[0] + vector1[0]*10], [point[1], point[1] + vector1[1] * 10], [point[2], point[2] + vector1[2] * 10],
            label='Line between points', marker='o')
    ax.plot([point[0], point[0] + vector2[0]* 10], [point[1],  point[1] + vector2[1]* 10], [point[2],  point[2] + vector2[2] * 10],
            label='Line between points', marker='o')

    ax.plot([point_rotated[0], point_rotated[0] + v1_rotated[0]*10], [point_rotated[1], point_rotated[1] + v1_rotated[1] * 10], [point_rotated[2], point_rotated[2] + v1_rotated[2]* 10],
            label='Line between points', marker='o')
    ax.plot([point_rotated[0], point_rotated[0] + v2_rotated[0]* 10], [point_rotated[1],  point_rotated[1] + v2_rotated[1]* 10], [point_rotated[2],  point_rotated[2] + v2_rotated[2]* 10],
            label='Line between points', marker='o')

    ax.plot([point_inverse[0], point_inverse[0] + v1_inverse[0]*10], [point_inverse[1], point_inverse[1] + v1_inverse[1] * 10], [point_inverse[2], point_inverse[2] + v1_inverse[2] * 10],
            label='Line between points', marker='o')
    ax.plot([point_inverse[0], point_inverse[0] + v2_inverse[0]* 10], [point_inverse[1],  point_inverse[1] + v2_inverse[1]* 10], [point_inverse[2],  point_inverse[2] + v2_inverse[2] * 10],
            label='Line between points', marker='o')

    # display the corners of image plan
    ax.scatter(corner1_rotated[0], corner1_rotated[1], corner1_rotated[2], color='black', s=1)
    ax.scatter(corner2_rotated[0], corner2_rotated[1], corner2_rotated[2], color='blue', s=1)

    # display the vectors - border of image plan
    ax.plot([corner1_rotated[0], corner1_rotated[0] + v1_rotated[0]*20], [corner1_rotated[1], corner1_rotated[1] + v1_rotated[1] * 20], [corner1_rotated[2], corner1_rotated[2] + v1_rotated[2]* 20],
            label='Line between points', marker='o')
    ax.plot([corner1_rotated[0], corner1_rotated[0] + v2_rotated[0]* 20], [corner1_rotated[1],  corner1_rotated[1] + v2_rotated[1]* 20], [corner1_rotated[2],  corner1_rotated[2] + v2_rotated[2]* 20],
            label='Line between points', marker='o')
    ax.plot([corner2_rotated[0], corner2_rotated[0] + v1_rotated[0]*-20], [corner2_rotated[1], corner2_rotated[1] + v1_rotated[1] * -20], [corner2_rotated[2], corner2_rotated[2] + v1_rotated[2]* -20],
            label='Line between points', marker='o')
    ax.plot([corner2_rotated[0], corner2_rotated[0] + v2_rotated[0]* -20], [corner2_rotated[1],  corner2_rotated[1] + v2_rotated[1]* -20], [corner2_rotated[2],  corner2_rotated[2] + v2_rotated[2]* -20],
           label='Line between points', marker='o')

    # display the diraction vector cut rotate plane - image plane
    # # TODO:1. plot line:  face direction from 0*face_diraction to 10*face_diraction
    # end_point = point_rotated + p[0]*v1_rotated + p[1]*v2_rotated # point_rotated * -1
    # ax.plot([point_rotated[0], end_point[0]], [point_rotated[1], end_point[1]], [point_rotated[2], end_point[2]],
    #        label='Line between points', marker='o')
    # print('equation diff: ', point_rotated + p[0]*v1_rotated + p[1]*v2_rotated - p[2]*face_diraction) # point_rotated * -1
    # # TODO:2. plot face direction until the plane 0*face_diraction to p[2]*face_direction
    # # TODO:3. check: p[0]*v1_rotated + p[1]*v2_rotated = p[2]*face_direction
    # ax.plot([point[0], p[2]*face_diraction[0]], [point[1], p[2]*face_diraction[1]], [point[2], p[2]*face_diraction[2]],
    #      label='Line between points', marker='o')

    # display the diraction vector cut inverse plane - image plane
    # TODO:1. plot line:  face direction from 0*face_diraction to 10*face_diraction
    end_point = point_inverse + p[0]*v1_inverse + p[1]*v2_inverse # point_rotated * -1
    ax.plot([point_inverse[0], end_point[0]], [point_inverse[1], end_point[1]], [point_inverse[2], end_point[2]],
           label='Line between points', marker='o')
    print('equation diff: ', point_inverse + p[0]*v1_inverse + p[1]*v2_inverse - p[2]*face_diraction) # point_rotated * -1
    # TODO:2. plot face direction until the plane 0*face_diraction to p[2]*face_direction
    # TODO:3. check: p[0]*v1_rotated + p[1]*v2_rotated = p[2]*face_direction
    ax.plot([point_rotated[0], p[2]*face_diraction[0]], [point_rotated[1], p[2]*face_diraction[1]], [point_rotated[2], p[2]*face_diraction[2]],
         label='Line between points', marker='o')

    # CHECK IF FACE_DIRACTION VECTOR IS PLUMB ##########################################################################################################
    print(np.dot(v1_rotated, face_diraction))
    print(np.dot(v1_rotated, face_diraction))
    print(np.dot(v1_inverse, face_diraction))
    print(np.dot(v1_inverse, face_diraction))

    # CHECK IF THE POINT OVER THE PLANE ##################################################################################################################
    # chack if the vector go up or down
    if p[0] < 0 or p[1] < 0:
        new_p = check_the_point(p, face_diraction * p[2], point_rotated, corner1_rotated, v2_rotated, v1_rotated)
    else:
        new_p = check_the_point(p, face_diraction * p[2], point_rotated, corner2_rotated, v1_rotated*-1, v2_rotated*-1)

    # display the close point on the image plan
    ax.scatter(new_p[0], new_p[1], new_p[2], color='black', s=3)


    # הגדרות צירים
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # הצגת הגרף
    plt.show()

def check_the_point(p, cut_point, point_rotated, c, v1, v2):
    # calc the vector from the נקודת מוצא to the cut point
    vec = np.array([cut_point[0] - point_rotated[0], cut_point[1] - point_rotated[1], cut_point[2] - point_rotated[2]])
    normalized_vec = vec / np.linalg.norm(vec)
    #find the cut point by the 2 vectors (v1, vec)
    A = np.vstack([v1, -normalized_vec]).T
    b = point_rotated - c
    x , residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # if it's more than ten its mean that it's not need to be this vector and we cheek the v2 vec (v2, vec)
    if x[0] > 11 or x[1] > 11:
        A = np.vstack([v2, -normalized_vec]).T
        b = point_rotated - c
        x , residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return (c + x[0] * v2 , point_rotated + x[1] * vec)[0] if (abs(p[0]) > abs(x[0]) or abs(p[1]) > abs(x[1])) else cut_point

    return (c + x[0] * v1, point_rotated + x[1] * vec)[0] if (abs(p[0]) > abs(x[0]) or abs(p[1]) > abs(x[1])) else cut_point

def gaze(t, r):
    vector1 = np.array([0.0, 20.0, 0.0])
    vector2 = np.array([20.0, 0.0, 0.0])

    rot = R.from_euler('xyz', r.flatten(), degrees=True)
    # rot = rot.as_matrix() # as rotation matrix

    # סיבוב של שני הוקטורים
    v1_rotated = rot.apply(vector1 + t.flatten())
    v2_rotated = rot.apply(vector2 + t.flatten())
    # t_rotated = rot.apply(t)

    # R*(v+t) = R*v + R*t

    # הוספת ההזזה לשני הוקטורים
    v1_transformed = v1_rotated
    v2_transformed = v2_rotated

    a = np.vstack([v1_transformed, v2_transformed, np.array([0, 0, -1])])  # TODO: check a.shape = 3x3
    b = rot.apply(np.array([0, 0, 0]) + t.flatten())
    x = np.linalg.solve(a, b)
    return x

def main(image_path):
    image = load_image(image_path)
    h, w = image.shape[:2]
    c_matrix(w,h)
    image_points, face_blendshapes = get_face_keypoints(image)

    if image_points is None:
        print("No face detected.")
        return




    if face_blendshapes[0][9].score >= 0.55 and face_blendshapes[0][10].score >= 0.55:
        print("close")
    elif face_blendshapes[0][9].score >= 0.55:
        print("left blink")
    elif face_blendshapes[0][10].score >= 0.55:
        print("right blink")
    else:
        print("open")

    for point in image_points:
        pt = (int(point[0]), int(point[1]))
        cv2.circle(image, pt, 2, (0, 0, 255), -1)

    rotation_vector, translation_vector = estimate_pose(image_points)
    projected_points = project_points(rotation_vector, translation_vector)

    euler_angles = rotation_vector_to_euler_angles(rotation_vector)
    # dir_vector = draw_look_vector(image, image_points[4], rotation_vector)

    simulation(translation_vector,rotation_vector)
    x = gaze(translation_vector, rotation_vector)


    #Draw the projected points on the image
    for point in projected_points:
        pt = (int(point[0][0]), int(point[0][1]))
        cv2.circle(image, pt, 2, (0, 255, 0), -1)





    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main("simulation/1.jpg")
    main("simulation/2.jpg")
    main("simulation/3.jpg")
    main("simulation/4.jpg")
    main("simulation/5.jpg")
    main("simulation/6.jpg")
    main("simulation/7.jpg")






