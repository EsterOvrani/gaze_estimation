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
    face_diraction = np.array([0, 0, 1])
    # נקודת מוצא
    point = np.array([0, 0, 1])
    # וקטורים על המישור
    vector1 = np.array([0, 1, 0])
    vector2 = np.array([1, 0, 0])

    # create the size of the Coordinates
    s, t = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    # canonical face modle plane
    plane1 = point + s[:, :, np.newaxis] * vector1 + t[:, :, np.newaxis] * vector2

    # create rotation object
    rotation = R.from_rotvec(r_vec.flatten())
    # rotate the vectors ant translate
    v1_rotated = rotation.apply(vector1)
    v2_rotated = rotation.apply(vector2)
    point_rotated = rotation.apply(point - t_vec.flatten())  # TODO: to check if it's fine to mines the t_vec instead of to add t_vec.

    # image plane
    plane2 = point_rotated + s[:, :, np.newaxis] * v1_rotated + t[:, :, np.newaxis] * v2_rotated

    # canonical face modle points
    points = VERTISES

    # Extracting x, y, z coordinates from the points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # create the Coordinate system
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # add the planes to the Coordinate system
    ax.plot_surface(plane1[:, :, 0], plane1[:, :, 1], plane1[:, :, 2], alpha=0.5, rstride=1, cstride=1, color='blue')
    ax.plot_surface(plane2[:, :, 0], plane2[:, :, 1], plane2[:, :, 2], alpha=0.5, rstride=1, cstride=1, color='red')

    # add the points of the canonical face model
    ax.scatter(x, y, z, color='red', s=1)
    ax.scatter(point_rotated[0], point_rotated[1], point_rotated[2], color='green', s=1)

    #find the cut point
    a = np.vstack([v1_rotated, v2_rotated, np.array([0, 0, -1])])  # TODO: check a.shape = 3x3
    b = rotation.apply(np.array([0, 0, 0]) + t_vec.flatten())
    p = np.linalg.solve(a, b)

    ax.plot([VERTISES[1][0], p[0]], [VERTISES[1][1],  p[1]], [VERTISES[1][2],  p[2]],
            label='Line between points', marker='o')

    # הגדרות צירים
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # הצגת הגרף
    plt.show()

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






