import math

import cv2
import numpy as np


def euler_to_RotMatrix(roll, pitch, yaw):
    """
    roll, pitch and yow in radians
    in order to change degree to radian you can use math.radians(degree)
    """
    su = math.sin(roll)
    cu = math.cos(roll)
    sv = math.sin(pitch)
    cv = math.cos(pitch)
    sw = math.sin(yaw)
    cw = math.cos(yaw)

    A =np.zeros((3,3),dtype=np.float64)
    A[0,0] = cv * cw
    A[0,1] = su * sv * cw - cu * sw
    A[0,2] = su * sw + cu * sv * cw
    A[1,0] = cv * sw
    A[1,1] = cu * cw + su * sv * sw
    A[1,2] = cu * sv * sw - su * cw
    A[2,0] = -sv
    A[2,1] = su * cv
    A[2,2] = cu * cv

    return A


def RotMatrix_to_euler(R):
    angle =np.zeros((3,),dtype= np.float32)
    angle[2] = -math.degrees(math.asin(R[2,0]))  #Pitch

    # Gymbal lock: pitch = -90
    if (R[2,0] == 1) :
        angle[0] = 0.0 # yaw = 0
        angle[1] = math.atan2( - R[0,1], -R[0,2] ); # Roll
        print("Gimbal lock: pitch = -90")
    # Gymbal lock: pitch = 90
    elif (R[2,0] == -1):
        angle[0] = 0.0 # yaw = 0
        angle[1] = math.atan2( R[0,1], R[0,2] ) #Roll
        print("Gimbal lock: pitch = 90")

    # General solution
    else :
        angle[0] = math.degrees(math.atan2(R[1,0], R[0,0])) #yaw
        angle[1] = math.degrees(math.atan2(R[2,1], R[2,2])) #roll

    return angle #Euler angles in order yaw, roll, pitch


def simulate_camera(K, R, cam_pos, pst3d) :
    """
    K: camera calibartion matrix = [f_x, 0, c_x,
                                    0, f_y, c_y,
                                    0, 0,   1   ]

    R: rotation matrix
    cam_pos: 3D position of the camera in 3D world
    pts3d: The interested point in 3d World coordinate system

    return:
        pts2d: position of the point in image
    """
    pst3d -= cam_pos
    icsp = np.matmul(R, pst3d.transpose())
    icsp = np.matmul(K, icsp)
    icsp = icsp[:, :] / icsp[2, :]
    icsp = icsp.transpose()

    return icsp


def Create_projection_matrix(K,R,T):
    RT0 = np.zeros((3, 4))  # combined Rotation/Translation matrix
    RT0[:3, :3] = R
    RT0[:3, 3] = np.squeeze(T)
    P = np.matmul(K, RT0)  # Projection matrix
    return P


def DLT_project(P,world_points):
    image_points = list()
    for item in world_points:
        homogen_3D_point = np.append(item,1)
        # Project point into image
        icsp0 = np.matmul(P, homogen_3D_point)  # project this point using the first camera pose
        point2d_image = icsp0 / icsp0[-1]  # Normalize as we are in homogenuous coordinates
        image_points.append(point2d_image[:2])
    return image_points

if __name__ == '__main__':
    # R_cam = {'yaw': -3.8649417056027384, 'roll': -1.4287310593374152, 'pitch': 0.03757400827778057}
    # R_GPS = {'yaw': np.radians(220.8718),'roll': np.radians(1.7607), 'pitch': np.radians(-0.9411)}
    # print("image 1")
    # print(f"diff yaw = {R_GPS['yaw']- R_cam['yaw']}, roll = {R_GPS['roll']- R_cam['roll']}, pitch = {R_GPS['pitch']- R_cam['pitch']}")
    # # create rotation matrix
    # R_init = euler_to_RotMatrix(R_cam["roll"], R_cam["pitch"], R_cam["yaw"])
    # R_calc = euler_to_RotMatrix(R_GPS["roll"], R_GPS["pitch"], R_GPS["yaw"])
    # #
    # correction_R = np.linalg.inv(R_init) @ R_calc
    # print(f"correction {correction_R}")
    #
    # R_init_d = {'yaw': -3.8599984095623148, 'roll': -1.4303095238408319, 'pitch': 0.016124144027035523}
    # R_GPS = {'yaw': np.radians(220.5888), 'roll': np.radians(1.1634), 'pitch': np.radians(-1.3107)}
    # print("image 2")
    # print(
    #     f"diff yaw = {R_GPS['yaw'] - R_cam['yaw']}, roll = {R_GPS['roll'] - R_cam['roll']}, pitch = {R_GPS['pitch'] - R_cam['pitch']}")
    # # craete rotation matrix
    # R_init = euler_to_RotMatrix(R_cam["roll"], R_cam["pitch"], R_cam["yaw"])
    # R_calc = euler_to_RotMatrix(R_GPS["roll"], R_GPS["pitch"], R_GPS["yaw"])
    # #
    # correction_R = np.linalg.inv(R_init) @ R_calc
    # print(f"correction {correction_R}")
    # print("test")
    # print(correction_R @ R_init)
    # print(R_calc)

    yaw = 2.4282436015768476
    roll = -1.4357310593374153
    pitch = 0.04357400827778057
    ### Create R matrix from rotation angles ###
    # R = euler_to_RotMatrix(roll=math.radians(0),pitch=math.radians(0),yaw=math.radians(360))
    R = euler_to_RotMatrix(roll, pitch, yaw)
    print(f"Euler (radians) to Rotation matrix {R}")
    ### Get rotation angles from R matrix ###
    print(f"rotation angles from R matrix")
    yaw, pitch, roll = RotMatrix_to_euler(R)
    print(f"angles in radians yaw: {np.radians(yaw)} , pitch: {np.radians(pitch)}, roll: {np.radians(roll)}")
    print(f"angles in degree yaw: {yaw} degree, pitch: {pitch} degree, roll: {roll}")

    # using the rotation matrix Simulate image points for the camera  -> 3d to 2d
    camera_matrix = np.array([2.4553043868491550e+03, 0., 2.3078808917816095e+03,
                              0., 2.4564151520404121e+03, 2.6612355154218340e+03,
                              0., 0., 1.]).reshape(3, 3)

    cam_pos = np.expand_dims(np.array([4.77079453e+05, 5.77378952e+06, 6.83590000e+01]), axis=0)
    pst3d = np.expand_dims(np.array([4.76983900e+05, 5.77376017e+06, 7.19400000e+01]), axis=0)
    R = np.linalg.inv(R)
    # yaw, pitch, roll = RotMatrix_to_euler(R)
    # print(f"yaw: {yaw} degree, pitch: {pitch} degree, roll: {roll}")
    pts2d = simulate_camera(camera_matrix, R, cam_pos, pst3d.copy())
    print(f"3D to 2d my method {pts2d}")
    pts2d_ground_truth = np.array([3.83896900e+03, 2.82278539e+03, 1.00000000e+00])
    print(f"difference with ground truth(in pixel): { pts2d_ground_truth - pts2d}")
    print("\n\n//////////////// DLT and opencv method//////////////////")
    P = Create_projection_matrix(camera_matrix, R, np.array([0,0,0]))
    # print("Projection matrix:{}".format(P))  # Projection matrix
    reduced_point = pst3d - cam_pos
    image_points_DLT = DLT_project(P, reduced_point)
    # print_repro_error(Image_points, image_points_DLT)
    print(f"3D to 2d DLT {image_points_DLT}")


    distCoeffs = np.zeros((4, 1))
    r_rodi, _ = cv2.Rodrigues(R)
    zero_cam_pos = np.expand_dims(np.array([0.0,0.0,0.0]), axis=0)
    image_points_cv2_method, _ = cv2.projectPoints(reduced_point, R,
                                                   zero_cam_pos,
                                                   camera_matrix, distCoeffs)
    print(f"3D to 2d opencv method {image_points_cv2_method}")


    ## Do the simulation for multiple points
    pst3d = np.array([[4.76983900e+05, 5.77376017e+06, 7.19400000e+01],
                      [4.7699394e+05, 5.7737383e+06, 7.7800000e+01],
                      [4.76994030e+05, 5.77373896e+06, 7.82600000e+01],
                      [4.76993630e+05, 5.77373956e+06, 7.90100000e+01],
                      [4.76994070e+05, 5.77373935e+06, 7.85900000e+01],
                      [4.76992730e+05, 5.77374258e+06, 8.12000000e+01],
                      [4.76997730e+05, 5.77373906e+06, 7.35200000e+01],
                      [4.76995080e+05, 5.77373766e+06, 7.66500000e+01],
                      [4.76998270e+05, 5.77373899e+06, 7.36400000e+01],
                      [4.76997390e+05, 5.77373785e+06, 7.56100000e+01],
                      [4.76994930e+05, 5.77373652e+06, 7.23900000e+01],
                      [4.76999300e+05, 5.77373965e+06, 7.28500000e+01],
                      [4.76996530e+05, 5.77373968e+06, 7.43200000e+01],
                      [4.76998570e+05, 5.77373887e+06, 7.11900000e+01],
                      [4.76997860e+05, 5.77373828e+06, 7.21200000e+01],
                      [4.76998060e+05, 5.77373842e+06, 7.54800000e+01],
                      [4.76998620e+05, 5.77373889e+06, 7.47900000e+01],
                      [4.76998470e+05, 5.77373877e+06, 7.51400000e+01],
                      [4.76998430e+05, 5.77373873e+06, 7.53500000e+01],
                      [4.76998280e+05, 5.77373862e+06, 7.24800000e+01],
                      [4.76998720e+05, 5.77373897e+06, 7.51700000e+01]
                      ])

    pts2d_ground_truth = np.array([[3.83896900e+03, 2.82278539e+03, 1.00000000e+00],
                                   [3.09916513e+03, 2.71433597e+03, 1.00000000e+00],
                                   [3.11214233e+03, 2.70045493e+03, 1.00000000e+00],
                                   [3.13002310e+03, 2.67978368e+03, 1.00000000e+00],
                                   [3.11992547e+03, 2.69070171e+03, 1.00000000e+00],
                                   [3.20939645e+03, 2.61397387e+03, 1.00000000e+00],
                                   [3.07199401e+03, 2.82169386e+03,  1.00000000e+00],
                                   [3.07085863e+03, 2.74457150e+03, 1.00000000e+00],
                                   [3.06203903e+03, 2.81843895e+03, 1.00000000e+00],
                                   [3.04396265e+03, 2.76931351e+03, 1.00000000e+00],
                                   [3.05634786e+03, 2.85706270e+03, 1.00000000e+00],
                                   [3.06416633e+03, 2.83807793e+03, 1.00000000e+00],
                                   [3.10248351e+03, 2.79944649e+03, 1.00000000e+00],
                                   [3.06029903e+03, 2.88467672e+03, 1.00000000e+00],
                                   [3.05482586e+03, 2.86068998e+03, 1.00000000e+00],
                                   [3.04758225e+03, 2.77080827e+03, 1.00000000e+00],
                                   [3.05182554e+03, 2.78766178e+03, 1.00000000e+00],
                                   [3.05044746e+03, 2.77868954e+03, 1.00000000e+00],
                                   [3.04962973e+03, 2.77321147e+03, 1.00000000e+00],
                                   [3.05576748e+03,2.85035333e+03, 1.00000000e+00],
                                   [3.05135662e+03, 2.77721411e+03, 1.00000000e+00]])

    pts2d = simulate_camera(camera_matrix, R, cam_pos, pst3d.copy())

    print(f"max difference with ground truth(in pixel): {np.max(pts2d_ground_truth - pts2d, axis=0)}")
    distCoeffs = np.zeros((4, 1))
    r_rodi, _ = cv2.Rodrigues(R)
    image_points_cv2_method, _ = cv2.projectPoints(pst3d.copy() - cam_pos, r_rodi,
                                                   zero_cam_pos,
                                                   camera_matrix, distCoeffs)
    image_points_cv2_method = image_points_cv2_method.reshape(image_points_cv2_method.shape[0], 2)
    print(f"max difference with ground truth(in pixel): {np.max(pts2d_ground_truth[:,:2] - image_points_cv2_method, axis=0)}")

    # print(pst3d.shape)

    # assume we have the K but no T, R -> we want to calculate R based on 3GCP
    # pts2d = K.R [pts3d-cam_pos] -> (inv(K)*pts2d) = R *([pts3d-cam_pos])
    # (inv(K)*pts2d) * np.transpose([pts3d-cam_pos]) = R * ([pts3d-cam_pos]) * np.transpose([pts3d-cam_pos])
    # dist_coeffs = np.zeros((4, 1))
    dist_coeffs = np.array([[ -3.0381978555843195e-02, 5.5463006431906974e-03, 6.3206168823101456e-04,
                              -1.0961502937806469e-03 ]]).reshape(4,1)

    pts2d_new = pts2d[:,:2].reshape(pts2d.shape[0],1,2)



    # assume we have approximate coords
    cam_pos = np.expand_dims(np.array([4.77079453e+05 +3 , 5.77378952e+06 +1, 6.83590000e+01+0.5]), axis=0)
    # success, rotation_vector, translation_vector = cv2.solvePnP(pst3d, pts2d, camera_matrix, dist_coeffs, flags=0)
    # success, R_exp, t = cv2.solvePnP(np.array(pst3d) - cam_pos, np.array(pts2d_new), camera_matrix, dist_coeffs)
    # success, R_exp, t, inliers_idx = cv2.solvePnPRansac(np.array(pst3d) - cam_pos, np.array(pts2d_new), camera_matrix, dist_coeffs)
    success, R_exp, t = cv2.solvePnP(np.array(pst3d) - cam_pos, np.array(pts2d_new), camera_matrix,
                                                        dist_coeffs, cv2.SOLVEPNP_ITERATIVE)
    # success, R_exp, t, inliers_idx = cv2.solvePnPRansac(image1_pts3d - image1_pos, image1_pts2d, camera_matrix,
    #                                                     dist_coeffs)
    # print(inliers_idx)
    print("\nCalculated parameters using PnP:\n")
    print("t_calc:{}".format(t.T))
    # print("R_calc:{}".format(R_exp.T))
    R, _ = cv2.Rodrigues(R_exp.T)
    # print(RotMatrix_to_euler(R.T))
    print(f"rotation angles :{RotMatrix_to_euler(R)}")
    print(f" R: {R}")
    print("The translation vector is in image coordinate system definition so we need to rotate it ")
    print(f"T: {t.T @ R}")


    image1_pos = np.array([477079.453,	5773789.519,	68.359]).reshape(1,3)
    image1_pts3d = np.array([[477078.717,	5773766.515,	81.839],
                             [477076.626,	5773780.743,	66.860],
                             [477068.479,	5773759.106,	71.107],
                             [477058.638,	5773734.913,	81.915],
                             [477055.049,	5773769.174,	69.384],
                             [477024.885,	5773755.559,	78.446],
                             [477070.828,	5773785.151,	68.568],
                             [477061.321,	5773788.095,	76.682],
                            ])
    #
    image1_pts2d = np.array([[407, 1376],
                             [1195, 3511],
                             [1397, 2821],
                             [1381, 2446],
                             [2680, 2919],
                             [3034, 2540],
                             [3284, 2911],
                             [4399, 1492]
                             ], dtype=np.float32)


    image1_pts2d = image1_pts2d.reshape(image1_pts2d.shape[0], 1, 2)
    success, R_exp, t, _ = cv2.solvePnPRansac(image1_pts3d - image1_pos, image1_pts2d, camera_matrix,
                                                        dist_coeffs)
    # print(inliers_idx)
    print("\nCalculated parameters using PnP:\n")
    print("t_calc:{}".format(t.T))
    print("R_calc:{}".format(R_exp))
    R, _ = cv2.Rodrigues(R_exp.T)
    # print(RotMatrix_to_euler(R.T))
    print(f"rotation angles :{RotMatrix_to_euler(R)}")
    print(f" R: {R}")
    print("The translation vector is in image coordinate system definition so we need to rotate it ")
    print(f"T: {t.T @ R}")
    new_pos = image1_pos-(t.T @ R)
    print(f"correct cam pos = {image1_pos-(t.T @ R)}")
    print("-"*50)
    aa = image1_pts3d.copy() - image1_pos
    print(R)
    image_points_cv2_method, _ = cv2.projectPoints(image1_pts3d.copy() - new_pos, R,
                                                   zero_cam_pos,
                                                   camera_matrix, distCoeffs)
    image_points_cv2_method = image_points_cv2_method.reshape(image_points_cv2_method.shape[0], 2)
    # print(image_points_cv2_method)
    print(
        f"Difference with ground truth(in pixel): {np.round(image1_pts2d.reshape(image_points_cv2_method.shape[0],2) - image_points_cv2_method, 1)}")


    image2_pos = np.array([476982.488,	5773708.563,    68.276]).reshape(1,3)
    image2_pts3d = np.array([[476975.297,	5773685.849,	75.289],
                             [476968.946,	5773689.168,	68.518],
                             [476903.566,	5773660.561,	71.676],
                             [476943.416,	5773687.786,	69.110],
                             [476968.487,	5773704.610,	69.168],
                             [476959.502,	5773705.207,	76.453],
                             [476970.179,	5773708.526,	67.750]
                            ])
    #
    image2_pts2d = np.array([[559, 2227],
                             [1449, 3034],
                             [2485, 2940],
                             [2634, 2980],
                             [3170, 2848],
                             [3494, 2062],
                             [4104, 3103]
                             ], dtype=np.float32)

    image2_pts2d = image2_pts2d.reshape(image2_pts2d.shape[0], 1, 2)
    success, R_exp, t, inliers_idx = cv2.solvePnPRansac(image2_pts3d - image2_pos, image2_pts2d, camera_matrix,
                                                        dist_coeffs)
    print(inliers_idx)
    print("\nCalculated parameters using PnP:\n")
    print("t_calc:{}".format(t.T))
    print("R_calc:{}".format(R_exp))
    R, _ = cv2.Rodrigues(R_exp.T)
    # print(RotMatrix_to_euler(R.T))
    print(f"rotation angles :{RotMatrix_to_euler(R)}")
    print(f" R: {R}")
    print("The translation vector is in image coordinate system definition so we need to rotate it ")
    print(f"T: {t.T @ R}")
    new_pos = image2_pos - (t.T @ R)
    print(f"correct cam pos = {image2_pos - (t.T @ R)}")
    print("-" * 50)
    aa = image2_pts3d.copy() - image2_pos
    print(R)
    image_points_cv2_method, _ = cv2.projectPoints(image2_pts3d.copy() - new_pos, R,
                                                   zero_cam_pos,
                                                   camera_matrix, distCoeffs)
    image_points_cv2_method = image_points_cv2_method.reshape(image_points_cv2_method.shape[0], 2)
    # print(image_points_cv2_method)
    print(
        f"Difference with ground truth(in pixel): {np.round(image2_pts2d.reshape(image_points_cv2_method.shape[0], 2) - image_points_cv2_method, 1)}")
    exit()
    image3_pos = np.array([476919.042,	5773663.257,	68.468]).reshape(1,3)
    image3_pts3d = np.array([[476905.060,	5773630.022,	69.591],
                             [476901.934,	5773620.970,	79.954],
                             [476892.858,	5773633.203,	68.552],
                             [476882.133,	5773644.740,	69.923],
                             [476880.546,	5773647.521,	80.441],
                             [476904.084,	5773659.753,	71.681],
                             [476899.884,	5773662.212,	70.078]
                            ])
    #
    image3_pts2d = np.array([[767, 2985],
                             [798, 2342],
                             [1717, 3048],
                             [2674, 2943],
                             [2852, 2314],
                             [3451, 2460],
                             [3846, 2756]
                             ], dtype=np.float32)

    image3_pts2d = image3_pts2d.reshape(image3_pts2d.shape[0], 1, 2)
    success, R_exp, t, inliers_idx = cv2.solvePnPRansac(image3_pts3d - image3_pos, image3_pts2d, camera_matrix,
                                                        dist_coeffs)
    print(inliers_idx)
    print("\nCalculated parameters using PnP:\n")
    print("t_calc:{}".format(t.T))
    print("R_calc:{}".format(R_exp.T))
    R, _ = cv2.Rodrigues(R_exp.T)
    # print(RotMatrix_to_euler(R.T))
    print(f"rotation angles :{RotMatrix_to_euler(R)}")
    print(f" R: {R}")
    print("The translation vector is in image coordinate system definition so we need to rotate it ")
    print(f"T: {t.T @ R}")

    image4_pos = np.array([476851.091,	5773703.925,	67.883]).reshape(1,3)
    image4_pts3d = np.array([[476837.666,	5773706.381,	67.793],
                             [476829.846,	5773714.391,	67.316],
                             [476819.114,	5773730.805,	65.532],
                             [476722.289,	5773856.664,	89.163],
                             [476812.799,	5773749.365,	68.623],
                             [476841.197,	5773726.099,	68.919]
                            ])
    #
    image4_pts2d = np.array([[500, 3089],
                             [1389, 3103],
                             [2029, 3161],
                             [2441, 2742],
                             [2521, 2964],
                             [3169, 2863]
                             ], dtype=np.float32)

    image4_pts2d = image4_pts2d.reshape(image4_pts2d.shape[0], 1, 2)

    success, R_exp, t, inliers_idx = cv2.solvePnPRansac(image4_pts3d - image4_pos, image4_pts2d, camera_matrix,
                                                        dist_coeffs)
    print(inliers_idx)
    print("\nCalculated parameters using PnP:\n")
    print("t_calc:{}".format(t.T))
    # print("R_calc:{}".format(R_exp.T))
    R, _ = cv2.Rodrigues(R_exp.T)
    # print(RotMatrix_to_euler(R.T))
    print(f"rotation angles :{RotMatrix_to_euler(R)}")
    print(f" R: {R}")
    print("The translation vector is in image coordinate system definition so we need to rotate it ")
    print(f"T: {t.T @ R}")