import numpy as np
from scipy.optimize import minimize
import csv
import json

def find_global_best_angles():
    H = 7.0  # Camera height in meters
    
    # Perfect Wide-Angle K-Matrix (2560x1920)
    K = np.array([
        [1325.4,      0, 1280.0],
        [     0, 1325.4,  960.0],
        [     0,      0,      1]
    ], dtype=np.float64)
    
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    # ==========================================
    # 🎯 MULTI-POINT CALIBRATION DATA
    # Add as many known points as you want here!
    # ==========================================
    calibration_points = [
        # Point 1: Centerline, 5m out
        # {'u': 1755.3504638671875, 'v': 1527.0423583984375, 'true_X': 0.0,  'true_Z': 5.0},
        
        # # Point 2: Left side, 5m out
        # {'u': 548.5708618164062,  'v': 1737.449462890625, 'true_X': -4.4, 'true_Z': 5.0},

        # {'u': 300.3877258300781,'v': 626.58056640625,'true_X': 0.0, 'true_Z': 16.0}

        # {'u': 1298.5810546875,'v': 1194.504150390625,'true_X': 4.83, 'true_Z': 6.9},
        # {'u': 1944.55078125,'v': 1074.4798583984375,'true_X': 8.33121433628, 'true_Z': 8.02522124306},
        # {'u': 2405.728759765625,'v': 942.6558837890625,'true_X': 12.4376303756, 'true_Z': 9.96012931643},
        # {'u': 904.2249755859375,'v': 1639.103271484375,'true_X': 2.6, 'true_Z': 5.0}, #should z be negative here?? greater than 90 deg
    
        # {'u': 248.20706176757812,'v': 1574.53369140625,'true_X': 8.33121433628, 'true_Z': 8.02522124306},
        # {'u': 536.4527587890625,'v': 1199.762939453125,'true_X': 12.4376303756, 'true_Z': 9.96012931643},
        # {'u': 763.6453857421875,'v': 928.8372802734375,'true_X': 16.9423007424, 'true_Z': 12.3289973639},
        # {'u': 923.8876953125,'v': 728.266357421875,'true_X': 21.6803183955, 'true_Z': 14.9266264105},
        # {'u': 1045.9188232421875,'v': 598.07568359375,'true_X': 26.557070185, 'true_Z': 17.6523136161}

        # {'u': 1770.226318359375,'v': 648.5950927734375,'true_X': 0.0, 'true_Z': 13.0},
        # {'u': 1780.216796875,'v': 1193.007568359375,'true_X': 0.0, 'true_Z': 7.0},
        # {'u': 1755.3504638671875,'v': 1527.0423583984375,'true_X': 0.0, 'true_Z': 5.0},
        # {'u': 1780.3740234375,'v': 856.560546875,'true_X': 0.0, 'true_Z': 10.0}

        {'u': 2301.154296875,'v': 1020.3768310546875,'true_X': -5.451, 'true_Z': 4.2297},
        {'u': 1022.1502685546875,'v':  501.550048828125,'true_X': -10.335, 'true_Z': -2.0903},
        {'u': 1922.864501953125,'v': 843.5469360351562,'true_X': -6.667, 'true_Z': 2.6497},
        {'u': 819.412109375,'v': 441.94708251953125,'true_X': -11.581, 'true_Z': -3.6703},
        {'u': 1252.08984375,'v': 589.343505859375,'true_X': -9.129, 'true_Z': -0.5103},
        {'u': 1546.197265625,'v': 700.607421875,'true_X': -7.903, 'true_Z': 1.0697},
        {'u': 667.30224609375,'v': 401.0665283203125,'true_X': -12.807, 'true_Z': -5.2503}
    ]

    def calculate_average_error(angles):
        pitch_deg, yaw_deg = angles
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        
        R = Ry @ Rx
        
        total_error = 0.0
        
        # Test the guess against EVERY point in our list
        for pt in calibration_points:
            ray_cam = np.array([(pt['u'] - cx) / fx, (pt['v'] - cy) / fy, 1.0])
            ray_world = R @ ray_cam
            dx, dy, dz = ray_world

            if dy <= 1e-6:
                return 999999.0 # Heavily penalize pointing at the sky

            t = H / dy
            pred_X = t * dx
            pred_Z = t * dz

            # Calculate error for this specific point
            point_error = np.sqrt((pred_X - pt['true_X'])**2 + (pred_Z - pt['true_Z'])**2)
            total_error += point_error

        # We want to minimize the AVERAGE error across the whole image
        return total_error / len(calibration_points)

    print(f"Running global optimization on {len(calibration_points)} points...")
    # Start guessing at 45 Pitch, 0 Yaw
    result = minimize(calculate_average_error, [-40.0, -30.0], method='Nelder-Mead')
    
    best_pitch, best_yaw = result.x
    average_error = result.fun
    
    pitch = np.radians(best_pitch)
    yaw = np.radians(best_yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R = Ry @ Rx

    # List to hold all our row data
    export_data = []

    for i, pt in enumerate(calibration_points):
        ray_cam = np.array([(pt['u'] - cx) / fx, (pt['v'] - cy) / fy, 1.0])
        ray_world = R @ ray_cam
        dx, dy, dz = ray_world

        if dy > 1e-6:
            t = H / dy
            pred_X = t * dx
            pred_Z = t * dz
            
            error_X = abs(pred_X - pt['true_X'])
            error_Z = abs(pred_Z - pt['true_Z'])
            total_point_error = np.sqrt(error_X**2 + error_Z**2)
            
            # Store the data in a dictionary
            export_data.append({
                "Point_ID": i + 1,
                "u_pixel": round(pt['u'], 2),
                "v_pixel": round(pt['v'], 2),
                "True_X_m": round(pt['true_X'], 3),
                "True_Z_m": round(pt['true_Z'], 3),
                "Pred_X_m": round(pred_X, 3),
                "Pred_Z_m": round(pred_Z, 3),
                "Error_X_m": round(error_X, 3),
                "Error_Z_m": round(error_Z, 3),
                "Total_Error_m": round(total_point_error, 3)
            })

    # --- SAVE TO CSV ---
    csv_filename = "calibration_errors.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=export_data[0].keys())
        writer.writeheader()
        writer.writerows(export_data)
    print(f"\n📁 Saved CSV to: {csv_filename}")

    print("\n✅ MULTI-POINT CALIBRATION COMPLETE")
    print("-" * 40)
    print(f"Optimal Pitch: {best_pitch:.2f} degrees")
    print(f"Optimal Yaw:   {best_yaw:.2f} degrees")
    print(f"Average Error: {average_error:.2f} meters per point")
    print("-" * 40)
    print("Use these numbers for Channel 4!")

if __name__ == "__main__":
    find_global_best_angles()