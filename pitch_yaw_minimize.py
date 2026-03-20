import numpy as np
from scipy.optimize import minimize

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

        {'u': 2301.154296875,'v': 1020.3768310546875,'true_X': -6.072, 'true_Z': 6.9},
        {'u': 1922.864501953125,'v': 843.5469360351562,'true_X': -8.073, 'true_Z': 7.184},
        {'u': 1546.197265625,'v': 700.607421875,'true_X': -10.498, 'true_Z': 7.976},
        {'u': 1252.08984375,'v': 589.343505859375,'true_X': -13.289, 'true_Z': 9.144}, #should z be negative here?? greater than 90 deg
        {'u': 1022.1502685546875,'v': 501.550048828125,'true_X': -16.355, 'true_Z': 10.565},
        {'u': 819.412109375,'v': 441.94708251953125,'true_X': -19.608, 'true_Z': 12.149},
        {'u': 667.30224609375,'v': 401.0665283203125,'true_X': -22.993, 'true_Z': 13.842}
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

    print("\n✅ MULTI-POINT CALIBRATION COMPLETE")
    print("-" * 40)
    print(f"Optimal Pitch: {best_pitch:.2f} degrees")
    print(f"Optimal Yaw:   {best_yaw:.2f} degrees")
    print(f"Average Error: {average_error:.2f} meters per point")
    print("-" * 40)
    print("Use these numbers for Channel 4!")

if __name__ == "__main__":
    find_global_best_angles()