import os
import json
import sys
import pandas as pd
import numpy as np
import logging

# Add parent directory to path to allow importing from Model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try importing, handle potential import errors
try:
    from utils.CurveDetector import CurveDetector
except ImportError:
    # Fallback if running from a different context
    sys.path.append(os.path.join(parent_dir, 'utils'))
    from CurveDetector import CurveDetector

# Constants
# Assuming script is run from data_analysis/utils/ or similar depth
# data/2025-main is at ../../data/2025-main relative to this script
DATA_DIR = os.path.abspath(os.path.join(parent_dir, "..", "data", "2025-main"))
OUTPUT_FILE = os.path.abspath(os.path.join(parent_dir, "dataset_curves.csv"))
MAX_POINTS = 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_padded_array(arr, max_len=MAX_POINTS):
    """Pads or truncates an array to max_len."""
    # Ensure input is a numpy array or list
    if not isinstance(arr, (list, np.ndarray)):
        arr = []
    
    arr = np.array(arr)
    
    if len(arr) == 0:
        return np.zeros(max_len)
        
    if len(arr) > max_len:
        return arr[:max_len]
    else:
        return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0)

def process_grand_prix(gp_path):
    """Processes a single Grand Prix directory."""
    gp_name = os.path.basename(gp_path)
    race_path = os.path.join(gp_path, "Race")
    
    if not os.path.exists(race_path):
        return []

    # Find corners file
    corners_file = os.path.join(race_path, "corners.json")
    if not os.path.exists(corners_file):
        candidates = [f for f in os.listdir(race_path) if f.startswith("corners") and f.endswith(".json")]
        if candidates:
            corners_file = os.path.join(race_path, candidates[0])
        else:
             logging.warning(f"No corners file found for {gp_name}, skipping.")
             return []

    logging.info(f"Processing {gp_name}...")
    
    gp_curves_data = []

    # Iterate over drivers
    for driver_dir in os.listdir(race_path):
        driver_path = os.path.join(race_path, driver_dir)
        if not os.path.isdir(driver_path):
            continue

        laptimes_file = os.path.join(driver_path, "laptimes.json")
        if not os.path.exists(laptimes_file):
            continue

        try:
            with open(laptimes_file, 'r') as f:
                laptimes_data = json.load(f)
        except Exception as e:
            logging.error(f"Error reading laptimes for {driver_dir}: {e}")
            continue

        # Create lookup for lap info
        lap_info = {}
        if "lap" in laptimes_data:
            for i, lap_val in enumerate(laptimes_data["lap"]):
                lap_num = int(lap_val)
                lap_info[lap_num] = {
                    "Compound": laptimes_data["compound"][i] if "compound" in laptimes_data else "UNKNOWN",
                    "TireLife": laptimes_data["life"][i] if "life" in laptimes_data else 0,
                    "Stint": laptimes_data["stint"][i] if "stint" in laptimes_data else 0
                }

        # Iterate over telemetry files
        for filename in os.listdir(driver_path):
            if filename.endswith("_tel.json"):
                try:
                    parts = filename.split("_")
                    if not parts[0].isdigit():
                        continue
                        
                    lap_num = int(parts[0])
                    telemetry_file = os.path.join(driver_path, filename)
                    
                    # Instantiate CurveDetector
                    # Note: CurveDetector API might change slightly based on inspection, 
                    # but looking at file content, it takes filenames in init.
                    detector = CurveDetector(
                        telemetry_filename=telemetry_file,
                        corners_filename=corners_file
                    )
                    
                    curves = detector.calcolo_curve()
                    
                    for curve in curves:
                        # Basic info
                        curve_entry = {
                            "GrandPrix": gp_name,
                            "Session": "Race",
                            "Driver": driver_dir,
                            "Lap": lap_num,
                            "CornerID": curve.corner_id,
                            "Compound": lap_info.get(lap_num, {}).get("Compound", "UNKNOWN"),
                            "TireLife": lap_info.get(lap_num, {}).get("TireLife", 0),
                            "Stint": lap_info.get(lap_num, {}).get("Stint", 0)
                        }

                        # Arrays to pad
                        # Based on Curve.py, attributes available are:
                        # speed, rpm, throttle, brake, acc_x, acc_y, acc_z, x, y, z, distance, time
                        arrays_to_process = {
                            "speed": curve.speed,
                            "rpm": curve.rpm,
                            "throttle": curve.throttle,
                            "brake": curve.brake,
                            "acc_x": curve.acc_x,
                            "acc_y": curve.acc_y,
                            "acc_z": curve.acc_z,
                            "x": curve.x,
                            "y": curve.y,
                            "z": curve.z,
                            "distance": curve.distance,
                            "time": curve.time
                        }

                        for name, arr in arrays_to_process.items():
                            padded = get_padded_array(arr)
                            # Flatten into columns: speed_0, speed_1, ... speed_49
                            for i in range(len(padded)):
                                curve_entry[f"{name}_{i}"] = padded[i]
                        
                        gp_curves_data.append(curve_entry)

                except Exception as e:
                    # Logging explicitly to debug if needed, but not crashing
                    logging.debug(f"Error processing {filename} in {driver_dir}: {e}")
                    continue

    return gp_curves_data

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return

    all_data = []
    
    # Iterate through possible Grand Prix directories
    items = os.listdir(DATA_DIR)
    for item in items:
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and "Grand Prix" in item:
            gp_data = process_grand_prix(item_path)
            all_data.extend(gp_data)

    if not all_data:
        print("No curves extracted.")
        return

    print(f"Extracted {len(all_data)} total curves. Saving to CSV...")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved dataset to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    main()
