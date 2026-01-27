import os
import json
import sys
import pandas as pd
import numpy as np
import logging
import gc

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
PADDING = -1000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_padded_array(arr, max_len=MAX_POINTS):
    """Pads or truncates an array to max_len."""
    # Ensure input is a numpy array or list
    if not isinstance(arr, (list, np.ndarray)):
        arr = []
    
    arr = np.array(arr)
    
    if len(arr) == 0:
        return np.full(max_len,PADDING)
        
    if len(arr) > max_len:
        return arr[:max_len]
    else:
        return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=PADDING)

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


        # Laptimes extraction moved to CurveDetector


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
                            "Compound": detector.compound,
                            "TireLife": detector.tire_life,
                            "Stint": detector.stint
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

    # Initialize output file: remove if exists to start fresh
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing output file: {OUTPUT_FILE}")

    total_curves = 0
    header_written = False
    
    # Iterate through possible Grand Prix directories
    items = os.listdir(DATA_DIR)
    # Sort items to ensure deterministic order if needed, or just standard os.listdir
    items.sort()
    
    for item in items:
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and "Grand Prix" in item:
            # Process one Grand Prix
            gp_data = process_grand_prix(item_path)
            
            if not gp_data:
                continue
            
            
            print(f"Creating DataFrame... ")
            # Convert to DataFrame
            df = pd.DataFrame(gp_data)
            
            # Save incrementally
            try:
                print(f"Saveing on CSV... ")
                # Append mode 'a', header only if not written yet
                df.to_csv(OUTPUT_FILE, mode='a', index=False, chunksize=10000,header=not header_written)
                header_written = True
                
                count = len(df)
                total_curves += count
                print(f"Saved {count} curves from {item}. Total so far: {total_curves}")
                
                # Free memory explicitly
                del df
                del gp_data
                gc.collect()
                
            except Exception as e:
                print(f"Error saving CSV for {item}: {e}")

    if total_curves == 0:
        print("No curves extracted.")
    else:
        print(f"Finished processing. Total curves extracted: {total_curves}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
