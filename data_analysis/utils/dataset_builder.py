import os
import json
import sys
import pandas as pd
import numpy as np
import logging
import gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ============================================================================
# PATH SETUP
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from utils.CurveDetector import CurveDetector
except ImportError:
    sys.path.append(os.path.join(parent_dir, 'utils'))
    from CurveDetector import CurveDetector

# ============================================================================
# CONSTANTS
# ============================================================================
BASE_DATA_DIR = os.path.abspath(os.path.join(parent_dir, "..", "data"))
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "dataset", "dataset_curves.csv")
MAX_POINTS = 50
PADDING_VALUE = -1000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class DatasetConfig:
    """
    Configuration for dataset building.
    
    Attributes:
        years: List of years to include (e.g., [2024, 2025]). If None, includes all available.
        drivers: List of driver codes to include (e.g., ["LEC", "HAM"]). If None, includes all.
        sessions: List of sessions to process (e.g., ["Race", "Qualifying"]).
        output_file: Path to the output CSV file.
        max_points: Maximum number of data points per curve.
        padding_value: Value used for padding shorter curves.
    """
    years: Optional[List[int]] = None
    drivers: Optional[List[str]] = None
    sessions: List[str] = field(default_factory=lambda: ["Race"])
    output_file: str = DEFAULT_OUTPUT_FILE
    max_points: int = MAX_POINTS
    padding_value: float = PADDING_VALUE


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_padded_array(arr: Any, max_len: int = MAX_POINTS, padding: float = PADDING_VALUE) -> np.ndarray:
    """
    Pads or truncates an array to a fixed length.
    
    Args:
        arr: Input array (list or numpy array).
        max_len: Target length for the output array.
        padding: Value used for padding.
    
    Returns:
        Numpy array of length max_len.
    """
    if not isinstance(arr, (list, np.ndarray)):
        arr = []
    
    arr = np.array(arr)
    
    if len(arr) == 0:
        return np.full(max_len, padding)
    
    if len(arr) > max_len:
        return arr[:max_len]
    
    return np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=padding)


def get_data_directories(years: Optional[List[int]] = None) -> List[str]:
    """
    Gets data directories based on specified years.
    
    Args:
        years: List of years to include. If None, scans for all available years.
    
    Returns:
        List of absolute paths to data directories.
    """
    if years is None:
        # Find all available year directories
        directories = []
        if os.path.exists(BASE_DATA_DIR):
            for item in os.listdir(BASE_DATA_DIR):
                item_path = os.path.join(BASE_DATA_DIR, item)
                if os.path.isdir(item_path) and "-main" in item:
                    directories.append(item_path)
        return sorted(directories)
    
    # Build paths for specific years
    return [
        os.path.join(BASE_DATA_DIR, f"{year}-main")
        for year in years
    ]


def find_corners_file(session_path: str) -> Optional[str]:
    """
    Finds the corners file in a session directory.
    
    Args:
        session_path: Path to the session directory.
    
    Returns:
        Path to corners file, or None if not found.
    """
    corners_file = os.path.join(session_path, "corners.json")
    
    if os.path.exists(corners_file):
        return corners_file
    
    # Look for alternative corners files
    candidates = [
        f for f in os.listdir(session_path) 
        if f.startswith("corners") and f.endswith(".json")
    ]
    
    if candidates:
        return os.path.join(session_path, candidates[0])
    
    return None


# ============================================================================
# CURVE PROCESSING
# ============================================================================
def extract_curve_data(
    curve, 
    detector: CurveDetector, 
    gp_name: str, 
    session: str, 
    driver: str, 
    lap_num: int,
    config: DatasetConfig
) -> Dict[str, Any]:
    """
    Extracts data from a single curve into a dictionary.
    
    Args:
        curve: Curve object containing telemetry data.
        detector: CurveDetector instance with compound/tire info.
        gp_name: Name of the Grand Prix.
        session: Session name (Race, Qualifying, etc.).
        driver: Driver code.
        lap_num: Lap number.
        config: Dataset configuration.
    
    Returns:
        Dictionary with curve data flattened for DataFrame.
    """
    # Basic metadata
    curve_entry = {
        "GrandPrix": gp_name,
        "Session": session,
        "Driver": driver,
        "Lap": lap_num,
        "CornerID": curve.corner_id,
        "Compound": detector.compound,
        "TireLife": detector.tire_life,
        "Stint": detector.stint
    }
    
    # Telemetry arrays to process
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
    
    # Pad and flatten each array into columns
    for name, arr in arrays_to_process.items():
        padded = get_padded_array(arr, config.max_points, config.padding_value)
        for i, value in enumerate(padded):
            curve_entry[f"{name}_{i}"] = value
    
    return curve_entry


def process_lap_telemetry(
    telemetry_file: str,
    corners_file: str,
    gp_name: str,
    session: str,
    driver: str,
    lap_num: int,
    config: DatasetConfig
) -> List[Dict[str, Any]]:
    """
    Processes a single lap's telemetry file.
    
    Args:
        telemetry_file: Path to telemetry JSON file.
        corners_file: Path to corners JSON file.
        gp_name: Grand Prix name.
        session: Session name.
        driver: Driver code.
        lap_num: Lap number.
        config: Dataset configuration.
    
    Returns:
        List of curve data dictionaries.
    """
    curves_data = []
    
    detector = CurveDetector(
        telemetry_filename=telemetry_file,
        corners_filename=corners_file
    )
    
    curves = detector.calcolo_curve()
    
    for curve in curves:
        curve_data = extract_curve_data(
            curve, detector, gp_name, session, driver, lap_num, config
        )
        curves_data.append(curve_data)
    
    return curves_data


def process_driver(
    driver_path: str,
    corners_file: str,
    gp_name: str,
    session: str,
    driver: str,
    config: DatasetConfig
) -> List[Dict[str, Any]]:
    """
    Processes all laps for a single driver.
    
    Args:
        driver_path: Path to driver's telemetry directory.
        corners_file: Path to corners JSON file.
        gp_name: Grand Prix name.
        session: Session name.
        driver: Driver code.
        config: Dataset configuration.
    
    Returns:
        List of curve data dictionaries.
    """
    driver_curves_data = []
    
    for filename in os.listdir(driver_path):
        if not filename.endswith("_tel.json"):
            continue
        
        parts = filename.split("_")
        if not parts[0].isdigit():
            continue
        
        try:
            lap_num = int(parts[0])
            telemetry_file = os.path.join(driver_path, filename)
            
            lap_data = process_lap_telemetry(
                telemetry_file, corners_file, gp_name, session, driver, lap_num, config
            )
            driver_curves_data.extend(lap_data)
            
        except Exception as e:
            logger.debug(f"Error processing {filename} for {driver}: {e}")
            continue
    
    return driver_curves_data


def process_session(
    gp_path: str,
    session: str,
    config: DatasetConfig
) -> List[Dict[str, Any]]:
    """
    Processes a single session (Race, Qualifying, etc.) for a Grand Prix.
    
    Args:
        gp_path: Path to Grand Prix directory.
        session: Session name to process.
        config: Dataset configuration.
    
    Returns:
        List of curve data dictionaries.
    """
    gp_name = os.path.basename(gp_path)
    session_path = os.path.join(gp_path, session)
    
    if not os.path.exists(session_path):
        return []
    
    corners_file = find_corners_file(session_path)
    if not corners_file:
        logger.warning(f"No corners file found for {gp_name}/{session}, skipping.")
        return []
    
    logger.info(f"Processing {gp_name} - {session}...")
    
    session_curves_data = []
    
    # Determine which drivers to process
    available_drivers = [
        d for d in os.listdir(session_path) 
        if os.path.isdir(os.path.join(session_path, d))
    ]
    
    if config.drivers is not None:
        drivers_to_process = [d for d in config.drivers if d in available_drivers]
    else:
        drivers_to_process = available_drivers
    
    for driver in drivers_to_process:
        driver_path = os.path.join(session_path, driver)
        
        driver_data = process_driver(
            driver_path, corners_file, gp_name, session, driver, config
        )
        session_curves_data.extend(driver_data)
    
    return session_curves_data


def process_grand_prix(gp_path: str, config: DatasetConfig) -> List[Dict[str, Any]]:
    """
    Processes all configured sessions for a Grand Prix.
    
    Args:
        gp_path: Path to Grand Prix directory.
        config: Dataset configuration.
    
    Returns:
        List of curve data dictionaries.
    """
    gp_curves_data = []
    
    for session in config.sessions:
        session_data = process_session(gp_path, session, config)
        gp_curves_data.extend(session_data)
    
    return gp_curves_data


# ============================================================================
# MAIN BUILDER
# ============================================================================
def build_dataset(config: Optional[DatasetConfig] = None) -> int:
    """
    Builds the complete dataset based on configuration.
    
    Args:
        config: Dataset configuration. Uses defaults if None.
    
    Returns:
        Total number of curves extracted.
    """
    if config is None:
        config = DatasetConfig()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(config.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove existing output file
    if os.path.exists(config.output_file):
        os.remove(config.output_file)
        print(f"Removed existing output file: {config.output_file}")
    
    total_curves = 0
    header_written = False
    
    # Get data directories
    data_dirs = get_data_directories(config.years)
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Data directory not found, skipping: {data_dir}")
            continue
        
        print(f"\n--- Processing data directory: {data_dir} ---")
        
        # Find Grand Prix directories
        gp_dirs = sorted([
            item for item in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, item)) and "Grand Prix" in item
        ])
        
        for gp_name in gp_dirs:
            gp_path = os.path.join(data_dir, gp_name)
            
            # Process Grand Prix
            gp_data = process_grand_prix(gp_path, config)
            
            if not gp_data:
                continue
            
            # Convert to DataFrame and save incrementally
            print(f"Creating DataFrame...")
            df = pd.DataFrame(gp_data)
            
            try:
                print(f"Saving to CSV...")
                df.to_csv(
                    config.output_file, 
                    mode='a', 
                    index=False, 
                    chunksize=10000,
                    header=not header_written
                )
                header_written = True
                
                count = len(df)
                total_curves += count
                print(f"Saved {count} curves from {gp_name}. Total so far: {total_curves}")
                
                # Free memory
                del df
                del gp_data
                gc.collect()
                
            except Exception as e:
                print(f"Error saving CSV for {gp_name}: {e}")
    
    if total_curves == 0:
        print("No curves extracted.")
    else:
        print(f"Finished processing. Total curves extracted: {total_curves}. Saved to {config.output_file}")
    
    return total_curves


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def main():
    """
    Main entry point with example configuration.
    Modify the config below to customize data extraction.
    """
    # === CONFIGURATION ===
    # Customize these settings as needed:
    
    config = DatasetConfig(
        years=[2024, 2025],              # Years to include (None for all available)
        drivers=None,                     # Driver codes to include (None for all)
        sessions=["Qualifying", "Race"],  # Sessions to process
        output_file=DEFAULT_OUTPUT_FILE,  # Output CSV path
    )

    # === RUN ===
    build_dataset(config)


if __name__ == "__main__":
    main()
