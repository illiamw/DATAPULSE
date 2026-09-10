import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import app_mlops.data.load_data as ld
import app_mlops.data.gerate_data as gd
import app_mlops.data.preprocess_data as ppd
import app_mlops.features.build_features as bf

def main():
    # Generate and save the final DataFrame to a new CSV file
    file_path_raw = "data/raw/industrial_data_raw.csv"
    df = gd.gerate_data(file_path_raw)

    # Load raw data
    df = ld.load_data("data/raw/industrial_data_raw.csv")

    # Preprocess data
    df = ppd.main(df)

    df.to_csv("data/silver/industrial_data_silver.csv", index=False)  # Replace with desired path

    # Build features
    df = bf.main(df)

    # Save the final DataFrame to a new CSV file
    df.to_csv("data/serving/industrial_data_serving.csv", index=False)  # Replace with desired path




if __name__ == "__main__":
    main()