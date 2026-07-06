"""
run_all.py
-----------
MASTER AUTOMATION SCRIPT — runs the ENTIRE project end-to-end in one go:

  Step 1: Generate all 4 raw datasets
  Step 2: Run data fusion + feature engineering pipeline
  Step 3: Train all 3 ML models
  Step 4: Launch the live Streamlit dashboard

Run this ONE file from the project root folder:
    python run_all.py

This is the single command for the full project — no need to run
each script separately.
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(step_name, script_path, cwd):
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    result = subprocess.run([sys.executable, script_path], cwd=cwd)
    if result.returncode != 0:
        print(f"\n❌ {step_name} failed. Stopping pipeline.")
        sys.exit(1)
    print(f"✅ {step_name} completed successfully.")


if __name__ == "__main__":
    src_dir = os.path.join(BASE_DIR, "src")
    dashboard_dir = os.path.join(BASE_DIR, "dashboard")

    run_step("1/4 Generating datasets", "generate_datasets.py", src_dir)
    run_step("2/4 Running data fusion pipeline", "data_pipeline.py", src_dir)
    run_step("3/4 Training ML models", "train_models.py", src_dir)

    print("\n" + "=" * 70)
    print("STEP: 4/4 Launching live dashboard (browser will open automatically)")
    print("=" * 70)
    print("Press CTRL+C in this terminal to stop the dashboard.\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        cwd=dashboard_dir,
    )
