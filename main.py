"""
Main entry point. Runs the full pipeline.
Usage: python main.py
"""
import subprocess, sys

steps = [
    ("Preprocessing", "src/preprocess.py"),
    ("Training",      "src/train.py"),
    ("Plotting",      "src/plot.py"),
]

for name, script in steps:
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"Error in {script}. Stopping.")
        sys.exit(1)

print("\nPipeline complete. Check data/results.json and figures/")
