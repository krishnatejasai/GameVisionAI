import subprocess
import sys

print("\n========== GAMEVISION AI PIPELINE ==========\n")

python = sys.executable

print("Step 1: Running detection + tracking...")
subprocess.run([python, "src/track.py"], check=True)

print("\nStep 2: Running analytics...")
subprocess.run([python, "src/analytics.py"], check=True)

print("\n========== PIPELINE COMPLETE ==========")
print("GameVision AI pipeline completed successfully.")