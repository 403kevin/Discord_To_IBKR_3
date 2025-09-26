import sys
import os

print("="*50)
print("--- PYTHON ENVIRONMENT DIAGNOSTIC ---")
print("="*50)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\n[INFO] This diagnostic script is located in:\n{script_dir}\n")

# Check if the project root is in the Python path
print("[DIAGNOSIS] Checking if the project root is in Python's list of searchable paths (sys.path)...\n")
is_in_path = script_dir in sys.path
if is_in_path:
    print(">>> RESULT: SUCCESS! The project root is correctly included in the Python path.")
else:
    print(">>> RESULT: FAILURE! The project root is NOT in the Python path. This is the cause of the ModuleNotFoundError.")

# Display the full path for forensic analysis
print("\n--- Full Python Path (sys.path) ---")
for i, p in enumerate(sys.path):
    print(f"{i}: {p}")
print("------------------------------------")

print("\n[CONCLUSION] If the result above was FAILURE, the environment is misconfigured.")
print("If the result was SUCCESS but you still get an error, the project's __init__.py files may be missing or named incorrectly.")
print("="*50)

