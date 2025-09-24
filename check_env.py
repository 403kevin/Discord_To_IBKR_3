import os
from dotenv import load_dotenv

# This is the exact same process our config.py uses.
print("Attempting to load the .env file...")
load_dotenv()

# We will now check for the specific token variable.
token = os.getenv("DISCORD_AUTH_TOKEN")

print("-" * 30)
if token:
    # If the token is found, we will print a confirmation WITHOUT showing the secret.
    # We will only show the first 8 characters to prove it was loaded.
    print("✅ SUCCESS: DISCORD_AUTH_TOKEN was found.")
    print(f"   Token starts with: {token[:8]}...")
else:
    # If the token is not found, it means the .env file was not loaded correctly.
    print("❌ FAILURE: DISCORD_AUTH_TOKEN was NOT found.")
    print("   This is the cause of the 'Improper token' error.")
    print("   Please check the following:")
    print("   1. Is the .env file in the SAME FOLDER as main.py?")
    print("   2. Is the file named exactly '.env' (with a period at the start)?")
    print("   3. Is the variable name inside the file spelled 'DISCORD_AUTH_TOKEN'?")
print("-" * 30)
