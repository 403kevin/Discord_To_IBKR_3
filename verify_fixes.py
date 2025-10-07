"""
Simple Verification Script
Run this to check if your files need fixing after the buzzword consolidation
"""

import os

print("=" * 80)
print("VERIFICATION SCRIPT - Checking for issues...")
print("=" * 80)

issues_found = []

# Check 1: config.py line with self.buzzwords
print("\n1️⃣  Checking services/config.py...")
try:
    with open('services/config.py', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if 'self.buzzwords =' in line and 'self.buzzwords_buy' in line:
                print(f"   Found at line {i}: {line.strip()}")
                if 'buzzwords_sell' in line:
                    print("   ❌ ERROR: Still references buzzwords_sell (deleted attribute)")
                    issues_found.append("config.py line " + str(i) + " has buzzwords_sell")
                else:
                    print("   ✅ OK: No buzzwords_sell reference")
except FileNotFoundError:
    print("   ❌ ERROR: services/config.py not found")
    issues_found.append("config.py not found")

# Check 2: signal_processor.py reconciliation task
print("\n2️⃣  Checking bot_engine/signal_processor.py...")
try:
    with open('bot_engine/signal_processor.py', 'r') as f:
        content = f.read()
        if 'asyncio.create_task(self._reconcile_positions_periodically())' in content:
            print("   ✅ OK: Reconciliation task is started")
        else:
            print("   ❌ ERROR: Reconciliation task NOT started in main loop")
            issues_found.append("signal_processor.py missing reconciliation task")
except FileNotFoundError:
    print("   ❌ ERROR: bot_engine/signal_processor.py not found")
    issues_found.append("signal_processor.py not found")

# Check 3: ib_interface.py cancel method
print("\n3️⃣  Checking interfaces/ib_interface.py...")
try:
    with open('interfaces/ib_interface.py', 'r') as f:
        content = f.read()
        if 'def cancel_all_orders_for_contract' in content:
            print("   ✅ OK: cancel_all_orders_for_contract() method exists")
        else:
            print("   ❌ ERROR: cancel_all_orders_for_contract() method missing")
            issues_found.append("ib_interface.py missing cancel method")
        
        if 'def close_position' in content:
            print("   ❌ ERROR: close_position() method exists (shouldn't - not in original code)")
            issues_found.append("ib_interface.py has close_position method (remove it)")
        else:
            print("   ✅ OK: No close_position() method (correct)")
except FileNotFoundError:
    print("   ❌ ERROR: interfaces/ib_interface.py not found")
    issues_found.append("ib_interface.py not found")

# Check 4: signal_parser.py (should be fixed now)
print("\n4️⃣  Checking services/signal_parser.py...")
try:
    with open('services/signal_parser.py', 'r') as f:
        content = f.read()
        if 'self.config.buzzwords_sell' in content:
            # Check if it's only in comments
            lines_with_ref = [line for line in content.split('\n') if 'self.config.buzzwords_sell' in line]
            only_comments = all(line.strip().startswith('#') for line in lines_with_ref)
            if only_comments:
                print("   ✅ OK: Only mentions buzzwords_sell in comments (safe)")
            else:
                print("   ❌ ERROR: Still has active buzzwords_sell reference")
                issues_found.append("signal_parser.py has active buzzwords_sell")
        else:
            print("   ✅ OK: No buzzwords_sell references")
except FileNotFoundError:
    print("   ❌ ERROR: services/signal_parser.py not found")
    issues_found.append("signal_parser.py not found")

# Summary
print("\n" + "=" * 80)
if not issues_found:
    print("✅ ALL CHECKS PASSED - Your code is ready!")
else:
    print("❌ ISSUES FOUND - Need to fix these files:")
    for issue in issues_found:
        print(f"   • {issue}")
    print("\nCopy the output above and send it back for fixes.")
print("=" * 80)
