print("Starting imports...")

try:
    print("1. Importing sys...")
    import sys
    print("   ✓ sys imported")
    
    print("2. Importing torch...")
    import torch
    print(f"   ✓ torch imported: {torch.__version__}")
    
    print("3. Importing torchvision...")
    import torchvision
    print(f"   ✓ torchvision imported: {torchvision.__version__}")
    
    print("4. Importing PIL...")
    from PIL import Image
    print("   ✓ PIL imported")
    
    print("5. Importing numpy...")
    import numpy as np
    print(f"   ✓ numpy imported: {np.__version__}")
    
    print("6. Importing models directly (we're already in Mobile_Hi_SAM)...")
    from models import mobile_hisam_model
    print("   ✓ mobile_hisam_model imported")
    
    print("7. Importing adapters...")
    from adapters import adapter
    print("   ✓ adapter imported")
    
    print("\n✓ All imports successful!")
    
except Exception as e:
    print(f"\n✗ Error at import: {e}")
    import traceback
    traceback.print_exc()
