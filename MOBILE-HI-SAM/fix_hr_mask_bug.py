# Read the mask_decoder.py file
with open('Mobile_Hi_SAM/models/mask_decoder.py', 'r') as f:
    lines = f.readlines()

# Find the lines where hr_masks is sliced
for i, line in enumerate(lines):
    if 'hr_masks = hr_masks[:, mask_slice, :, :]' in line:
        print(f"Found hr_masks slicing at line {i+1}: {line.strip()}")
        # Comment it out or change it to not slice
        lines[i] = '        # hr_masks stays as-is (only 1 mask)\n'
        print(f"Changed to: {lines[i].strip()}")

# Write back
with open('Mobile_Hi_SAM/models/mask_decoder.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Fixed hr_masks slicing bug")
