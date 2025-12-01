# Fix the embed_dim in mobile_encoder.py
with open('Mobile_Hi_SAM/models/mobile_encoder.py', 'r') as f:
    content = f.read()

# Replace embed_dim=768 with embed_dim=320
content = content.replace('embed_dim=768,', 'embed_dim=320,')

with open('Mobile_Hi_SAM/models/mobile_encoder.py', 'w') as f:
    f.write(content)

print("✓ Fixed embed_dim from 768 to 320")
