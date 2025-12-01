# Read the file
with open('Mobile_Hi_SAM/models/mobile_hisam_model.py', 'r') as f:
    content = f.read()

# Fix the bug: remove .unsqueeze(0) from sparse_prompt_embeddings
content = content.replace(
    'sparse_prompt_embeddings=sparse_prompt_embs.unsqueeze(0),',
    'sparse_prompt_embeddings=sparse_prompt_embs,'
)

# Write back
with open('Mobile_Hi_SAM/models/mobile_hisam_model.py', 'w') as f:
    f.write(content)

print("✓ Fixed the dimension bug in mobile_hisam_model.py")
