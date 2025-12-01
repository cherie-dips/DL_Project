# Fix parameter name in mobile_encoder.py
with open('Mobile_Hi_SAM/models/mobile_encoder.py', 'r') as f:
    content = f.read()

# Replace ckpt_path with checkpoint_path
content = content.replace('ckpt_path: Optional[str] = None', 'checkpoint_path: Optional[str] = None')
content = content.replace('ckpt_path: str', 'checkpoint_path: str')
content = content.replace('if ckpt_path and os.path.exists(ckpt_path)', 'if checkpoint_path and os.path.exists(checkpoint_path)')
content = content.replace('self.load_checkpoint(ckpt_path)', 'self.load_checkpoint(checkpoint_path)')
content = content.replace('ckpt_path: str)', 'checkpoint_path: str)')
content = content.replace('"[MobileSAMEncoder] Loading checkpoint from {ckpt_path}"', '"[MobileSAMEncoder] Loading checkpoint from {checkpoint_path}"')
content = content.replace('ckpt = torch.load(ckpt_path,', 'ckpt = torch.load(checkpoint_path,')

with open('Mobile_Hi_SAM/models/mobile_encoder.py', 'w') as f:
    f.write(content)

print("✓ Fixed parameter name from ckpt_path to checkpoint_path")
