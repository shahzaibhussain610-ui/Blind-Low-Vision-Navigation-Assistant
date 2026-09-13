import os
base = r"C:\Users\Shahzaib Hussain\OneDrive\Desktop\Arbotrix Project 2"
files = []
for r, d, fs in os.walk(base):
    if '__pycache__' in r or '.git' in r or 'venv' in r or 'node_modules' in r:
        continue
    for f in fs:
        files.append(os.path.join(r, f))
files.sort()
for f in files:
    print(f)