import os

dirs = ['templates', 'static', 'media']
for dir in dirs:
    os.makedirs(dir, exist_ok=True)
