import psutil

print psutil.virtual_memory().percent / 100
