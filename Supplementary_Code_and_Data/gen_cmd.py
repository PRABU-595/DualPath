import base64

with open(r'c:\Users\iampr\Desktop\MY PAPERS\DualPath-VLM\fix_data.py', 'r') as f:
    s = f.read()
    
script_b64 = base64.b64encode(s.encode('utf-8')).decode('ascii')

cmd = f"dualpath_env/Scripts/python.exe -c \"import base64; exec(base64.b64decode('{script_b64}').decode())\" && dualpath_env/Scripts/python.exe gpu_train.py --mode unified --skip_data"

with open('c:/Users/iampr/Desktop/MY PAPERS/DualPath-VLM/command_for_user.txt', 'w') as f:
    f.write(cmd)
