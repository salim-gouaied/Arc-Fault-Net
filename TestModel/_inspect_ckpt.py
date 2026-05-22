import torch, json, sys
ckpt = torch.load(sys.argv[1], map_location='cpu')
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
    sd = ckpt['state_dict']
else:
    sd = ckpt
for k, v in sorted(sd.items()):
    print(f"{k:60s}  {str(list(v.shape)):30s}")
