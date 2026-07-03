"""Export trained model to binary format for Go inference."""
import sys; sys.path.insert(0,'training')
import torch, os, math
import numpy as np
from bsm import BinaryStateMachine

def export_model(checkpoint_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    model = BinaryStateMachine(vocab_size=4096, hidden_dim=128, dec_dim=32)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model._clamp_params()
    
    # Binarize all parameters
    for name in ['T', 'W1', 'W2', 'D_dec', 'B_dec']:
        p = getattr(model, name).data
        bits = torch.where(p >= 0, 1, -1).to(torch.int8).cpu().numpy()
        np.save(f'{output_dir}/{name}.npy', bits)
        print(f"{name}: {bits.shape} -> {output_dir}/{name}.npy")
    
    # Total size
    total_bytes = sum(os.path.getsize(f'{output_dir}/{name}.npy') for name in ['T', 'W1', 'W2', 'D_dec', 'B_dec'])
    print(f"Total: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print("Export complete.")

if __name__ == '__main__':
    import sys as _sys
    ckpt = _sys.argv[1] if len(_sys.argv) > 1 else 'checkpoints/bsm_d128_v26/v26_25000.pt'
    out = _sys.argv[2] if len(_sys.argv) > 2 else 'export/v26'
    export_model(ckpt, out)
