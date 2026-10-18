import torch
import os
import struct

def export_tensor(tensor, filename):
    # Move to CPU first
    t = tensor.cpu().contiguous()
    
    # FIX: NumPy doesn't support BF16. 
    # We view it as int16 to preserve the exact bits without changing data.
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
        
    data = t.numpy().tobytes()
    with open(filename, "wb") as f:
        f.write(data)
    print(f"  -> Saved {filename} ({len(data)} bytes) | Shape: {tensor.shape}")

def main():
    print("🚀 EXPORTING TENSORS TO RAW BINARY...")
    
    # 1. Load Input (Force CPU map to avoid CUDA errors)
    print("  Loading dense_input.pt...")
    d = torch.load("dense_input.pt", map_location=torch.device('cpu'), weights_only=True)
    
    if not os.path.exists("bins"): os.makedirs("bins")
    
    export_tensor(d['q'], "bins/q.bin")
    export_tensor(d['k_cache'], "bins/k_cache.bin")
    export_tensor(d['block_table'].to(torch.int32), "bins/block_table.bin") 
    export_tensor(d['cache_seqlens'].to(torch.int32), "bins/seqlens.bin")
    
    # Save Scalars as a tiny config file
    with open("bins/config.bin", "wb") as f:
        # Batch, Heads, Dim, BlockSize, Scale(float)
        B, _, H, D = d['q'].shape
        scale = d['softmax_scale']
        # Pack ints and float
        config = struct.pack("iiiif", B, H, D, 64, scale)
        f.write(config)
    print(f"  -> Saved config.bin (Batch={B}, Heads={H}, Dim={D}, Scale={scale})")

    # 2. Load Output (For Verification)
    print("  Loading dense_output.pt...")
    out = torch.load("dense_output.pt", map_location=torch.device('cpu'), weights_only=True)
    
    # Save Output as FP32 (float) for easier C++ error checking
    # (We want high precision for the ground truth)
    export_tensor(out['out'].to(torch.float32), "bins/out_golden.bin") 
    
    print("\n✅ Data Export Complete. Ready for C++.")

if __name__ == "__main__":
    main()