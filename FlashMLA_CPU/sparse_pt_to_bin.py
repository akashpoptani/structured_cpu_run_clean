import torch
import os
import struct

def export_tensor(tensor, filename):
    t = tensor.cpu().contiguous()
    
    # Handle FP8 e4m3 types if they exist; otherwise view bf16 as int16
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    elif t.dtype == getattr(torch, "float8_e4m3fn", None): 
        t = t.view(torch.int8)
        
    data = t.numpy().tobytes()
    with open(filename, "wb") as f:
        f.write(data)
    print(f"  -> Saved {filename} ({len(data)} bytes) | Shape: {tensor.shape}")

def main():
    print("🚀 EXPORTING SPARSE TENSORS TO RAW BINARY...")
    
    d = torch.load("sparse_input.pt", map_location=torch.device('cpu'), weights_only=False)
    
    out_dir = "sparse_bins"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    export_tensor(d['q'], f"{out_dir}/q.bin")
    
    # 🔥 FLATTEN KV CACHE: [2300, 64, 1, 656] -> [147200, 656] 🔥
    # We do NOT slice it. We keep the raw 656-byte tokens.
    k_cache_flat = d['k_cache'].contiguous().view(-1, 656)
    export_tensor(k_cache_flat, f"{out_dir}/k_cache.bin")
    
    # Flatten the active indices (Shape: [1, 1, 2048] -> [2048])
    indices = d['indices_in_kvcache'].to(torch.int32).reshape(-1)
    export_tensor(indices, f"{out_dir}/indices.bin")
    
    B, _, H, D = d['q'].shape
    num_indices = indices.numel() # This will be exactly 2048
    
    scale = d.get('softmax_scale', None)
    if scale is None:
        scale = 1.0
        print("  -> softmax_scale is None (Fused into weights). Setting scale to 1.0.")
    
    # Save Config (B, H, D, NumIndices, Scale)
    with open(f"{out_dir}/config.bin", "wb") as f:
        config = struct.pack("iiiif", B, H, D, num_indices, scale)
        f.write(config)
    print(f"  -> Saved config.bin (Batch={B}, Heads={H}, Dim={D}, ActiveTokens={num_indices}, Scale={scale})")

    print("  Loading sparse_output.pt...")
    out = torch.load("sparse_output.pt", map_location=torch.device('cpu'), weights_only=False)
    export_tensor(out['out'].to(torch.float32), f"{out_dir}/out_golden.bin") 
    
    print("\n✅ Sparse Data Export Complete. Ready for C++.")

if __name__ == "__main__":
    main()