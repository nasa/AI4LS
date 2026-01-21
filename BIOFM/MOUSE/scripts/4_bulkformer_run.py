import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist

try:
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("⚠ safetensors not installed. Install with: pip install safetensors")

from model.bulkformer import BulkFormer, model_params  


# JC
# pip install torch_geometric performer_pytorch

# ===============================================================
# 1. Dataset with MLM-style masking
# ===============================================================
class BulkMLMDataset(Dataset):
    def __init__(self, X_np, mask_ratio=0.15, mask_token=-10):
        self.X = X_np.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        # mask 15% of genes
        g = x.shape[0]
        num_mask = int(g * self.mask_ratio)

        mask_idx = np.random.choice(g, num_mask, replace=False)
        x_masked = x.copy()
        x_masked[mask_idx] = self.mask_token

        return (
            torch.tensor(x_masked),
            torch.tensor(x),
            torch.tensor(mask_idx)
        )


# ===============================================================
# 2. Main training function
# ===============================================================
def main():
    script_start = time.time()
   
    # JC
    # Set environment variables before init_process_group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Any free port
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    # /JC

    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Only rank 0 prints
    is_main = rank == 0
    
    if is_main:
        print("\n" + "="*70)
        print(f"BulkFormer Training with DDP ({world_size} GPUs)")
        print("="*70)
        print(f"\n[SETUP] Rank: {rank}, Device: {device}, World size: {world_size}")

    # -----------------------------------------------------------
    # Load expression data (all ranks load the same data)
    # -----------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading expression data...")
    t_start = time.time()
    X_df = pd.read_parquet("./data/archs4/processed_short_proteins/train_expr_logtpm_short.parquet")
    X_df = X_df.T
    X_np = X_df.values   # shape [N, G]
    num_samples, num_genes = X_np.shape
    t_load = time.time() - t_start
    if is_main:
        print(f"  ✓ Shape: {X_np.shape}, Time: {t_load:.2f}s")

    # -----------------------------------------------------------
    # Load ESM2 gene identity embeddings
    # -----------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading ESM2 gene embeddings...")
    t_start = time.time()
    # esm3_sm_open_v1_gene_embeddings.pt  esm3_sm_open_v1_gene_order.csv
    esm2_data = torch.load("./data/embeddings/esm3_sm_open_v1_gene_embeddings.pt", map_location="cpu")
    esm2_raw = esm2_data['embeddings'].float().to(device)
    t_load = time.time() - t_start
    if is_main:
        print(f"  ✓ Shape: {esm2_raw.shape}, Time: {t_load:.2f}s")

    # -----------------------------------------------------------
    # Load gene–gene graph
    # -----------------------------------------------------------
    if is_main:
        print("\n[DATA] Loading gene graph edges...")
    t_start = time.time()
    edge_index = torch.load("./graph/edge_index_top20.pt", map_location="cpu").long().to(device)
    t_load = time.time() - t_start
    if is_main:
        print(f"  ✓ Shape: {edge_index.shape}, Time: {t_load:.2f}s")

    # -----------------------------------------------------------
    # Configure model parameters dynamically
    # -----------------------------------------------------------
    if is_main:
        print("\n[MODEL] Configuring model parameters...")
    model_params["graph"] = edge_index
    model_params["gene_emb"] = esm2_raw
    model_params["gene_length"] = num_genes

    # -----------------------------------------------------------
    # Build model and wrap with DDP
    # -----------------------------------------------------------
    if is_main:
        print("\n[MODEL] Initializing BulkFormer...")
        print("\n[MODEL PARAMS]")
        for k, v in model_params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: tensor{tuple(v.shape)}")
            else:
                print(f"  {k}: {v}")
    print()
    t_start = time.time()
    model = BulkFormer(**model_params).to(device)
    # Wrap model with DDP (find_unused_parameters=True allows some params to not be used in loss)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    t_init = time.time() - t_start
    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  ✓ Parameters: {total_params:,}, Init time: {t_init:.2f}s")

    # -----------------------------------------------------------
    # Prepare dataset with DistributedSampler
    # -----------------------------------------------------------
    if is_main:
        print("\n[DATA] Creating dataset and dataloader...")

    t_start = time.time()
    dataset = BulkMLMDataset(X_np)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    
    # Load validation data
    X_val_df = pd.read_parquet("./data/archs4/processed_short_proteins/val_expr_logtpm_short.parquet")
    X_val_df = X_val_df.T
    X_val_np = X_val_df.values
    val_dataset = BulkMLMDataset(X_val_np)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, sampler=val_sampler)
    
    t_prep = time.time() - t_start
    if is_main:
        print(f"  ✓ Train: {len(dataset)} samples, {len(loader)} batches")
        print(f"  ✓ Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        print(f"  ✓ Batch size: {loader.batch_size}, Time: {t_prep:.2f}s")

    # -----------------------------------------------------------
    # Optimizer & loss
    # -----------------------------------------------------------
    if is_main:
        print("\n[OPTIM] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    if is_main:
        print(f"  ✓ AdamW optimizer ready (lr=1e-4)")

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    if is_main:
        print("\n[TRAIN] Starting training loop...")
        print("="*70 + "\n")
    epochs = 5

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        sampler.set_epoch(epoch)  # Important for DistributedSampler
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_masked, x_true, mask_idx) in enumerate(loader):
            batch_start = time.time()

             # Heartbeat every 30 seconds (rank 0 only)
            if is_main:
                now = time.time()
                if not hasattr(main, "_last_beat") or now - main._last_beat > 30:
                    main._last_beat = now
                    pct = 100.0 * batch_idx / len(loader)
                    elapsed = now - script_start
                    print(f"[HEARTBEAT] Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx}/{len(loader)} ({pct:.2f}%) | "
                        f"Elapsed: {elapsed/60:.1f} min")
            
            x_masked = x_masked.to(device)
            x_true = x_true.to(device)

            # Forward pass
            pred = model(x_masked)    # shape [B, G]

            # Compute reconstruction loss *only at masked positions*
            loss_list = []
            for i in range(len(mask_idx)):
                idxs = mask_idx[i]
                loss_list.append(mse_loss(pred[i, idxs], x_true[i, idxs]))

            loss = torch.stack(loss_list).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            num_batches += 1
            batch_time = time.time() - batch_start
            
            # Print per-batch progress every 25% of batches (rank 0 only)
            if is_main and (batch_idx + 1) % max(1, len(loader) // 4) == 0:
                avg_so_far = running_loss / num_batches
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {loss_val:.6f} | Avg: {avg_so_far:.6f} | Time: {batch_time:.3f}s")

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = running_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_masked, x_true, mask_idx in val_loader:
                x_masked = x_masked.to(device)
                x_true = x_true.to(device)
                pred = model(x_masked)
                
                loss_list = []
                for i in range(len(mask_idx)):
                    idxs = mask_idx[i]
                    loss_list.append(mse_loss(pred[i, idxs], x_true[i, idxs]))
                
                val_loss += torch.stack(loss_list).mean().item()
                val_batches += 1
        
        val_avg_loss = val_loss / val_batches
        
        if is_main:
            print(f"\n  ╔════════════════════════════════════════════╗")
            print(f"  ║ Epoch {epoch+1}/{epochs}")
            print(f"  ║ Train Loss: {epoch_avg_loss:.6f}")
            print(f"  ║ Val Loss:   {val_avg_loss:.6f}")
            print(f"  ║ Time: {epoch_time:.2f}s")
            print(f"  ╚════════════════════════════════════════════╝\n")

    # -----------------------------------------------------------
    # Save checkpoint (rank 0 only)
    # -----------------------------------------------------------
    if is_main:
        print("\n[SAVE] Saving model checkpoints...")
        save_start = time.time()
        
        # Create checkpoint directory
        ckpt_dir = Path(f"bulkformer_checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        
        # Get the underlying model (unwrap DDP)
        model_to_save = model.module if isinstance(model, DDP) else model
        
        # 1. PyTorch native format (.pt)
        pt_path = ckpt_dir / f"epoch_{epoch}.pt"
        torch.save(model_to_save.state_dict(), pt_path)
        print(f"  ✓ PyTorch format: {pt_path}")
        
        # 2. Hugging Face SafeTensors format (.safetensors) - recommended for HF Hub
        if HAS_SAFETENSORS:
            # Convert state_dict to CPU tensors for safetensors
            state_dict_cpu = {k: v.cpu() for k, v in model_to_save.state_dict().items()}
            safetensors_path = ckpt_dir / f"epoch_{epoch}.safetensors"
            save_safetensors(state_dict_cpu, str(safetensors_path))
            print(f"  ✓ SafeTensors format (HF-compatible): {safetensors_path}")
        
        # 3. Model config JSON for HF Hub
        config = {
            "model_type": "bulkformer",
            "num_genes": num_genes,
            "dim": model_params.get("dim", 320),
            "gb_repeat": model_params.get("gb_repeat", 2),
            "bins": model_params.get("bins", 10),
            "bin_head": model_params.get("bin_head", 2),
            "full_head": model_params.get("full_head", 2),
            "p_repeat": model_params.get("p_repeat", 1),
            "training_epoch": epoch,
            "final_loss": epoch_avg_loss
        }
        # JC
        print("config: ", config)
        config_path = ckpt_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Config: {config_path}")
        
        save_time = time.time() - save_start
        print(f"\n  ✓ All checkpoints saved in {ckpt_dir}/, Time: {save_time:.2f}s")
        print(f"\n  📦 To upload to Hugging Face Hub:")
        print(f"     huggingface-cli upload <username>/<repo-name> bulkformer_checkpoints/")
        
        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        total_time = time.time() - script_start
        print("\n" + "="*70)
        print(f"Training completed!")
        print(f"  Total script time: {total_time:.2f}s ({total_time/60:.1f}m)")
        print("="*70 + "\n")
    
    # Cleanup
    dist.destroy_process_group()


# ===============================================================
# Run
# ===============================================================
if __name__ == "__main__":
    main()
