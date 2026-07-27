import torch
from dataset import SpaceflightDataset, make_dataloaders
from model import SpaceflightCVAE
from losses import nb_nll_loss, kl_divergence, classification_loss

dataset = SpaceflightDataset("/home/jcasalet/nobackup/AI4LS/CVAE/DATA/subset_final.h6")
_, _, test_loader = make_dataloaders(dataset, batch_size=64)

#ckpt  = torch.load("/home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v2/best_model.pt",
ckpt  = torch.load("/home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v10/best_model.pt",
                  map_location="cpu", weights_only=False)

# read architecture from checkpoint instead of hardcoding
args = ckpt["args"]
print("Checkpoint args:", args)   # verify latent_dim and other settings

model = SpaceflightCVAE(
    n_genes=dataset.n_genes,
    n_strains=dataset.n_strains,
    n_sexes=dataset.n_sexes,
    n_studies=dataset.n_studies,
    n_tissues=dataset.n_tissues,
    n_euths=dataset.n_euths,
    latent_dim=args["latent_dim"],
    hidden_dims=[256, 128],
    dropout=0.0,
    grl_alpha=0.0,
)
model.load_state_dict(ckpt["model_state"])
model.eval()

total_recon = total_kl = total_cls = n = 0
with torch.no_grad():
    for batch in test_loader:
        out = model(
            batch["x"], batch["strain"], batch["sex"],
            batch["study"], batch["tissue"], batch["euth"], batch["flight"]
        )
        total_recon += nb_nll_loss(batch["x_raw"], out["log_r"], out["p"]).item()
        total_kl    += kl_divergence(out["mu"], out["log_var"]).item()
        total_cls   += classification_loss(
            out["flight_logit"], batch["flight"].float()
        ).item()
        n += 1

print(f"Test recon:  {total_recon/n:.4f}")
print(f"Test KL:     {total_kl/n:.4f}")
print(f"Test BCE:    {total_cls/n:.4f}")
print(f"Test total:  {(total_recon + total_kl + total_cls)/n:.4f}")
