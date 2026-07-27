# Confirm hidden_dims is a CLI arg in both scripts
grep "hidden_dims" pretrain_archs4.py train.py hpsearch.py

# Confirm hidden_dims is saved in the checkpoint after pretraining
python -c "
import torch
ckpt = torch.load('checkpoints/pretrain/.../pretrain_best.pt', map_location='cpu')
print('latent_dim: ', ckpt.get('latent_dim'))
print('hidden_dims:', ckpt.get('hidden_dims'))
print('label_encoders keys:', list(ckpt.get('label_encoders', {}).keys()))
"
