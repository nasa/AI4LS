from synthetic.synthetic_generator import synthetic_generator
from synthetic.synthetic_generator import synthetic_generator_nonlinear
from synthetic.synthetic_generator import synthetic_generator_signed_nonlinear

N_mc = 1

for n in [5,10,50]:
    for i in range(N_mc):
        synthetic_df = synthetic_generator(n=100,d_layer=3,n_layer=[50,100,200],mu=0,sigma=1,n_causal=n)
        synthetic_name = 'full_fw_synthetic_sem_n_causal_'+str(n)+'_'+str(i)+'.pickle'
        synthetic_loc = 'data/synthetic/'+synthetic_name
        synthetic_df.to_pickle(synthetic_loc)

for n in [5,10,50]:
    for i in range(N_mc):
        synthetic_df = synthetic_generator_signed_nonlinear(n=100,d_layer=3,n_layer=[50,100,200],mu=0,sigma=1,n_causal=n)
        synthetic_name = 'signed_synthetic_sem_n_causal_'+str(n)+'_'+str(i)+'.pickle'
        synthetic_loc = 'data/synthetic/'+synthetic_name
        print(synthetic_loc)
        synthetic_df.to_pickle(synthetic_loc)
    
print("Generated synthetic datasets")
