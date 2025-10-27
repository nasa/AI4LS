import numpy as np
import pandas as pd
from scipy.stats import randint, bernoulli


def synthetic_generator_counfounder(n=100, d_layer=3, n_layer=[50, 100, 200], mu=0, sigma=1, n_causal=10, n_env=2, n_unc=0, output_data_type="real-valued"):
    assert d_layer == len(n_layer)
    assert n_causal%2==0
    
    max_weight = 4

    d = sum(n_layer)
    # determining children (make sure the first node has at least one child)
    n_children = np.array(randint.rvs(1, 2 * n_causal, size=1).tolist() + randint.rvs(0, 2 * n_causal, size=sum(n_layer[:-1])-1).tolist() )
    children = []
    child_weight = []
    count = 0
    for i in range(d_layer - 1):
        count += n_layer[i]
        for j in range(n_layer[i]):
            temp = randint.rvs(count, d - 1, size=n_children[count + j - n_layer[i]])  
            children.append(temp)
            child_weight.append(np.random.uniform(-max_weight, max_weight, n_children[count + j - n_layer[i]]))

    
    # generate target variables from grandfather 0, n_causal/2-1 grandfathers, 
    # the first child of grandfather 0, n_causal/2-1 parents
    child_of_grandad0 = np.random.choice(children[0], size=1)
    i_causal = np.array([0] + 
                        list(np.random.choice(np.arange(1, n_layer[0]), size=int(n_causal/2), replace=False)) +
                        list(child_of_grandad0) +
                        list(np.random.choice(np.arange(n_layer[0], n_layer[0]+n_layer[1]), size=int(n_causal/2-1), replace=False))
                        )
    w_causal = randint.rvs(-max_weight, max_weight, size=n_causal+1)
    
    
    # Let's iterate over the environments
    n = int(n/n_env)
    X_tot = []
    for env in range(n_env):
        # generate environment variance
        environment_std = np.random.normal(0,3)
        
        # generating features based on the SEM structure
        X = np.zeros((n, d + 1))
        for i in range(n):
            for j in range(n_layer[0]):  # Change order of for loops to fix wieghts across samples
                ID = j
                X[i][j] = np.random.uniform(low=-1, high=1)*( int(ID in i_causal) * environment_std +
                                                       (1-int(ID in i_causal))*1)                
                X[i][children[j]] += X[i][j] * child_weight[j]  # np.random.uniform(-2,2)
            count = n_layer[0]
            for k in range(1, d_layer - 1):
                for j in range(count, count + n_layer[k]):
                    ID = j
                    X[i][j] += np.random.normal(mu, sigma) * ( int(ID in i_causal) * environment_std +
                                                                (1-int(ID in i_causal))*1
                                                             )
                    X[i][children[j]] += X[i][j] * child_weight[j]  # *np.random.uniform(-2,2)
                count += n_layer[k]
            for j in range(count, count + n_layer[-1]):
                X[i][j] += np.random.normal(mu, sigma)

        for i in range(n):
            x = np.dot(X[i][i_causal], w_causal) + np.random.normal(mu, sigma)#*environment_std

            if output_data_type == "real-valued":
                X[i][d] = x
            else:
                y = 1 / (1 + np.exp(-x))
                X[i][d] = 1 if y > 0.5 else 0

            # normalize features
        for j in range(d):
            col = X[:, j]
            col = (col - np.mean(col)) / np.std(col)
            X[:, j] = col
        
        X_tot.append(X)
        
    X = np.vstack(X_tot)
        
    # generating a dataframe column names
    ii = 0
    columns = []
    
    for i in range(d):
        if i==0:
            columns.append("Confounder")
        elif i in i_causal:
            columns.append("Causal_" + str(ii))
            ii += 1
        else:
            columns.append("Non_causal_" + str(i - ii))
            
    columns.append("Target")
    df = pd.DataFrame(data=X, columns=columns)

    # adding the subject id and environment splits to the dataframe
    
    env_split = []
    for i in range(n_env):
        env_split += [i]*n
    subj_id = np.arange(n*n_env)
    df["Subj_ID"] = subj_id
    df["env_split"] = env_split
    
    # add the uncorrelated features
    mean_std = np.std(X[:,:d])
    
    for i in range(n_unc):
        df["Uncorrelated_"+str(i)] = np.random.randn(n*n_env)*np.random.rand(1)*2*mean_std

    print('Dimension is', df.shape)
    return df

def synthetic_generator_nonlinear_environments(n=100, d_layer=3, n_layer=[50, 100, 200], mu=0, sigma=1, n_causal=10, n_unc=0, n_env=2):
    assert d_layer == len(n_layer)

    max_weight = 10

    d = sum(n_layer)
    # determining childrens
    n_children = randint.rvs(0, 10, size=sum(n_layer[:-1]))
    children = []
    child_weight = []
    count = 0
    for i in range(d_layer - 1):
        count += n_layer[i]
        for j in range(n_layer[i]):
            temp = randint.rvs(count, d - 1, size=n_children[count + j - n_layer[i]])
            children.append(temp)
            child_weight.append(np.random.uniform(-max_weight, max_weight, n_children[count + j - n_layer[i]]))

    # generate target variables only from the top layer
    i_causal = np.random.choice(np.arange(n_layer[0]), size=n_causal, replace=False)
    randint.rvs(0, n_layer[0], size=n_causal)
    w_causal = randint.rvs(-max_weight, max_weight, size=n_causal)
    
    # Let's iterate over the environments
    n = int(n/n_env)
    X_tot = []
    for env in range(n_env):
        # generate environment variance
        environment_std = np.random.normal(0,3)
        
        # generating features based on the SEM structure
        X = np.zeros((n, d + 1))
        for i in range(n):
            for j in range(n_layer[0]):  # Change order of for loops to fix wieghts across samples
                ID = j
                X[i][j] = np.random.uniform(-1, 1)*( int(ID in i_causal) * environment_std +
                                                       (1-int(ID in i_causal))*1)                
                X[i][children[j]] += X[i][j] * child_weight[j]  # np.random.uniform(-2,2)
                # print('After:', X[i][children[j]])
            count = n_layer[0]
            for k in range(1, d_layer - 1):
                for j in range(count, count + n_layer[k]):
                    X[i][j] += np.random.normal(mu, sigma)
                    X[i][children[j]] += X[i][j] * child_weight[j]  # *np.random.uniform(-2,2)
                count += n_layer[k]
            for j in range(count, count + n_layer[-1]):
                X[i][j] += np.random.normal(mu, sigma)


        for i in range(n):
            x = np.dot(np.multiply(X[i][i_causal], X[i][i_causal]), w_causal) + np.random.normal(mu, sigma)*environment_std
            y = 1 / (1 + np.exp(-x))
            X[i][d] = 1 if y > 0.5 else 0

            # normalize features
        for j in range(d):
            col = X[:, j]
            col = (col - np.mean(col)) / np.std(col)
            X[:, j] = col
            
        X_tot.append(X)
        
    X = np.vstack(X_tot)

    # generating a dataframe column names
    ii = 0
    columns = []
    for i in range(d):
        if i in i_causal:
            columns.append("Causal_" + str(ii))
            ii += 1
        else:
            columns.append("Non_causal_" + str(i - ii))
    columns.append("Target")
    df = pd.DataFrame(data=X, columns=columns)

    # adding the subject id and environment splits to the dataframe
    env_split = []
    for i in range(n_env):
        env_split += [i]*n
    subj_id = np.arange(n*n_env)
    df["Subj_ID"] = subj_id
    df["env_split"] = env_split

    
    # add the uncorrelated features
    mean_std = np.std(X[:,:d])
    
    for i in range(n_unc):
        df["Uncorrelated_"+str(i)] = np.random.randn(n*n_env)*np.random.rand(1)*2*mean_std

    return df

def main():
    df = synthetic_generator_nonlinear_environments(n=100, d_layer=3, n_layer=[50, 100, 200], mu=0, sigma=1, n_causal=10, n_unc=0, n_env=2)

    df.to_pickle('/tmp/bob.pkl')

if __name__ == "__main__":
    main()