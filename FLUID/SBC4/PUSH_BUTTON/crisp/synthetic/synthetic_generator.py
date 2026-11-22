import numpy as np
import pandas as pd
from scipy.stats import randint, bernoulli


def synthetic_generator(n=100, d_layer=3, n_layer=[50, 100, 200], mu=0, sigma=1, n_causal=10):
    assert d_layer == len(n_layer)

    max_weight = 4

    d = sum(n_layer)
    # determining childrens
    n_children = randint.rvs(0, 2 * n_causal, size=sum(n_layer[:-1]))
    children = []
    child_weight = []
    count = 0
    for i in range(d_layer - 1):
        count += n_layer[i]
        for j in range(n_layer[i]):
            temp = randint.rvs(count, d - 1, size=n_children[count + j - n_layer[i]])
            children.append(temp)
            child_weight.append(np.random.uniform(-max_weight, max_weight, n_children[count + j - n_layer[i]]))

    # generating features based on the SEM structure
    X = np.zeros((n, d + 1))
    for i in range(n):
        for j in range(n_layer[0]):  # Change order of for loops to fix wieghts across samples
            X[i][j] = np.random.uniform(-1, 1)
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

    # generate target variables only from the top layer
    i_causal = randint.rvs(0, n_layer[0], size=n_causal)
    w_causal = randint.rvs(-max_weight, max_weight, size=n_causal)
    for i in range(n):
        x = np.dot(X[i][i_causal], w_causal) + np.random.normal(mu, sigma)
        y = 1 / (1 + np.exp(-x))
        X[i][d] = 1 if y > 0.5 else 0

        # normalize features
    for j in range(d):
        col = X[:, j]
        col = (col - np.mean(col)) / np.std(col)
        X[:, j] = col

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
    ID = 1
    subj_id = [ID]
    env = bernoulli.rvs(0.6)
    env_split = [env]
    for i in range(1, n):
        if X[i][d] != X[i - 1][d]:
            ID += 1
            env = bernoulli.rvs(0.6)
        subj_id.append(ID)
        env_split.append(env)
    df["Subj_ID"] = subj_id
    df["env_split"] = env_split

    #     print(df.shape)
    return df


def synthetic_generator_nonlinear(n=100, d_layer=3, n_layer=[50, 100, 200], mu=0, sigma=1, n_causal=10):
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

    # generating features based on the SEM structure
    X = np.zeros((n, d + 1))
    for i in range(n):
        for j in range(n_layer[0]):  # Change order of for loops to fix wieghts across samples
            X[i][j] = np.random.uniform(-1, 1)
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

    # generate target variables only from the top layer
    i_causal = randint.rvs(0, n_layer[0], size=n_causal)
    w_causal = randint.rvs(-max_weight, max_weight, size=n_causal)
    for i in range(n):
        x = np.dot(np.multiply(X[i][i_causal], X[i][i_causal]), w_causal) + np.random.normal(mu, sigma)
        y = 1 / (1 + np.exp(-x))
        X[i][d] = 1 if y > 0.5 else 0

        # normalize features
    for j in range(d):
        col = X[:, j]
        col = (col - np.mean(col)) / np.std(col)
        X[:, j] = col

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
    ID = 1
    subj_id = [ID]
    env = bernoulli.rvs(0.6)
    env_split = [env]
    for i in range(1, n):
        if X[i][d] != X[i - 1][d]:
            ID += 1
            env = bernoulli.rvs(0.6)
        subj_id.append(ID)
        env_split.append(env)
    df["Subj_ID"] = subj_id
    df["env_split"] = env_split

    #     print(df.shape)
    return df


def synthetic_generator_signed_nonlinear(n=100, d_layer=3, n_layer=[50, 100, 200], mu=0, sigma=1, n_causal=10):
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

    # generating features based on the SEM structure
    X = np.zeros((n, d + 1))
    for i in range(n):
        for j in range(n_layer[0]):  # Change order of for loops to fix wieghts across samples
            X[i][j] = np.random.uniform(-1, 1)
            X[i][children[j]] += X[i][j] * child_weight[j]  # np.random.uniform(-2,2)
            # print('After:', X[i][children[j]])
        count = n_layer[0]
        for k in range(1, d_layer - 1):
            for j in range(count, count + n_layer[k]):
                X[i][j] += np.random.normal(mu, sigma)
                X[i][children[j]] += np.power(X[i][j], 3) * child_weight[j]  # *np.random.uniform(-2,2)
            count += n_layer[k]
        for j in range(count, count + n_layer[-1]):
            X[i][j] += np.random.normal(mu, sigma)

    # generate target variables only from the top layer
    i_causal = randint.rvs(0, n_layer[0], size=n_causal)
    w_causal = randint.rvs(-max_weight, max_weight, size=n_causal)
    for i in range(n):
        x = np.dot(np.power(X[i][i_causal], 3), w_causal) + np.random.normal(mu, sigma)
        y = 1 / (1 + np.exp(-x))
        X[i][d] = 1 if y > 0.5 else 0

        # normalize features
    for j in range(d):
        col = X[:, j]
        col = (col - np.mean(col)) / np.std(col)
        X[:, j] = col

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
    ID = 1
    subj_id = [ID]
    env = bernoulli.rvs(0.6)
    env_split = [env]
    for i in range(1, n):
        if X[i][d] != X[i - 1][d]:
            ID += 1
            env = bernoulli.rvs(0.6)
        subj_id.append(ID)
        env_split.append(env)
    df["Subj_ID"] = subj_id
    df["env_split"] = env_split

    #     print(df.shape)
    return df


def synthetic_generator_glm(n=100, d_layer=3, n_layer=[50, 100, 200], mu=0, sigma=1, n_causal=10):
    assert d_layer == len(n_layer)

    max_weight = 3

    d = sum(n_layer)
    # determining childrens
    n_children = randint.rvs(n_causal // 2, 3 * n_causal // 2, size=sum(n_layer[:-1]))
    children = []
    child_weight = []
    count = 0
    for i in range(d_layer - 1):
        count += n_layer[i]
        for j in range(n_layer[i]):
            temp = randint.rvs(count, d - 1, size=n_children[count + j - n_layer[i]])
            children.append(temp)
            child_weight.append(np.random.uniform(-max_weight, max_weight, n_children[count + j - n_layer[i]]))

    # generating features based on the SEM structure
    X = np.zeros((n, d + 1))
    for i in range(n):
        for j in range(n_layer[0]):
            X[i][j] = np.random.normal(mu, sigma)  ######## changed to normal rv from uniform rv
            X[i][children[j]] += X[i][j] * child_weight[j]
        count = n_layer[0]
        for k in range(1, d_layer - 1):
            for j in range(count, count + n_layer[k]):
                X[i][j] += np.random.normal(mu, sigma)
                X[i][children[j]] += X[i][j] * child_weight[j]
            count += n_layer[k]
        for j in range(count, count + n_layer[-1]):
            X[i][j] += np.random.normal(mu, sigma)

    # apply a nonlinear transformation
    for i in range(1, d_layer):
        for j in range(n_layer[i]):
            X[i][j] = X[i][j] ** 2

    # normalize features
    for j in range(d):
        col = X[:, j]
        col = (col - np.mean(col)) / np.std(col)
        X[:, j] = col

    # generate target variables only from the top layer
    i_causal = randint.rvs(0, n_layer[0], size=n_causal)
    w_causal = randint.rvs(-max_weight, max_weight, size=n_causal)
    for i in range(n):
        x = np.dot(X[i][i_causal], w_causal) + np.random.normal(mu, sigma)
        y = 1 / (1 + np.exp(-x))
        X[i][d] = 1 if y > 0.5 else 0

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
    ID = 1
    subj_id = [ID]
    env = bernoulli.rvs(0.6)
    env_split = [env]
    for i in range(1, n):
        if X[i][d] != X[i - 1][d]:
            ID += 1
            env = bernoulli.rvs(0.6)
        subj_id.append(ID)
        env_split.append(env)
    df["Subj_ID"] = subj_id
    df["env_split"] = env_split

    #     print(df.shape)
    return df
