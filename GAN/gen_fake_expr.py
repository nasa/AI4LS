import tensorflow as tf
import datetime
import numpy as np
from sklearn.decomposition import PCA
import operator
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from numpy import linalg as LA
import json
import sys
from scipy.stats import zscore


tfk = tf.keras
tfkl = tf.keras.layers
tf.compat.v1.enable_eager_execution()

# ---------------------
# DATA UTILITIES
# ---------------------
def standardize(x, mean=None, std=None, kappa=1):
    """
    Shape x: (nb_samples, nb_vars)
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / (std * kappa)


def split_train_test(x, train_rate=0.75, seed=0):
    """
    Split data into a train and a test sets
    :param train_rate: percentage of training samples
    :return: x_train, x_test
    """
    nb_samples = x.shape[0]
    split_point = int(train_rate * nb_samples)
    x_train = x[:split_point]
    x_test = x[split_point:]
    return x_train, x_test

# ---------------------
# CORRELATION UTILITIES
# ---------------------

def my_correlation(x, y):
    #from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    return pearsonr(list(x.flatten()), list(y.flatten()))[0]


def pearson_correlation(x, y, kappa=1):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :return: Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a, kappa=1):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        if not np.any(a_std):
            return np.zeros(a.shape)
        return (a - a_off) / (a_std * kappa)
    assert x.shape[0] == y.shape[0]
    x_ = standardize(x, kappa)
    y_ = standardize(y, kappa)
    return np.dot(x_.T, y_) / x.shape[0]

def cosine_similarity(x, y):
    """
    Computes cosine similarity between vectors x and y
    :param x: Array of numbers. Shape=(n,)
    :param y: Array of numbers. Shape=(n,)
    :return: cosine similarity between vectors
    """
    return np.dot(x.T, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def upper_diag_list(m_):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    :param m_: numpy array of float. Shape=(N, N)
    :return: list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # upper-diagonal matrix
    tril = np.zeros_like(m_) + np.nan
    tril = np.tril(tril)
    m += tril
    m = np.ravel(m)
    m = m[~np.isnan(m)]
    return m


def correlations_list(x, y, corr_fn=pearson_correlation, kappa=1):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :param corr_fn: correlation function taking x and y as inputs
    """
    #corr = pearson_correlation(x,y, kappa)
    corr = pearson_correlation(x, y, kappa)
    result = upper_diag_list(corr)
    return result

def gamma_coef(x, y, kappa=1):
    """
    Compute gamma coefficients for two given expression matrices
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z)
    """
    print('x shape = ', str(x.shape), 'y shape = ', str(y.shape))
    dists_x = 1 - correlations_list(x, x, kappa)
    dists_y = 1 - correlations_list(y, y, kappa)
    #gamma_dx_dy = cosine_similarity(dists_x, dists_y)
    gamma_dx_dy = pearson_correlation(dists_x, dists_y, kappa)
    return gamma_dx_dy


# ------------------
# LIMIT GPU USAGE
# ------------------

def limit_gpu(gpu_idx=0, mem=2 * 1024):
    """
    Limits gpu usage
    :param gpu_idx: Use this gpu
    :param mem: Maximum memory in bytes
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # Use a single gpu
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')

            # Limit memory
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem)])  # 2 GB
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


# ------------------
# WGAN-GP
# ------------------

def make_generator(x_dim, vocab_sizes, nb_numeric, h_dims=None, z_dim=10):
    """
    Make generator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :param z_dim: Number of input units
    :return: generator
    """
    # Define inputs
    z = tfkl.Input((z_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.float32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []
    total_emb_dim = 0

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)
        total_emb_dim += emb_dim
    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    embeddings = tfkl.Concatenate(axis=-1)([num, embeddings])
    total_emb_dim += nb_numeric

    def make_generator_emb(x_dim, emb_dim, h_dims=None, z_dim=10):
        if h_dims is None:
            h_dims = [256, 256]

        z = tfkl.Input((z_dim,))
        t_emb = tfkl.Input((emb_dim,), dtype=tf.float32)
        h = tfkl.Concatenate(axis=-1)([z, t_emb])
        for d in h_dims:
            h = tfkl.Dense(d)(h)
            h = tfkl.ReLU()(h)
        h = tfkl.Dense(x_dim)(h)
        model = tfk.Model(inputs=[z, t_emb], outputs=h)
        return model

    gen_emb = make_generator_emb(x_dim=x_dim,
                                 emb_dim=total_emb_dim,
                                 h_dims=h_dims,
                                 z_dim=z_dim)
    model = tfk.Model(inputs=[z, cat, num], outputs=gen_emb([z, embeddings]))
    model.summary()
    return model


def make_discriminator(x_dim, vocab_sizes, nb_numeric, h_dims=None):
    """
    Make discriminator
    :param x_dim: Number of genes
    :param vocab_sizes: List of ints. Size of vocabulary for each categorical covariate
    :param nb_numeric: Number of numerical covariates
    :param h_dims: Number of units for each hidden layer
    :return: discriminator
    """
    if h_dims is None:
        h_dims = [256, 256]

    x = tfkl.Input((x_dim,))
    nb_categoric = len(vocab_sizes)
    cat = tfkl.Input((nb_categoric,), dtype=tf.float32)
    num = tfkl.Input((nb_numeric,), dtype=tf.float32)

    embed_cats = []

    for n, vs in enumerate(vocab_sizes):
        emb_dim = int(vs ** 0.5) + 1  # Rule of thumb
        c_emb = tfkl.Embedding(input_dim=vs,  # Vocabulary size
                               output_dim=emb_dim  # Embedding size
                               )(cat[:, n])
        embed_cats.append(c_emb)

    if nb_categoric == 1:
        embeddings = embed_cats[0]
    else:
        embeddings = tfkl.Concatenate(axis=-1)(embed_cats)
    h = tfkl.Concatenate(axis=-1)([x, num, embeddings])
    for d in h_dims:
        h = tfkl.Dense(d)(h)
        h = tfkl.ReLU()(h)
    h = tfkl.Dense(1)(h)
    model = tfk.Model(inputs=[x, cat, num], outputs=h)
    return model


def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein loss
    """
    return tf.reduce_mean(y_true * y_pred)


def generator_loss(fake_output):
    """
    Generator loss
    """
    return wasserstein_loss(-tf.ones_like(fake_output), fake_output)


def gradient_penalty(f, real_output, fake_output):
    """
    Gradient penalty of WGAN-GP
    :param f: discriminator function without sample covariates as input
    :param real_output: real data
    :param fake_output: fake data
    :return: gradient penalty
    """
    alpha = tf.random.uniform([real_output.shape[0], 1], 0., 1.)
    diff = fake_output - real_output
    inter = real_output + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))  # real_output
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp


def discriminator_loss(real_output, fake_output):
    """
    Critic loss
    """
    real_loss = wasserstein_loss(-tf.ones_like(real_output), real_output)
    fake_loss = wasserstein_loss(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_disc(x, z, cc, nc, gen, disc, disc_opt, grad_penalty_weight=10, p_aug=0, norm_scale=0.5):
    """
    Train critic
    :param x: Batch of expression data
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param disc_opt: Critic optimizer
    :param grad_penalty_weight: Weight for the gradient penalty
    :return: Critic loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as disc_tape:
        # Generator forward pass
        x_gen = gen([z, cc, nc], training=False)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, nb_genes)
        x = x + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))

        # Forward pass on discriminator
        disc_out = disc([x_gen, cc, nc], training=True)
        disc_real = disc([x, cc, nc], training=True)

        # Compute losses
        disc_loss = discriminator_loss(disc_real, disc_out) \
                    + grad_penalty_weight * gradient_penalty(lambda x: disc([x, cc, nc], training=True), x, x_gen)

    disc_grad = disc_tape.gradient(disc_loss, disc.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grad, disc.trainable_variables))

    return disc_loss


@tf.function
def train_gen(z, cc, nc, gen, disc, gen_opt, p_aug=0, norm_scale=1):
    """
    Train generator
    :param z: Batch of latent variables
    :param cc: Batch of categorical covariates
    :param nc: Batch of numerical covariates
    :param gen: Generator
    :param disc: Critic
    :param gen_opt: Generator optimiser
    :return: Generator loss
    """
    bs = z.shape[0]
    nb_genes = gen.output.shape[-1]
    augs = np.random.binomial(1, p_aug, bs)

    with tf.GradientTape() as gen_tape:
        # Generator forward pass
        x_gen = gen([z, cc, nc], training=True)

        # Perform augmentations
        x_gen = x_gen + augs[:, None] * np.random.normal(0, norm_scale, (bs, nb_genes))

        # Forward pass on discriminator
        disc_out = disc([x_gen, cc, nc], training=False)

        # Compute losses
        gen_loss = generator_loss(disc_out)

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))

    return gen_loss


def train(dataset, cat_covs, num_covs, z_dim, epochs, batch_size, gen, disc, score_fn, save_fn,
          gen_opt=None, disc_opt=None, nb_critic=5, verbose=True, checkpoint_dir=None,
          log_dir=None, patience=10, p_aug=0, norm_scale=0.5, gamma_list=None, kappa=1, tf_version=1):
    """
    Train model
    :param dataset: Numpy matrix with data. Shape=(nb_samples, nb_genes)
    :param cat_covs: Categorical covariates. Shape=(nb_samples, nb_cat_covs)
    :param num_covs: Numerical covariates. Shape=(nb_samples, nb_num_covs)
    :param z_dim: Int. Latent dimension
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param gen: Generator model
    :param disc: Critic model
    :param gen_opt: Generator optimiser
    :param disc_opt: Critic optimiser
    :param score_fn: Function that computes the score: Generator => score.
    :param save_fn:  Function that saves the model.
    :param nb_critic: Number of critic updates for each generator update
    :param verbose: Print details
    :param checkpoint_dir: Where to save checkpoints
    :param log_dir: Where to save logs
    :param patience: Number of epochs without improving after which the training is halted
    """
    # Optimizers
    if gen_opt is None:
        gen_opt = tfk.optimizers.RMSprop(5e-4)
    if disc_opt is None:
        disc_opt = tfk.optimizers.RMSprop(5e-4)

    # Set up logs and checkpoints
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    gen_log_dir = log_dir + current_time + '/gen'
    disc_log_dir = log_dir + current_time + '/disc'

    if tf_version == 1:
        gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
        disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
    else:
        gen_summary_writer = tf.contrib.summary.create_file_writer(gen_log_dir)
        disc_summary_writer = tf.contrib.summary.create_file_writer(disc_log_dir)


    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=disc_opt,
                                     generator=gen,
                                     discriminator=disc)

    gen_losses = tfk.metrics.Mean('gen_loss', dtype=tf.float32)
    disc_losses = tfk.metrics.Mean('disc_loss', dtype=tf.float32)
    best_score = -np.inf
    initial_patience = patience

    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            x = dataset[i: i + batch_size, :]
            cc = cat_covs[i: i + batch_size, :]
            nc = num_covs[i: i + batch_size, :]

            # Train critic
            disc_loss = None
            for _ in range(nb_critic):
                z = tf.random.normal([x.shape[0], z_dim])
                disc_loss = train_disc(x, z, cc, nc, gen, disc, disc_opt, p_aug=p_aug, norm_scale=norm_scale)
            disc_losses(disc_loss)

            # Train generator
            z = tf.random.normal([x.shape[0], z_dim])
            gen_loss = train_gen(z, cc, nc, gen, disc, gen_opt, p_aug=p_aug, norm_scale=norm_scale)
            gen_losses(gen_loss)
        if not gamma_list is None:
            try:
                score = score_fn(gen, kappa=kappa)
                gamma_list.append(score)
            except Exception as e:
                print('exception in score function: ', str(e))
                sys.exit(1)
        # Logs
        with disc_summary_writer.as_default():
            if tf_version == 1:
                tf.summary.scalar('loss', disc_losses.result(), step=epoch)
            else:
                tf.summary.scalar('loss', disc_losses.result())

        with gen_summary_writer.as_default():
            if tf_version == 1:
                tf.summary.scalar('loss', gen_losses.result(), step=epoch)
            else:
                tf.summary.scalar('loss', gen_losses.result())

        # Save the model
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            try:
                score = score_fn(gen)
            except Exception as e:
                print('exception in score fcn: ', str(e))
                sys.exit(1)

            if score > best_score:
                print('Saving model ...')
                save_fn()
                best_score = score
                patience = initial_patience
            else:
                patience -= 1

            if verbose:
                print('Score: {:.3f}'.format(score))

        if verbose:
            print('Epoch {}. Gen loss: {:.2f}. Disc loss: {:.2f}'.format(epoch + 1,
                                                                         gen_losses.result(),
                                                                         disc_losses.result()))
        gen_losses.reset_states()
        disc_losses.reset_states()

        #if patience == 0:
        #    break



def predict(cc, nc, gen, z=None, training=False):
    """
    Make predictions
    :param cc: Categorical covariates
    :param nc: Numerical covariates
    :param gen: Generator model
    :param z: Latent input
    :param training: Whether training
    :return: Sampled data
    """
    nb_samples = cc.shape[0]
    if z is None:
        z_dim = gen.input[0].shape[-1]
        z = tf.random.normal([nb_samples, z_dim])
    out = gen([z, cc, nc], training=training)
    if not training:
        return out.numpy()
    return out



#### Data


def my_find_mostvaried(df, n):
    # df is genes X samples
    # calculate var, sort cols into n highest vars, drop shape[1]-n cols
    if n == 0:
        return df, None
    sdList = df.std(axis=0)
    sdDict = {k: v for v, k in enumerate(sdList)}
    sdDictSorted = sorted(sdDict.items(), key=operator.itemgetter(0), reverse=True) 
    topN = sdDictSorted[0:n]
    indices = [x[1] for x in topN]
    slicedDF = df[:,indices]
    return slicedDF, indices

def my_standardize(df):
    # this method requires df in samples x genes
    df = zscore(df, axis=0)
    return df

def transpose_df(df, cur_index_col, new_index_col):
    df = df.set_index(cur_index_col).T
    df.reset_index(level=0, inplace=True)
    cols = [new_index_col] + list(df.columns)[1:]
    df.columns=cols
    return df

def my_prep_data(n, expr_df, info_df, seed, use_meta_cols, train_percent):

    # Process categorical metadata
    cat_dicts = [] # big dict to hold all categorical dicts
    def cat(var):
        var_dict_inv = np.array(list(sorted(set(var))))
        var_dict = {t: i for i, t in enumerate(var_dict_inv)}
        var = np.vectorize(lambda t: var_dict[t])(var)
        cat_dicts.append(var_dict_inv) # add to big dict
        return var, var_dict_inv

    metaDict = dict()
    metaDict_inv = dict()
    my_tuple = tuple()
    for meta_param in use_meta_cols['cat']:
        metaDict[meta_param] = info_df[meta_param]
        metaDict[meta_param], metaDict_inv[meta_param] = cat(metaDict[meta_param])
        my_tuple = my_tuple + (metaDict[meta_param][:, None],)

    cat_covs = np.concatenate(my_tuple, axis=-1)
    cat_covs = np.float32(cat_covs)
    print('Cat covs: ', cat_covs.shape)

    num_dicts = [] # big dict to hold all numerical dicts
    def num(var):
        '''Function to repeatedly process categorical metadata. Pass in a column ("var") from info_df as a variable.'''
        var_dict_inv = np.array(list(sorted(set(var))))
        var_dict = {t: i for i, t in enumerate(var_dict_inv)}
        var = np.vectorize(lambda t: var_dict[t])(var)
        num_dicts.append(var_dict_inv) # add to big dict
        return var, var_dict_inv

    metaDict = dict()
    metaDict_inv = dict()
    my_tuple = tuple()
    for meta_param in use_meta_cols['num']:
        metaDict[meta_param] = info_df[meta_param]
        metaDict[meta_param], metaDict_inv[meta_param] = num(metaDict[meta_param])
        my_tuple = my_tuple + (metaDict[meta_param][:, None],)

    num_covs = np.concatenate(my_tuple, axis=-1)
    num_covs = np.float32(num_covs)
    print('num covs: ', num_covs.shape)

    # 1. Log-transform data
    x = np.log2(1+ expr_df)

    # 2. convert df to array
    x = np.float32(x)
    #x = expr_df

    # 3. standardize expression data
    #x = (x - x.mean()) / x.std()
    x = my_standardize(df=x.T)

    # 4. transpose matrix (don't need to do this step if in previous step you did the transpose
    #x = x.T

    # 5. find n most varied genes
    #x, indices = my_find_mostvaried(x, n)



    # Train/test split
    x_train, x_test = split_train_test(x=x, train_rate=train_percent, seed=seed)
    num_covs_train, num_covs_test = split_train_test(x=num_covs, train_rate=train_percent, seed=seed)
    cat_covs_train, cat_covs_test = split_train_test(x=cat_covs, train_rate=train_percent, seed=seed)

    return cat_dicts, cat_covs, cat_covs_test, cat_covs_train, num_covs, num_covs_test, num_covs_train, x, x_test, x_train


def my_train(CONFIG, cat_dicts, cat_covs, cat_covs_test, cat_covs_train, num_covs, num_covs_test, num_covs_train, x,
             x_test, x_train, checkpoint_dir, gamma_list, odir, kappa=1, tf_version=1):
    # Train on GL liver...

    MODELS_DIR = odir + '/models/'

    # GPU limit
    limit_gpu(CONFIG['gpu'])

    # Define model
    vocab_sizes = [len(c) for c in cat_dicts]
    print('Vocab sizes: ', vocab_sizes)
    nb_numeric = num_covs.shape[-1]
    x_dim = x.shape[-1]
    gen = make_generator(x_dim, vocab_sizes, nb_numeric,
                         h_dims=[CONFIG['hdim']] * CONFIG['nb_layers'],
                         z_dim=CONFIG['latent_dim'])
    disc = make_discriminator(x_dim, vocab_sizes, nb_numeric,
                              h_dims=[CONFIG['hdim']] * CONFIG['nb_layers'])

    # Evaluation metrics
    def score_fn(x_test, cat_covs_test, num_covs_test, kappa=1):
        def _score(gen, kappa=1):
            x_gen = predict(cc=cat_covs_test, nc=num_covs_test, gen=gen)
            gamma_dx_dz_orig = gamma_coef(x_test, x_gen, kappa)
            '''x_mean = np.mean(x_train, axis=0)
            x_std = np.std(x_train, axis=0)
            x_gen = x_gen * x_std + x_mean
            print('after unstdize: ', str(gamma_coef(x_test, x_gen, kappa)))
            if gamma_dx_dz_orig > 0.95:
                #my_x_gen = predict(cc=cat_covs,  nc=num_covs, gen=gen)
                num_samples = cat_covs_test.shape[0]
                np.savetxt('x_gen_' + str(num_samples) + '_' + str(gamma_dx_dz_orig) + '.csv', x_gen, delimiter=',')
                np.savetxt('cat_covs_' + str(num_samples) + '_' + str(gamma_dx_dz_orig) + '.csv', cat_covs_test, delimiter=',')
                np.savetxt('num_covs' + str(num_samples) + '_' + str(gamma_dx_dz_orig) + '.csv', num_covs_test, delimiter=',')'''

            return gamma_dx_dz_orig

        return _score

    # Function to save models
    def save_fn(models_dir=MODELS_DIR):
        gen.save(models_dir + 'gen_liver.h5')


    # Train model
    gen_opt = tfk.optimizers.RMSprop(CONFIG['lr'])
    disc_opt = tfk.optimizers.RMSprop(CONFIG['lr'])

    train(dataset=x_train,
          cat_covs=cat_covs_train,
          num_covs=num_covs_train,
          z_dim=CONFIG['latent_dim'],
          batch_size=CONFIG['batch_size'],
          epochs=CONFIG['epochs'],
          nb_critic=CONFIG['nb_critic'],
          gen=gen,
          disc=disc,
          gen_opt=gen_opt,
          disc_opt=disc_opt,
          score_fn=score_fn(x_test, cat_covs_test, num_covs_test, kappa=1),
          save_fn=save_fn,
          log_dir=checkpoint_dir + '/logs',
          checkpoint_dir=checkpoint_dir,
          gamma_list=gamma_list,
          kappa=kappa,
          tf_version=tf_version)

    # Evaluate data
    score = score_fn(x_test, cat_covs_test, num_covs_test, kappa=1)(gen)
    print('Gamma(Dx, Dz): {:.4f}'.format(score))


def pcaPlot(pca, df, info_df, variable, title, gen_dir, use_meta_cols):
    pcaDF = pd.DataFrame(data=pca.fit_transform(df), columns=['PC 1', 'PC 2'])
    pcaDF.index = info_df.index
    for meta_param in list(use_meta_cols['cat']):
        pcaDF = pd.concat([pcaDF, info_df[[meta_param]]], axis=1)

    sns.set(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(5,5))

    hue_cat = pd.Categorical(pcaDF[variable])
    ax = sns.scatterplot(x=pcaDF['PC 1'], y=pcaDF['PC 2'], hue=hue_cat, s=100)

    ax.set_xlabel('PC 1 ' + '(' + str(round(pca.explained_variance_ratio_[0]*100, 1)) + '% variance)', fontsize=15)
    ax.set_ylabel('PC 2 ' + '(' + str(round(pca.explained_variance_ratio_[1]*100, 1)) + '% variance)', fontsize=15)
    ax.set_title(title, fontsize=20)
    #plt.show()
    if gen_dir is None:
        gen_dir = '.'
    plt.savefig(gen_dir + '/' + title, dpi=300)
    plt.close()

def tsne_2d(data, **kwargs):
    """
    Transform data to 2d tSNE representation
    :param data: expression data. Shape=(dim1, dim2)
    :param kwargs: tSNE kwargs
    :return:
    """
    from sklearn.manifold import TSNE
    print('... performing tSNE')
    tsne = TSNE(n_components=2, **kwargs)
    return tsne.fit_transform(data)

def plot_tsne_2d(data, labels, **kwargs):
    """
    Plots tSNE for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    dim1, dim2 = data.shape

    # Prepare label dict and color map
    label_set = set(labels)
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Perform tSNE
    if dim2 == 2:
        # print('plot_tsne_2d: Not performing tSNE. Shape of second dimension is 2')
        data_2d = data
    elif dim2 > 2:
        data_2d = tsne_2d(data, **kwargs)
    else:
        raise ValueError('Shape of second dimension is <2: {}'.format(dim2))

    # Plot scatterplot
    for k, v in label_dict.items():
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v)
    plt.legend()
    return plt.gca()

def myPlot(x, x_gen, info_df, output_dir, use_meta_cols):

    # tsne plots
    '''import umap.umap_ as umap
    print('x shape = ' + str(x.shape))
    print('x_gen shape = ' + str(x_gen.shape))
    x_combined = np.concatenate((x, x_gen))
    categories = ['real'] * x.shape[0] + ['fake'] * x_gen.shape[0]
    #tissues_test = [info_df[tidx] for tidx in info_df[:, 0]]
    #tissues_combined = tissues_test + tissues_test
    emb_2d = umap.UMAP().fit_transform(x_combined)
    plt.figure(figsize=(10, 10))
    plot_tsne_2d(emb_2d, labels=np.array(categories), s=4)
    plt.title('UMAP real/synthetic')
    #plt.show()
    plt.savefig(output_dir + '/umap_real_v_synthetic.png', dpi=300)'''

    pca = PCA(n_components=2)

    #x = standardize(x)

    for meta_param in list(use_meta_cols['cat']):
        pcaPlot(pca, x, info_df, meta_param, meta_param + '_Real_Dataset_' + 'n=' + str(x.shape[0]), output_dir, use_meta_cols)
        pcaPlot(pca, x_gen, info_df, meta_param, meta_param + '_Fake_Dataset_' + 'n=' + str(x_gen.shape[0]), output_dir, use_meta_cols)


def plot_gamma(gamma_list):

    plt.plot(list(range(len(gamma_list))), gamma_list)
    plt.ylim([0, 1])
    plt.savefig('./' + 'gamma_scores', dpi=300)

def calculate_norms(A, B):
    N = A - B
    #frobenius_norm = LA.norm(N, ord='fro', axis=1)
    l1_norm = LA.norm(LA.norm(N, ord=1, axis=1))
    l2_norm = LA.norm(LA.norm(N, ord=2, axis=1))
    print('L1 norm = ' + str(l1_norm))
    print('L2 norm = ' + str(l2_norm))

def calculate_close(A, B, delta):
    N = A - B
    shape = N.shape
    size = shape[0] * shape[1]
    avgDelta = np.sum(N, axis=None) / size
    print('avg delta = ' + str(avgDelta))
    N[N<delta] = 1
    N[N>delta] = 0
    arraySum = np.sum(N, axis=None)
    print('num close = ' + str(arraySum))
    print('num far = ' + str(size - arraySum))

def un_standardize(real_expr, fake_expr, exp):
    '''x = np.log10(1+ expr_df)
    x = np.float32(x)
    x = (x - x.mean()) / x.std()'''
    real_mean = real_expr.mean()
    real_std = real_expr.std()
    x = (fake_expr * real_std) + real_mean
    x = np.power(x, exp)
    return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='boolean train or not', default='True')
    parser.add_argument('-s', '--seed', help='integer rng seed', default=0)
    parser.add_argument('-ie', '--input_expr', help='input expression data', default=None)
    parser.add_argument('-im', '--input_meta', help='input meta data', default=None)
    parser.add_argument('-umf', '--use_meta_file', help='file to specify meta data to use in analysis', default=None, required=True)
    parser.add_argument('-cd', '--checkpoint_dir', help='checkpoint directory', default='checkpoints')
    parser.add_argument('-gf', '--genes_file', help='input genelist data', default='top-liver-genes.txt')
    parser.add_argument('-m', '--model', help='model file to use instead of training', default=None)
    parser.add_argument('-od', '--output_dir', help='output dir', default=None, required=True)
    parser.add_argument('-g', '--gpu', help='number of gpus', default=0)
    parser.add_argument('-e', '--epochs', help='epochs', default=100)
    parser.add_argument('-ld', '--latent_dim', help='number of latent dimensions', default=64)
    parser.add_argument('-bs', '--batch_size', help='batch size', default=16)
    parser.add_argument('-nl', '--nb_layers', help='number of layers', default=2)
    parser.add_argument('-hd', '--hdim', help='number of units per hidden layer ', default=256)
    parser.add_argument('-lr', '--lr', help='learning rate', default=5e-04)
    parser.add_argument('-nb', '--nb_critic', help='number of critic batches per gen batch', default=5)
    parser.add_argument('-ng', '--num_genes', help='number of genes with highest variance', default=0)
    parser.add_argument('-pg', '--plot_gamma', help='boolean plot gamma vals', default='False')
    parser.add_argument('-osr', '--over_sample_rate', help='integer over sample rate', default=1)
    parser.add_argument('-ns', '--num_samples', help='integer number of samples to generate', default=0)
    parser.add_argument('-k', '--kappa', help='float multiple in denom of stdize', default=1)
    parser.add_argument('-x', '--excl', help='list of meta params to exclude', default=None, required=False)
    parser.add_argument('-tp', '--train_percent', help='percentage to split for training', default=0.80, required=False)
    parser.add_argument('-us', '--un_standardize', help='boolean unstdize gen.csv', default='False', required=False)
    parser.add_argument('-tf', '--tf_version', help='version of tf (1 or 2)', default=1, required=False)
    return parser.parse_args()
    
def main():
    # -g 0 -e 1 -ld 8 -bs 16 -nl 2 -hd 256 -lr 1e-03 -nb 5 -ng 0 -pg False -s 23 -ns 112 \
    # -ie expanded_expr_df.csv -im expanded_info_df.csv

    # -t False -m MODELS/gamma_0.983_ld_8_bs_2_nl_2_hd_64_lr_1e-04.h5 -ns 112 -od /tmp
    options = parse_args()
    CONFIG = {'gpu': int(options.gpu), 'epochs': int(options.epochs), 'latent_dim': int(options.latent_dim),
              'batch_size': int(options.batch_size), 'nb_layers': int(options.nb_layers), 'hdim': int(options.hdim),
              'lr': float(options.lr), 'nb_critic': int(options.nb_critic)}
    expr_df = pd.read_csv(options.input_expr, index_col=0)
    info_df = pd.read_csv(options.input_meta, header=0, sep=',')
    with open(options.use_meta_file, 'r') as f:
        use_meta_cols = json.load(f)
    f.close()


    cat_dicts, cat_covs, cat_covs_test, cat_covs_train, num_covs, num_covs_test, num_covs_train, x, x_test, \
    x_train, = my_prep_data(int(options.num_genes), expr_df, info_df, int(options.seed), use_meta_cols, float(options.train_percent))

    if eval(options.train):
        checkpoint_dir = options.checkpoint_dir
        kappa = float(options.kappa)

        np.random.seed(int(options.seed))
        tf.random.set_seed(int(options.seed))

        if eval(options.plot_gamma):
            gamma_list = list()
        else:
            gamma_list = None
        print('training!')
        my_train(CONFIG, cat_dicts, cat_covs, cat_covs_test, cat_covs_train, num_covs, num_covs_test, num_covs_train, x, \
             x_test, x_train, checkpoint_dir, gamma_list, options.output_dir, kappa, int(options.tf_version))
        gen = tf.keras.models.load_model(options.output_dir + '/models/gen_liver.h5') # this is the one I just trained
        num_samples = int(options.num_samples)
        if num_samples == 0:
            num_samples = len(cat_covs)
        cc = cat_covs[0:num_samples]
        nc = num_covs[0:num_samples]

        x_gen = predict(cc=cc, nc=nc, gen=gen)
        x_gen_df = pd.DataFrame(data=x_gen.T, index=expr_df.index, columns=expr_df.T.index[0:num_samples])
        x_gen_df.to_csv(options.output_dir + '/gen.csv', sep=',', header=True, index=True)
        myPlot(x, x_gen, info_df, options.output_dir, use_meta_cols)

    else:
        print('not training!')
        gen = tf.keras.models.load_model(options.model, compile=False)
        num_samples = int(options.num_samples)
        if num_samples == 0:
            num_samples = len(cat_covs)
        cc = cat_covs[0:num_samples]
        nc = num_covs[0:num_samples]

        x_gen = predict(cc=cc, nc=nc, gen=gen)
        expr_df_samples = expr_df.T.index[0:num_samples]
        expr_df_genes = expr_df.index
        x_gen_df = pd.DataFrame(data=x_gen.T, index=expr_df_genes, columns=expr_df_samples)
        if eval(options.un_standardize):
            x_gen_df = un_standardize(expr_df, x_gen_df, exp=10)
        x_gen_df.to_csv(options.output_dir + '/gen.csv', sep=',', header=True, index=True)
        expr_df.to_csv(options.output_dir + '/expr_subset.csv', sep=',', header=True, index=True)
        myPlot(x, x_gen, info_df, options.output_dir, use_meta_cols)



if __name__ == "__main__":
    main()
