import os
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
from tensorflow_probability import edward2 as ed
from scipy import sparse, stats
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Deconfounder(object):
    """
    Class that implements the Deconfounder, inspired by https://github.com/blei-lab/deconfounder_tutorial
    """
    def __init__(self, train_environments, val_environment, test_environment, args):
        """
        Initialisation and training. Fits Assignment model in loop over hyperparameters until pvalue of held
        out data is in desired range (between args["minP"] and args["maxP"]). To this end all environments (incl. validation
        and test) are merged but random entries across all environments are held out to ensure generalisation.
        Once Assignment model is found, outcome model is fitted to desired accuracy looping over hyperparameter.
        """
        self.selected_features = []
        self.minP = args["minP"]
        self.maxP = args["maxP"]
        self.minFeatures = args["minFeatures"]
        self.minAccuracy = args["minAccuracy"]
        self.seed = args["seed"]
        self.verbose = args["verbose"]
        self.latent_representation_found = False
        self.Terminate = False
        self.latent_variables = None
        self.latent_dim = None
        self.model = None
        self.targetkey = args["target"][0]
        self.output_pvals = args["output_pvals"]

        # Fix the seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # Set up test set df
        columns = args["columns"]

        x_test, y_test = test_environment.get_all()
        df_test = pd.DataFrame(x_test.numpy(), columns=columns)
        df_test[self.targetkey] = y_test.numpy()
        self.df_test = df_test
        self.n_test = len(df_test)

        # Set up validation set df
        x_val, y_val = val_environment.get_all()
        df_val = pd.DataFrame(x_val.numpy(), columns=columns)
        df_val[self.targetkey] = y_val.numpy()
        self.df_val = df_val
        self.n_val = len(df_val)

        # Set up main df
        x_all = []
        y_all = []

        for e, env in enumerate(train_environments):
            x, y = env.get_all()
            x_all.append(x.numpy())
            y_all.append(y.numpy())


        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)


        df_train = pd.DataFrame(x_all, columns=columns)
        df_train[self.targetkey] = y_all
        self.df_train = df_train
        self.n_train = len(df_train)

        if self.verbose:
            print('testshape:', df_test.shape)
            print('valshape:', df_val.shape)
            print('trainshape:', df_train.shape)

        self.dfX = pd.concat([df_train, df_val], axis=0)
        self.dfX = self.dfX.drop(columns=[self.targetkey], axis=1).astype(float)

        d = self.df_train.shape[1] - 1
        if self.verbose:
            print('Number of features:', d)


        corr_threshold = 0.9 #This is in line with DCF tutorial on https://github.com/blei-lab/deconfounder_tutorial
        corr = self.dfX.corr()
        npcorr = np.array(corr)
        deletedcols = {}
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    continue
                elif (npcorr[i, j] > corr_threshold) or (npcorr[i, j] < -corr_threshold):
                    column = corr.columns[i]
                    if column not in deletedcols:
                        self.dfX = self.dfX.drop(columns=[column], axis=1)
                        self.df_train = self.df_train.drop(columns=[column], axis=1)
                        self.df_val = self.df_val.drop(columns=[column], axis=1)
                        self.df_test = self.df_test.drop(columns=[column], axis=1)
                        deletedcols[column] = True
                else:
                    continue

        # Drop 0 std columns that are still around for some reason:
        self.dfXstd = self.dfX.std()
        to_drop = np.where(np.array(self.dfXstd) == 0.0)[0]
        self.to_drop = list(self.dfX.columns[to_drop])
        print('to drop:', self.to_drop)

        self.dfX = self.dfX.drop(columns=self.to_drop, axis=1)
        print('dropped 0 std columns of self.dfX')
        self.dfXmean = self.dfX.mean()
        self.dfXstd = self.dfX.std()

        self.df_train = self.df_train.drop(columns=self.to_drop, axis=1)
        self.df_val = self.df_val.drop(columns=self.to_drop, axis=1)
        self.df_test = self.df_test.drop(columns=self.to_drop, axis=1)

        d2 = self.df_test.shape[1] - 1
        if self.verbose:
            print('Number of features after correlated and 0 std features elimination:', d2)
        time.sleep(5)

        # Parameters for factor model
        latent_dim = 1
        std_list = [0.1, 0.5, 1.0]
        factor_model_list = ['linearPPCA', 'quadraticPPCA']

        # Parameters for outcome model
        if self.output_pvals:
            penal_list = [10.0, 1.0, 0.1, 0.01]
        else:
            penal_list = [100.0, 10.0, 1.0, 0.1, 0.01]

        while (latent_dim <= 10): #Limiting to 10 for computational budget reasons, can be increased if wanted
            idx_std = 0
            while (not self.latent_representation_found) and (idx_std < len(std_list)):
                idx_model = 0
                while (not self.latent_representation_found) and (idx_model < len(factor_model_list)):
                    pval, Zhat = self.fitAssignmentModel(self.df_train, self.df_val, self.df_test,
                                                         latent_dim, std_list[idx_std],
                                                         factor_model_list[idx_model], self.seed)
                    if self.verbose:
                        print('pval:', pval)
                    if (pval > self.minP) and (pval < self.maxP):
                        self.latent_representation_found = True
                        self.latent_variables = Zhat
                        self.latent_dim = latent_dim
                    else:
                        idx_model += 1
                        if (idx_model == len(factor_model_list)) and (idx_std == len(std_list)):
                            print('No assignment model fitted to standard')
                            return
                idx_std += 1
            latent_dim += 1

        idx_penal = 0
        if self.latent_representation_found:
            while not self.Terminate and (idx_penal < len(penal_list)):
                modelA = self.fitOutcomeModel(self.df_train, self.df_val, penal_list[idx_penal], self.latent_variables,
                                              'linearPlain')
                if (modelA['acc'] >= self.minAccuracy) and (modelA['numFeatures'] >= self.minFeatures):
                    self.Terminate = True
                    self.model = modelA
                else:
                    idx_penal += 1
                    if idx_penal == len(penal_list):
                        print('No outcome model fitted to standard')

        return

    def fitAssignmentModel(self, df_train, df_val, df_test, latent_dim, stddv_datapoints, modeltype, seed):

        # 1. Merge trainSet and valSet and testSet for factor model learning, as we hold out entries differently
        df = pd.concat([df_train, df_val, df_test], axis=0)
        dfX = df.drop(columns=[self.targetkey]).astype(float)
        #   standardize the data for PPCA
        X = np.array((dfX - dfX.mean()) / dfX.std())

        # 2. Hold out data (some entries of X) in order to assess fit of assignment model
        num_datapoints, data_dim = X.shape
        holdout_portion = 0.2
        n_holdout = int(holdout_portion * num_datapoints * data_dim)
        holdout_row = np.random.randint(num_datapoints, size=n_holdout)
        holdout_col = np.random.randint(data_dim, size=n_holdout)
        holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), \
                                           (holdout_row, holdout_col)), \
                                          shape=X.shape)).toarray()
        holdout_subjects = np.unique(holdout_row)
        holdout_mask = np.minimum(1, holdout_mask)
        x_train = np.multiply(1 - holdout_mask, X)
        x_vad = np.multiply(holdout_mask, X)

        log_joint = ed.make_log_joint_fn(self.ppca_model)

        # 3. Define factor model to be used as assignment model
        if modeltype == 'linearPPCA':
            model = self.ppca_model(data_dim=data_dim,
                                    latent_dim=latent_dim,
                                    num_datapoints=num_datapoints,
                                    stddv_datapoints=stddv_datapoints,
                                    mask=1 - holdout_mask,
                                    form="linear")
        elif modeltype == 'quadraticPPCA':
            model = self.ppca_model(data_dim=data_dim,
                                    latent_dim=latent_dim,
                                    num_datapoints=num_datapoints,
                                    stddv_datapoints=stddv_datapoints,
                                    mask=1 - holdout_mask,
                                    form="quadratic")

        # 4. Fit chosen factor model using VI
        def variational_model(qb_mean, qb_stddv, qw_mean, qw_stddv,
                              qw2_mean, qw2_stddv, qz_mean, qz_stddv):
            qb = ed.Normal(loc=qb_mean, scale=qb_stddv, name="qb")
            qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
            qw2 = ed.Normal(loc=qw2_mean, scale=qw2_stddv, name="qw2")
            qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
            return qb, qw, qw2, qz

        log_q = ed.make_log_joint_fn(variational_model)

        def target(b, w, w2, z):
            """Unnormalized target density as a function of the parameters."""
            return log_joint(data_dim=data_dim,
                             latent_dim=latent_dim,
                             num_datapoints=num_datapoints,
                             stddv_datapoints=stddv_datapoints,
                             mask=1 - holdout_mask,
                             w=w, z=z, w2=w2, b=b, x=x_train)

        def target_q(qb, qw, qw2, qz):
            return log_q(qb_mean=qb_mean, qb_stddv=qb_stddv,
                         qw_mean=qw_mean, qw_stddv=qw_stddv,
                         qw2_mean=qw2_mean, qw2_stddv=qw2_stddv,
                         qz_mean=qz_mean, qz_stddv=qz_stddv,
                         qw=qw, qz=qz, qw2=qw2, qb=qb)

        qb_mean = tf.Variable(np.ones([1, data_dim]), dtype=tf.float32)
        qw_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
        qw2_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
        qz_mean = tf.Variable(np.ones([num_datapoints, latent_dim]), dtype=tf.float32)
        qb_stddv = tf.nn.softplus(tf.Variable(0 * np.ones([1, data_dim]), dtype=tf.float32))
        qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32))
        qw2_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32))
        qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([num_datapoints, latent_dim]), dtype=tf.float32))

        qb, qw, qw2, qz = variational_model(qb_mean=qb_mean, qb_stddv=qb_stddv,
                                            qw_mean=qw_mean, qw_stddv=qw_stddv,
                                            qw2_mean=qw2_mean, qw2_stddv=qw2_stddv,
                                            qz_mean=qz_mean, qz_stddv=qz_stddv)

        energy = target(qb, qw, qw2, qz)
        entropy = -target_q(qb, qw, qw2, qz)

        elbo = energy + entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        t = []

        num_epochs = 500

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    t.append(sess.run([elbo]))

                b_mean_inferred = sess.run(qb_mean)
                b_stddv_inferred = sess.run(qb_stddv)
                w_mean_inferred = sess.run(qw_mean)
                w_stddv_inferred = sess.run(qw_stddv)
                w2_mean_inferred = sess.run(qw2_mean)
                w2_stddv_inferred = sess.run(qw2_stddv)
                z_mean_inferred = sess.run(qz_mean)
                z_stddv_inferred = sess.run(qz_stddv)

                if np.isnan(w_mean_inferred).any():
                    print('Found NaN values')
                    print(w_mean_inferred)
                    break

        if self.verbose:
            print("Inferred W:")
            print(w_mean_inferred)
            print("Standard Deviation:")
            print(w_stddv_inferred)

        def replace_latents(b, w, w2, z):

            def interceptor(rv_constructor, *rv_args, **rv_kwargs):
                """Replaces the priors with actual values to generate samples from."""
                name = rv_kwargs.pop("name")
                if name == "b":
                    rv_kwargs["value"] = b
                elif name == "w":
                    rv_kwargs["value"] = w
                elif name == "w":
                    rv_kwargs["value"] = w2
                elif name == "z":
                    rv_kwargs["value"] = z
                return rv_constructor(*rv_args, **rv_kwargs)

            return interceptor

        # Generate replicated datasets
        n_rep = 100  # number of replicated datasets we generate
        holdout_gen = np.zeros((n_rep, *(x_train.shape)))

        for i in range(n_rep):
            b_sample = npr.normal(b_mean_inferred, b_stddv_inferred)
            w_sample = npr.normal(w_mean_inferred, w_stddv_inferred)
            w2_sample = npr.normal(w2_mean_inferred, w2_stddv_inferred)
            z_sample = npr.normal(z_mean_inferred, z_stddv_inferred)

            if np.isnan(w_sample).any():
                break

            with ed.interception(replace_latents(b_sample, w_sample, w2_sample, z_sample)):
                generate = self.ppca_model(
                    data_dim=data_dim, latent_dim=latent_dim,
                    num_datapoints=num_datapoints, stddv_datapoints=stddv_datapoints,
                    mask=np.ones(x_train.shape))

            with tf.Session() as sess:
                x_generated, _ = sess.run(generate)

            # look only at the heldout entries
            holdout_gen[i] = np.multiply(x_generated, holdout_mask)

        # Evaluate mean loglikelihood in comparison and mean percentage of times that this is higher 
        # than of original held out entires
        n_eval = 100  # we draw samples from the inferred Z and W
        obs_ll = []
        rep_ll = []
        for j in range(n_eval):
            w_sample = npr.normal(w_mean_inferred, w_stddv_inferred)
            z_sample = npr.normal(z_mean_inferred, z_stddv_inferred)

            if np.isnan(w_sample).any():
                print('Found NaNs so return pval 0.0')
                print(w_sample)
                return 0.0, z_mean_inferred

            holdoutmean_sample = np.multiply(z_sample.dot(w_sample), holdout_mask)
            obs_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                                             stddv_datapoints).logpdf(x_vad), axis=1))

            rep_ll.append(np.mean(stats.norm(holdoutmean_sample, \
                                             stddv_datapoints).logpdf(holdout_gen), axis=2))

        obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)

        pvals = np.array([np.mean(rep_ll_per_zi[:, i] < obs_ll_per_zi[i]) for i in range(num_datapoints)])
        holdout_subjects = np.unique(holdout_row)
        overall_pval = np.mean(pvals[holdout_subjects])

        return overall_pval, z_mean_inferred


    def fitOutcomeModel(self, df_train, df_val, _lambda, Zhat, model='linearPlain', seed=10):

        for i in range(self.latent_dim):
            name = f'Z{i+1}'
            df_train[name] = self.latent_variables[:self.n_train, i]
            df_val[name] = self.latent_variables[self.n_train:(self.n_train + self.n_val), i]

        dfyTrain = df_train[self.targetkey]
        yTrain = np.array(dfyTrain)
        dfXTrain = df_train.drop(columns=[self.targetkey], axis=1).astype(float)
        
        # standardize the data for PPCA
        dfXTrain.iloc[:, :-self.latent_dim] = (dfXTrain.iloc[:, :-self.latent_dim] -
                                               self.dfXmean) / self.dfXstd
        XTrain = np.array(dfXTrain)

        dfyVal = df_val[self.targetkey]
        yVal = np.array(dfyVal)
        dfXVal = df_val.drop(columns=[self.targetkey], axis=1).astype(float)
        
        # standardize the data for PPCA
        dfXVal.iloc[:, :-self.latent_dim] = (dfXVal.iloc[:, :-self.latent_dim] - self.dfXmean) / self.dfXstd
        XVal = np.array(dfXVal)

        dcfX_train = sm.add_constant(XTrain, has_constant='add')

        def calculate_acc(y, yp):
            out = np.sum(y == yp) / len(y)
            return out

        columnnames = ['intercept']
        for k in range(len(dfXTrain.columns)):
            columnnames.append(dfXTrain.columns[k])
        columnnames = np.array(columnnames)

        if model == 'linearPlain':
            if self.output_pvals:
                dcflogit_model = sm.Logit(yTrain, dcfX_train)
                dcfresult = dcflogit_model.fit_regularized(L1_wt=0.0, alpha=_lambda, maxiter=5000, disp=0)
                dcf_coefs = np.vstack(
                    (dcfresult.params[dcfresult.params != 0.], dcfresult.pvalues[dcfresult.params != 0.])).T
                # print('shape of coefs:',np.shape(dcf_coefs))
                num_features = sum(dcfresult.params != 0)
                index_colnames = (dcfresult.params != 0)

                # make predictions with the causal model and evaluate
                dcfX_val = sm.add_constant(XVal, has_constant='add')
                dcfy_predprob = dcfresult.predict(dcfX_val)
                dcfy_pred = (dcfy_predprob > 0.5)
            else:
                dcfX_train = dcfX_train[:, 1:]
                columnnames = columnnames[1:]
                from sklearn.linear_model import LogisticRegression
                dcfresult = LogisticRegression(penalty='l1', C=(1.0 / _lambda), solver='liblinear', max_iter=100,
                                               verbose=0).fit(dcfX_train, yTrain)
                index_colnames = (dcfresult.coef_ != 0)
                dcf_coefs = np.vstack(
                    (dcfresult.coef_[index_colnames], np.zeros(shape=np.shape(dcfresult.coef_[index_colnames])))).T

                index_colnames = index_colnames[0]
                num_features = sum(index_colnames)

                # make predictions with the causal model 
                dcfX_val = XVal
                dcfy_pred = dcfresult.predict(dcfX_val)

            # Evaluate test accuracy
            dcf_acc = calculate_acc(yVal, dcfy_pred)
            print('predictions:', dcfy_pred)
            print('truth:', yVal)
            print('Acc:', dcf_acc)


        modelOut = {}
        print('dcf_coefs:', dcf_coefs)
        modelOut['Features'] = pd.DataFrame(dcf_coefs, index=columnnames[index_colnames], columns=['value', 'pval'])
        print('modelout features:', modelOut['Features'])
        modelOut['numFeatures'] = num_features
        modelOut['acc'] = dcf_acc

        return modelOut

    def get_test_stats(self, model, df_test):

        for i in range(self.latent_dim):
            name = f'Z{i+1}'
            df_test[name] = self.latent_variables[-self.n_test:, i]

        dfy = df_test[self.targetkey]
        y_test = np.array(dfy)

        df_test = df_test.drop(columns=[self.targetkey], axis=1).astype(float)
        df_test.iloc[:, :-self.latent_dim] = (df_test.iloc[:, :-self.latent_dim] -
                                              self.dfXmean) / self.dfXstd
        for name in df_test.columns:
            if name not in list(model['Features'].index.values):
                df_test = df_test.drop(columns=[name], axis=1)
        x_test = np.array(df_test)
        if ('intercept' in model['Features'].index.values) and self.output_pvals:
            x_test = sm.add_constant(x_test, has_constant='add')

        coefs = np.array(model['Features']['value'])

        def logit(x):
            return 1.0 / (1.0 + np.exp(-x))

        def predict_proba(X):
            f = np.matmul(x_test, coefs)
            return logit(f)

        def calculate_acc(y, yp):
            out = np.sum(y == yp) / len(y)
            return out

        def calculate_acc_std(y, yp):
            out = np.std(y == yp)
            return out

        from sklearn.metrics import confusion_matrix
        predict_probs = predict_proba(x_test)
        predictions = np.array(predict_probs > 0.5) * 1.0
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
        test_accuracy = calculate_acc(y_test, predictions)
        test_accuracy_std = calculate_acc_std(y_test, predictions)

        test_stats = {}
        test_stats['test_accuracy'] = test_accuracy.tolist()
        test_stats['test_accuracy_std'] = test_accuracy_std.tolist()
        test_stats['confusion_matrix'] = conf_matrix.tolist()

        return test_stats

    def predictor_results(self):
        """
        Outputs results of Deconfounder in form required by main.py 
        """

        results = {}
        results['test_accuracy'] = 0.0
        results['test_accuracy_std'] = 0.0
        to_bucket = {
            "method": "Deconfounder",
            "features": None,
            "coefficients": None,
            "pvals": None,
            "test_acc": results['test_accuracy'],
            "test_acc_std": results['test_accuracy_std']
        }

        if self.Terminate:
            test_stats = self.get_test_stats(self.model, self.df_test)

            results['Features'] = self.model['Features']
            results['test_accuracy'] = test_stats['test_accuracy']
            results['test_accuracy_std'] = test_stats['test_accuracy_std']
            results['train_accuracy'] = self.model['acc']
            results['confusion_matrix'] = test_stats['confusion_matrix']
            results['numFeatures'] = self.model['numFeatures']
            to_bucket = {
                "method": "Deconfounder",
                "features": np.array(self.model['Features'].index).tolist(),
                "coefficients": np.array(self.model['Features']['value'].values).tolist(),
                "pvals": np.array(self.model['Features']['pval'].values).tolist(),
                "test_acc": results['test_accuracy'],
                "test_acc_std": results['test_accuracy_std']
            }

        return {
            "solution": self.Terminate,
            "latent_representation": self.latent_representation_found,
            "results": results,
            "test_acc": results['test_accuracy'],
            "to_bucket": to_bucket
        }

    def ppca_model(self, data_dim, latent_dim, num_datapoints, stddv_datapoints, mask, form="linear"):
        w = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                      scale=tf.ones([latent_dim, data_dim]),
                      name="w")  # parameter
        z = ed.Normal(loc=tf.zeros([num_datapoints, latent_dim]),
                      scale=tf.ones([num_datapoints, latent_dim]),
                      name="z")  # local latent variable / substitute confounder
        if form == "linear":
            x = ed.Normal(loc=tf.multiply(tf.matmul(z, w), mask),
                          scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                          name="x")  # (modeled) data
        elif form == "quadratic":
            b = ed.Normal(loc=tf.zeros([1, data_dim]),
                          scale=tf.ones([1, data_dim]),
                          name="b")  # intercept
            w2 = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                           scale=tf.ones([latent_dim, data_dim]),
                           name="w2")  # quadratic parameter
            x = ed.Normal(loc=tf.multiply(b + tf.matmul(z, w) + tf.matmul(tf.square(z), w2), mask),
                          scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                          name="x")  # (modeled) data
        return x, (w, z)
