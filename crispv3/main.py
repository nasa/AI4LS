import numpy as np
import pandas as pd

from dataio.datasets import get_datasets_for_experiment
from utils.gcp_helpers import save_json_to_bucket, save_dataframe_to_bucket
from utils.plotting import plot_most_predictive
from utils.vm_helpers import save_dict_to_json
from image_scripts.train_image_model import train_image_model, train_image_model_loocv, train_image_model_kfold
from image_scripts.extract_image_features import extract_image_features
from image_scripts.merge_tabular_image_features import save_merged_features
from image_scripts.gradcam import save_gradcam_output
from image_scripts.validate_config import validate_json_config

def run(config):
    experiment_type = config.get("experiment_type", "tabular_only")  # Default to tabular_only
    if(experiment_type=='image_only' or experiment_type=='multimodal'):
        print('Validating JSON Config. . .')
        validate_json_config(config)

        # Train the image model first
        image_model_training_type = config["image_data"].get("image_model_training_type", "train_test_split")
        if image_model_training_type == 'train_test_split':
            model = train_image_model(config)
        elif image_model_training_type == 'full_loocv':
            model = train_image_model_loocv(config)
        elif image_model_training_type == 'k_fold':
            from image_scripts.train_image_model import train_image_model_kfold
            model = train_image_model_kfold(config)
        else:
            print("Error: Incorrect value for config parameter [image_model_training_type]! Correct choices: train_test_split, full_loocv, or k_fold. Value found:", image_model_training_type)
            return

        apply_gradcam = config["image_data"]["image_model_gradcam"]["apply_gradcam"]
        if(apply_gradcam==True):
            # Save GradCAM Output
            save_gradcam_output(config, model)

        # Extract Image Model features
        extract_image_features(config, model)

        if(experiment_type=='multimodal'):
            # Merge Extracted image features and tabular data
            save_merged_features(config)

    ############################ LOAD CHOSEN DATASET ###################################
    environment_datasets, val_dataset, test_dataset = get_datasets_for_experiment(config)

    ########################### CHECK IF TARGET IS CONSTANT ############################
    from utils.ZeroVarianceChecker import ZeroVarianceCheckerTarget
    from utils.exceptionclasses import TargetHasZeroVarianceError
    constant_target = ZeroVarianceCheckerTarget(environment_datasets, in_any_env=False)
    # If target variable is constant for dataset then stop experiment early
    if constant_target.zero_var:
        raise TargetHasZeroVarianceError()


    ######################### LIST CHOSEN ENSEMBLE METHODS #############################
    ensemble_options = config.get('ensemble_options', {})
    data_config = config.get('data_options', {})
    selection_config = config.get('feature_selection_options', {})
    # Load list of models to include in ensemble, if not specified, run them all
    selected_models = ensemble_options.get("models", ["ERM", "RF", "ICP", "IRM", "DCF", "LIRM", "NLICP"])
    print("Running Ensemble with the following models: ", selected_models)
    # Initialise empty list to store per method results to outputs to file system/ cloud storage
    to_bucket_results = []


    ############################# Correlation to Target Ranking #########################
    '''from utils.CorrelationToTarget import CorrelationToTarget
    ct_args = {'max_features': selection_config.get('max_features', 25)}
    # Calculate correlations of each feature to target variable and save to file system/cloud storage
    tg_corr = CorrelationToTarget(environment_datasets, val_dataset, test_dataset, ct_args)
    tg_corr_df = tg_corr.target_corr_df
    tg_corr_df.to_csv(config['results_directory'] + 'tg_correlation_pairs.csv')
    if config['use_cloud']:
        save_dataframe_to_bucket(tg_corr_df, config['bucket_path'] + config['bucket_exp_path'] + 'tg_correlation_pairs.csv',
                                config['bucket_project'], config['bucket_name'])'''

    ############################# FEATURE REDUCTION 1 ######################################
    # 1. Remove any zero variance features
    # - Save list to results directory
    from utils.ZeroVarianceChecker import ZeroVarianceChecker
    print('Checking for features with zero variance')
    # Flag for checking zero variance across all environments ([train_environments], val, test) or within any one environment
    var_args = {
        'in_each_env': selection_config.get('zero_variance_in_each_env', False)
    }
    zero_var_checker = ZeroVarianceChecker(environment_datasets, val_dataset, test_dataset, var_args)
    if var_args['in_each_env']:
        zero_var_columns = zero_var_checker.zero_var_cols
        if config['verbose']:
            print('In atleast one environment the following columns had zero variance:', zero_var_columns)
        save_dict_to_json({'zero std columns removed': zero_var_columns},
                        config['results_directory'] + 'zero_std_columns.json')
    else:
        zero_var_columns = zero_var_checker.zero_var_cols
        if config['verbose']:
            print('Across all environments the following columns had zero variance:', zero_var_columns)
    keep_columns = zero_var_checker.reduced_feature_list()

    selected_feature_list = keep_columns
    config['data_options']['predictors'] = selected_feature_list
    # Reinitialise datasets without zero variance columns
    environment_datasets, val_dataset, test_dataset = get_datasets_for_experiment(config)

    # 2. Correlation Analysis
    # - for pairs of features with high correlation, remove one of each pair
    '''from models.Correlation import Correlation
    print('Running correlation analysis')
    corr_args = {
        "correlation_threshold": selection_config.get("correlation_threshold", 0.97),
        "seed": selection_config.get("seed", 12),
        "verbose": selection_config.get("verbose", 0),
        "columns": test_dataset.predictor_columns,
        "target": data_config['targets'],
        "save_plot": selection_config.get('plot_correlation', False),
        "fname": config['results_directory']
    }
    corr = Correlation(environment_datasets, val_dataset, test_dataset, corr_args)
    corr_results_dict = corr.predictor_results()
    keep_columns = corr_results_dict["results"]['retained_columns']
    column_pairs_df = corr_results_dict["results"]['invariant_correlations']

    selected_feature_list = keep_columns
    original_dimensionality = len(selected_feature_list)
    config['data_options']['predictors'] = selected_feature_list
    # Reinitialise datasets without highly correlated feature pairs
    environment_datasets, val_dataset, test_dataset = get_datasets_for_experiment(config)'''

    #####################################################################################
    ############################# TRAIN SELECTED MODELS #################################
    #####################################################################################

    #####################################################################################
    ################################ NON CAUSAL ERM #####################################
    if 'ERM' in selected_models:
        from models.EmpericalRiskMinimization import EmpericalRiskMinimization
        print("Running Non-Causal Linear ERM Baseline")
        # Set up ERM args
        ERM_options = ensemble_options.get('ERM', {})
        ERM_args = {
            "method": 'Linear',
            "cuda": False ,
            "seed": 12,
            "epochs": 1,
            "hidden_dim": 256
        }
        ERM_args.update(ERM_options)

        # Initialise and train ERM (calls train and test internally)
        erm = EmpericalRiskMinimization(environment_datasets, val_dataset, test_dataset, ERM_args)
        # Get results of ERM on test set
        erm_results_dict = erm.results()
        if config['verbose']:
            print(erm_results_dict)
        print("Finished ERM")
        to_bucket = erm_results_dict['to_bucket']
        to_bucket_results.append(to_bucket)
    #####################################################################################

    #####################################################################################
    ################################ NON CAUSAL RF #####################################
    if 'RF' in selected_models:
        from models.RandomForest import RandomForest
        print("Running Non-Causal RF Baseline")
        # Set up RF args
        RF_options = ensemble_options.get('RF', {})
        RF_args = {
            "model_params": {
                "verbose": 1,
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0,
                "max_features": 20,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0,
                "bootstrap": True,
                "oob_score": False,
                "n_jobs": -1,
                "random_state": None,
                "warm_start": False,
                "class_weight": None,
                "ccp_alpha": 0,
                "max_samples": None
            },
            "running_params": {
                "sample_weight": None
            }
        }
        RF_args.update(RF_options)

        # Initialise and train RF (calls train and test internally)
        rf = RandomForest(environment_datasets, val_dataset, test_dataset, RF_args)
        # Get results of RF on test set
        rf_results_dict = rf.results()
        if config['verbose']:
            print(rf_results_dict)
        print("Finished RF")
        to_bucket = rf_results_dict['to_bucket']
        to_bucket_results.append(to_bucket)

    #####################################################################################
    ############################# FEATURE REDUCTION 2 ###################################
    #####################################################################################
    # Several methods in CRISP are not compatible with large numbers of features,
    # as such we perform feature reduction using Non-Linear Invaraint Risk Minimization

    if "ICP" in selected_models or "NLICP" in selected_models or "DCF" in selected_models:

        from models.NonLinearInvariantRiskMinimization import NonLinearInvariantRiskMinimization

        # Setup Linear and Non-Linear IRM args
        FRIRM_options = ensemble_options.get('FRIRM', {})
        FRIRM_args = {
            # Flag for model to use in Non-Linear IRM ['NN': MLP, 'DNN': Deeper MLP]
            "NN_method": "NN",
            "verbose": 1,
            "n_iterations": 1000,
            "seed":  0,
            "l2_regularizer_weight": 0.001,
            "lr": 0.001,
            "penalty_anneal_iters": 100,
            "penalty_weight": 10000.0,
            "cuda": False,
            "hidden_dim":  256
        }
        FRIRM_args.update(FRIRM_options)

        print('Running IRM (Feature Reduction Mode)')
        IRM_NN = NonLinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, FRIRM_args)
        irm_results_dict = IRM_NN.results()
        if config['verbose']:
            print(irm_results_dict)

        # Extract top weighted features to use for remaining methods requiring reduced feature set
        to_bucket = irm_results_dict['to_bucket']
        to_bucket['method'] = 'Non Linear IRM (Feature Reduction Mode)'
        to_bucket_results.append(to_bucket)

        print("Finished Non Linear IRM")

        coefs = pd.DataFrame()
        coefs['feature'] = to_bucket['features']
        coefs['coefficient'] = to_bucket['coefficients']
        coefs['sort'] = coefs['coefficient'].abs()
        sorted_coefs = coefs.sort_values('sort', ascending=False)

        keep_columns = list(sorted_coefs['feature'][0:selection_config['max_features']])
        removed_columns = [c for c in test_dataset.predictor_columns if c not in keep_columns]

        save_dict_to_json({'Non Linear IRM Feature Selection columns removed': removed_columns},
                        config['results_directory'] + 'irm_dropped_columns.json')

        keep_columns_indices_to_original_features = np.array(
            [test_dataset.predictor_columns.index(f) for f in keep_columns])

        print('Keeping:', keep_columns)

        selected_feature_list = keep_columns
        config['data_options']['predictors'] = selected_feature_list
        # Generate reduced feature set datasets
        reduced_environment_datasets, reduced_val_dataset, reduced_test_dataset = get_datasets_for_experiment(config)

        # write test feature values to to_bucket_results for plotting
        to_bucket = {'test_feature_values': reduced_test_dataset.data.tolist(),
                    'features': reduced_test_dataset.predictor_columns}
        to_bucket_results.append(to_bucket)

    #####################################################################################
    ################################# DECONFOUNDER ######################################
    if "DCF" in selected_models:
        print("Running Deconfounder")
        from models.Deconfounder import Deconfounder

        # Deconfounder args
        dcf_options = ensemble_options.get('DCF', {})
        dcf_args = {
            "minP": 0.1,
            "maxP": 0.9,
            "minFeatures": 1,
            "minAccuracy": 0.5,
            "seed": 0,
            "verbose": 1,
            "target": data_config['targets'],
            "output_pvals": True
        }
        dcf_args.update(dcf_options)

        # If we wish to output p vals for each feature, then use reduced feature set
        if dcf_args["output_pvals"] and (original_dimensionality > 500):
            dcf_args["columns"] = reduced_test_dataset.predictor_columns
            dcf = Deconfounder(reduced_environment_datasets, reduced_val_dataset, reduced_test_dataset, dcf_args)
        else:
            dcf_args["columns"] = test_dataset.predictor_columns
            dcf = Deconfounder(environment_datasets, val_dataset, test_dataset, dcf_args)

        dcf_results_dict = dcf.predictor_results()
        print("Finished DCF")
        to_bucket = dcf_results_dict['to_bucket']
        to_bucket_results.append(to_bucket)


    #####################################################################################
    ################################# Linear ICP ########################################
    icp_solution = False
    # run linear and nonlinear ICP
    if "ICP" in selected_models:
        from models.LinearInvariantCausalPrediction import InvariantCausalPrediction
        # Set up ICP args
        ICP_options = ensemble_options.get('ICP', {})
        ICP_args = {
            "max_set_size": 2,
            "alpha": 0.05,
            "seed": 12,
            "verbose": 1
        }
        ICP_args.update(ICP_options)

        # Initialise Linear ICP and run (train and test called internally)
        ICPmod = InvariantCausalPrediction(reduced_environment_datasets, reduced_val_dataset, reduced_test_dataset, ICP_args)
        # Get results on test_dataset
        ICP_results_dict = ICPmod.results()
        icp_solution = ICP_results_dict['solution']

        ICP_results_dict['original_feature_indices'] = keep_columns_indices_to_original_features[
            ICP_results_dict['selected_feature_indices']]

        if config['verbose']:
            print(ICP_results_dict)

        to_bucket = ICP_results_dict['to_bucket']
        to_bucket_results.append(to_bucket)

        print("Finished ICP")


    #####################################################################################
    ################################# Non-Linear ICP ####################################

    if 'NLICP' in selected_models:
        from models.NonLinearInvariantCausalPrediction import NonLinearInvariantCausalPrediction
        # Set up ICP args
        NLICP_options = ensemble_options.get('NLICP', {})
        NLICP_args = {
            "max_set_size": 2,
            "alpha": 0.05,
            "seed": 12,
            "verbose": 1,
            "method": "MLP",
            "hidden_dim": 256
        }
        NLICP_args.update(NLICP_options)
        print('running nonlinear ICP')

        # Initialise Linear ICP and run (train and test called internally)
        NLICPmod = NonLinearInvariantCausalPrediction(reduced_environment_datasets, reduced_val_dataset, reduced_test_dataset, NLICP_args)
        # Get results on test_dataset
        NLICP_results_dict = NLICPmod.results()

        NLICP_results_dict['original_feature_indices'] = keep_columns_indices_to_original_features[
            NLICP_results_dict['selected_feature_indices']]

        if config['verbose']:
            print(NLICP_results_dict)

        to_bucket = NLICP_results_dict['to_bucket']
        to_bucket_results.append(to_bucket)

        print("Finished NLICP")
    #####################################################################################

    #####################################################################################
    ############################# LINEAR IRM ###############################

    if "LIRM" in selected_models:
        print("Running Linear IRM")

        from models.LinearInvariantRiskMinimization import LinearInvariantRiskMinimization
        # Setup Linear and Non-Linear IRM args
        LIRM_options = ensemble_options.get('LIRM', {})
        LIRM_args = {
            "use_icp_initialization": False,
            "verbose": 1,
            "n_iterations": 1000,
            "seed": 0,
            "lr": 0.001,
            "cuda": False
        }
        LIRM_args.update(LIRM_options)

        # If using ICP to init weights of IRM:
        if icp_solution and LIRM_args['use_icp_initialization']:
            print('Using ICP weights to initialize IRM')
            LIRM_args['ICP_weights'] = ICP_results_dict['feature_coeffients']
            if LIRM_args["use_reduced_feature_set"]:
                LIRM_args['ICP_weight_indices'] = ICP_results_dict['selected_feature_indices']
            else:
                LIRM_args['ICP_weight_indices'] = ICP_results_dict['original_feature_indices']

        #IRM_Red = LinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, LIRM_args)
        IRM_Red = LinearInvariantRiskMinimization(reduced_environment_datasets, reduced_val_dataset, reduced_test_dataset, LIRM_args)

        feat_red_irm_results_dict = IRM_Red.results()
        to_bucket = feat_red_irm_results_dict['to_bucket']
        to_bucket['method'] = 'Linear IRM'
        to_bucket_results.append(to_bucket)

        print("Finished Linear IRM")

    #####################################################################################
    ############################# NON-LINEAR IRM ###############################

    ''' Commented out for now as we are always running Non Linear IRM for Feature Reduction'''

    if "IRM" in selected_models:
        from models.NonLinearInvariantRiskMinimization import NonLinearInvariantRiskMinimization

        # Setup Linear and Non-Linear IRM args
        IRM_options = ensemble_options.get('IRM', {})
        IRM_args = {
            # Flad for model to use in Non-Linear IRM ['NN': MLP]
            "use_reduced_feature_set": False,
            "NN_method": "DNN",
            "verbose": 1,
            "n_iterations": 1000,
            "seed": 0,
            "hidden_dim": 256,
            "l2_regularizer_weight": 0.001,
            "lr": 0.001,
            "penalty_anneal_iters": 100,
            "penalty_weight":  10000.0,
            "cuda": False
        }
        IRM_args.update(IRM_options)

        print('Running IRM')
        if IRM_args["use_reduced_feature_set"]:
            IRM_NN = NonLinearInvariantRiskMinimization(reduced_environment_datasets, reduced_val_dataset,
                                                        reduced_test_dataset, IRM_args)
        else:
            IRM_NN = NonLinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, IRM_args)

        irm_results_dict = IRM_NN.results()

        if config['verbose']:
            print(irm_results_dict)

        to_bucket = irm_results_dict['to_bucket']
        to_bucket_results.append(to_bucket)

        print("Finished IRM")

    #####################################################################################
    ############################ SAVE RESULTS TO BUCKET #################################

    '''save_dict_to_json({'zero std columns removed': zero_var_columns},
                    config['results_directory'] + 'zero_std_columns.json')

    column_pairs_df.to_csv(config['results_directory'] + 'correlation_pairs.csv')'''

    save_dict_to_json({"results": to_bucket_results}, config['results_directory'] + 'results_for_bucket.json')

    if config['use_cloud']:
        save_json_to_bucket({"zero std columns removed": zero_var_columns},
                            config['bucket_path'] + config['bucket_exp_path'] + 'zero_var_columns.json', config['bucket_project'],
                            config['bucket_name'])

        save_dataframe_to_bucket(column_pairs_df, config['bucket_path'] + config['bucket_exp_path'] + 'correlation_pairs.csv',
                                config['bucket_project'], config['bucket_name'])

        save_json_to_bucket({"results": to_bucket_results}, config['bucket_path'] + config['bucket_exp_path'] + 'results.json',
                            config['bucket_project'], config['bucket_name'])

        save_json_to_bucket(config, config['bucket_path'] + config['bucket_exp_path'] + 'config.json', config['bucket_project'],
                            config['bucket_name'])

    #####################################################################################
    ##################### GATHER RESULTS AND PLOT FEATURE IMPORTANCE ####################

    for i, method_dict in enumerate(to_bucket_results):
        if 'method' in method_dict.keys():
            method = method_dict['method']

            """
            Structure of method_dict
            "to_bucket": {
                'method': 'Method Name',
                'features': [X1, X2, X4, X32, .. Xmax],
                'coefficients': [w1, w2, w4, w32, .. wmax],
                'pvals': [p1, p2, p4, p32, .. pmax] || p_total || None
                'test_acc': 0.97 || None
            }
            """

            print('Processing', method)
            coefs = pd.DataFrame()
            coefs['feature'] = method_dict['features']
            coefs['coefficient'] = method_dict['coefficients']
            coefs['pvals'] = method_dict['pvals']

            fname = config['results_directory'] + method + '_features.pdf'
            plot_most_predictive(coefs, fname)

    ##################################        END        ################################
    #####################################################################################


if __name__ == '__main__':
    import argparse
    import os
    import traceback

    parser = argparse.ArgumentParser(description='NASA FDL 2020 Astronaut Health Team 2: CRISP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path to config json file
    parser.add_argument('--experiment_config', default='experiment_configs/subject_experiments/ensemble_config.json')
    # Flag to use gcp file storage
    parser.add_argument('--use_cloud', action='store_true', default=False)

    opt = parser.parse_args()

    # Read json config file
    import json
    with open(os.path.join(os.getcwd(), opt.experiment_config)) as json_file:
        config = json.load(json_file)
        config['use_cloud'] = opt.use_cloud

    if config['verbose']:
        print('Loaded config for experiment:', config['name'], ' \n' + json.dumps(config, indent=4, sort_keys=True))

    # Create results folder to save outputs to if it doesnt already exist:
    cwd = os.getcwd()
    results_directory = os.path.join(cwd, 'results', config['short_name'])
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # Configure results directory in config and save experiment config to folder
    config['results_directory'] = results_directory + '/'
    save_dict_to_json(config, config['results_directory'] + 'experiment_config.json')

    # Configure data filepath to load .pickle file from
    config['data_options']['dataset_fp'] = os.path.join(os.getcwd(), config['data_options']['dataset_fp'])

    # If using cloud storage, generate unique experiment folder name to save results to
    if config['use_cloud']:
        bucket_fp = 'exp_y_' + config['data_options']['targets'][0] + '_env_' + config['data_options']['environments'][
            0] + '_' + config['short_name'] + '/'
        config['bucket_exp_path'] = bucket_fp

    # Run experiment, catch any errors and save them to bucket for debugging
    try:
        run(config)
    except Exception as e:
        tb = traceback.format_exc()
        print('Caught exception ', e, tb)
        if config['use_cloud']:
            save_json_to_bucket({'exception': str(e), 'traceback': str(tb)},
                                config['bucket_path'] + config['bucket_exp_path'] + 'excpetion.json', config['bucket_project'],
                                config['bucket_name'])
