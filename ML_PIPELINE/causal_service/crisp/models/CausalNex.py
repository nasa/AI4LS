from causalnex.structure.notears import from_pandas
from causalnex.structure import DAGRegressor, DAGClassifier
import numpy as np
import torch
import pandas as pd

# def order_causality(sm):
#     temp = {}
#     for k, v in sm.adj.items():
#         for i, j in v.items():
#             temp[k] = np.abs(j["weight"])
#     temp = np.array(sorted(zip(temp.values(), temp.keys()), reverse=True))
#     ordered_names = temp[:,1]
#     ordered_values = np.float64(temp[:,0])
#     return ordered_names, ordered_values

class CausalNexClass(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        self.args = args
        #self.cuda = torch.cuda.is_available() and args.get('cuda', False)
        self.cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if self.cuda else "cpu")
        #self.cuda = False
        self.input_dim = environment_datasets[0].get_feature_dim()
        self.output_dim = len(np.unique(environment_datasets[0].targets))
        self.test_dataset = test_dataset

        self.feature_names = test_dataset.predictor_columns
        
        
        all_dataset = torch.utils.data.ConcatDataset(environment_datasets)
                
        X = []
        yy = []
        for x,y in all_dataset:
            '''if self.cuda:
                X.append(x.cuda().numpy())
                yy.append(y.cuda().item())
            else:
                X.append(x.cpu().numpy())
                yy.append(y.cpu().item())'''
            X.append(x.numpy())
            yy.append(y.item())
                
        if args.get("output_data_regime")=="real-valued":
            reg = DAGRegressor(tabu_child_nodes=[i for i,k in enumerate(self.feature_names) if k != 'Target' ],
                   dependent_target=True, enforce_dag=True, standardize=True)        
        else:
            yy = np.int64(yy)
            reg = DAGClassifier(tabu_child_nodes=[i for i,k in enumerate(self.feature_names) if k != 'Target' ],
                   dependent_target=True, enforce_dag=True, standardize=False)

        reg.fit(X,yy)
        df = pd.DataFrame(X, columns = test_dataset.predictor_columns)
        df['Target'] = yy
#         sm = from_pandas(df, tabu_child_nodes=list(df.keys().drop('Target')))

        # TODO JC added [0]
        self.importance = reg.feature_importances_[0]
        # TODO JC added [0]
        self.coef_ = reg.coef_[0]
        
        # computing the test accuracy
        self.batch_size = args.get('batch_size', 8)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        
        for i, (inputs, targets) in enumerate(self.test_loader):
            if i == 0 :
                X_t = inputs.numpy()
                yy_t = targets.numpy()
            else:
                X_t = np.concatenate((X_t, inputs.numpy()))
                yy_t = np.concatenate((yy_t, targets.numpy()))
        
        if args.get("output_data_regime")=="real-valued":
            yy_t = yy_t
        else:
            yy_t = np.int64(yy_t)
        
        self.acc = reg.score(X_t,yy_t)
        print('##### acc', self.acc)

    def train(self):
        return None

    def solution(self):
        return None

    def test(self, loader):
        return None

    def results(self):
        return {
            "test_logits" : None,
            "test_acc": self.acc, 
            "test_nll": None,
            "test_probs": None,
            "test_labels": None,
            "feature_coeffients": None,
            'to_bucket': {
                "test_logits" : None,
                'method': "CausalNex",
                'features': list(self.feature_names),
                'coefficients': self.coef_.tolist(),
                'pvals': self.importance.tolist(),
                'test_acc': self.acc,
                'test_acc_std': None,
                'coefficient_correlation_matrix': None
            }
        }

    def mean_nll(self, logits, y):
        return None

    def acc_preds(self, logits, y):
        return None

    def mean_accuracy(self, logits, y):
        return None
        
    def std_accuracy(self, logits, y):
        return None
    
    def pretty(self, vector):
        return None

    def get_corr_mat(self):
        return None
    
def update_cumulative_order(cum_ord, columns, sm):
    temp = {}
    for k, v in sm.adj.items():
        for i, j in v.items():
            temp[k] = np.abs(j["weight"])
    temp = np.array(sorted(zip(temp.values(), temp.keys()), reverse=False))
    ordered_names = temp[:,1]
    ordered_values = np.float64(temp[:,0])
    
    for i in range(len(cum_ord)):
        cum_ord[i]+= float(np.argwhere(ordered_names==columns[i]).item())
    
    return cum_ord  
    
class CausalNexClassEnv(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        self.args = args
        self.cuda = torch.cuda.is_available() and args.get('cuda', False)
        self.input_dim = environment_datasets[0].get_feature_dim()
        self.output_dim = len(np.unique(environment_datasets[0].targets))
        self.test_dataset = test_dataset

        self.feature_names = test_dataset.predictor_columns
        
        cumulative_orders = [0]*len(test_dataset.predictor_columns)
        
        for e, env in enumerate(environment_datasets):

            X, yy = env.get_all()
    
            df = pd.DataFrame(X.numpy(), columns = test_dataset.predictor_columns)
            df['Target'] = yy.numpy()
        
            sm = from_pandas(df, tabu_child_nodes=list(df.keys().drop('Target')))
        
            cumulative_orders = update_cumulative_order(cumulative_orders, self.feature_names, sm)
        
        temp = np.array(sorted(zip(cumulative_orders, self.feature_names), reverse=True))
        self.ordered_values = np.float64(temp[:,0])
        self.ordered_names = temp[:,1]
        
        
        

    def train(self):
        return None

    def solution(self):
        return None

    def test(self, loader):
        return None

    def results(self):
        return {
            "test_logits" : None,
            "test_acc": None, 
            "test_nll": None,
            "test_probs": None,
            "test_labels": None,
            "feature_coeffients": None,
            'to_bucket': {
                "test_logits" : None,
                'method': "CausalNex",
                'features': list(self.ordered_names),
                'coefficients': self.ordered_values,
                'pvals': None,
                "test_logits" : None,
                'test_acc': None, 
                'test_acc_std': None,
                'coefficient_correlation_matrix': None
            }
        }

    def mean_nll(self, logits, y):
        return None

    def acc_preds(self, logits, y):
        return None

    def mean_accuracy(self, logits, y):
        return None
        
    def std_accuracy(self, logits, y):
        return None
    
    def pretty(self, vector):
        return None

    def get_corr_mat(self):
        return None