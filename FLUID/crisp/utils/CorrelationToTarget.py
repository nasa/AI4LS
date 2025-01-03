import numpy as np
import pandas as pd


class CorrelationToTarget(object):
    def __init__(self, environment_datasets, val_dataset, test_dataset, args):
        self.max_features = args.get('max_features', False)
        self.predictor_columns = test_dataset.predictor_columns
        self.target_columns = test_dataset.target_columns

        all_x = []
        all_y = []
        for e in environment_datasets:
            e_x, e_y = e.get_all()
            all_x.append(e_x.numpy().squeeze())
            all_y.append(e_y.numpy())
        v_x, v_y = val_dataset.get_all()
        t_x, t_y = test_dataset.get_all()
        all_x.append(v_x.numpy().squeeze())
        all_x.append(t_x.numpy().squeeze())
        all_y.append(v_y.numpy())
        all_y.append(t_y.numpy())

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_x = np.vstack(all_x)
        all_y = np.vstack(all_y)

        all_x = np.concatenate((all_x, all_y), axis=1)
        all_df = pd.DataFrame(all_x, columns=self.predictor_columns + self.target_columns)

        all_corr = all_df.corr()
        target_corr = all_corr[self.target_columns].squeeze()
        target_corr = pd.DataFrame({'corr': target_corr, 'name': self.predictor_columns + self.target_columns})
        target_corr['sort'] = target_corr['corr'].abs()
        target_corr = target_corr.sort_values('sort', ascending=False)
        target_corr = target_corr.drop(columns=['sort'])
        self.target_corr_df = target_corr
