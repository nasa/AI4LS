from openfl.component.aggregation_functions import AggregationFunctionInterface
import numpy as np

class Pool(AggregationFunctionInterface):
    def __init__(self):
        """Compute pooled variance from variances of collaborators"""
        pass

    def call(self, local_tensors, db_iterator, tensor_name, fl_round, *__):
        """Aggregate tensors of variances

          Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training result.
                        These tensors are passed to the aggregator node after local learning.
                    - 'aggregated' indicates that tensor is a result of aggregation.
                        These tensors are sent to collaborators for the next round.
                    - 'delta' indicates that value is a difference between rounds
                        for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            np.ndarray: aggregated tensor
        """

        print("CUSTOM AGGREGATION", tensor_name, "fl_round:", fl_round)

        if tensor_name == "dummy":
            return 0
    
        variances = []
        sample_numbers = []
        for x in db_iterator:
   #         print(x.tensor_name, x.tags)
            if x.tensor_name == "feature_variances" and x.tags[0] == "variances":
                feature_variances = x.nparray
                variances.append(feature_variances)
                print("=========tags======:", x.tags)
                n_samples = int(x.tags[1].split("n_samples: ")[1])
                sample_numbers.append(n_samples)

        variances = np.array(variances)

#        print("========variances.shape=====:", variances.shape)
        output = pooled_variance(variances, sample_numbers)
#        print("=========output.shape======:", output.shape)
        print("RETURNING from custom aggregation", output.shape)
        return output
    
def pooled_variance(variances, sample_numbers):
    weighted_var_sum = 0
    n_samples_sum = 0

    for n_samples, var in zip(sample_numbers, variances):
        weighted_var_sum += (n_samples - 1) * var
        n_samples_sum += n_samples

    return weighted_var_sum / n_samples_sum 




