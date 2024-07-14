import torch
import torch.nn
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

class Solution:

    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        m,n = to_reshape.shape
        reshaped_tensor = torch.reshape(to_reshape,(m*n//2,2))
        return reshaped_tensor
        pass

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        result_avg = torch.mean(to_avg, dim = 0)
        return result_avg
        pass

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        result_cat = torch.cat((cat_one, cat_two), 1)
        return result_cat
        pass

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        output = torch.nn.functional.mse_loss(prediction, target)
        return output
        pass

if __name__ == "__main__":
    # Test the class with the given inputs
    to_reshape = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ])
    
    
    to_avg = torch.tensor([
        [0.8088, 1.2614, -1.4371],
        [-0.0056, -0.2050, -0.7201]
    ])

    cat_one = torch.tensor([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]])
    cat_two = torch.tensor([[1.0, 1.0],[1.0, 1.0]])

    prediction = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])

    solution = Solution()
    result_to_reshape = solution.reshape(to_reshape)
    result_to_average = solution.average(to_avg)
    result_concatenate = solution.concatenate(cat_one,cat_two)
    result_get_loss = solution.get_loss(prediction, target)
    
    print(f"Result of reshape: \n{result_to_reshape}")
    print(f"Result of average: \n{result_to_average}")
    print(f"Result of concatenate: \n{result_concatenate}")
    print(f"Result of get_loss: \n{result_get_loss}")
