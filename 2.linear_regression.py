import numpy as np 
from numpy.typing import NDArray 

class Solution:
    
    def get_model_prediction(self, X:NDArray[np.float64], weights:NDArray[np.float64]) -> NDArray[np.float64]:
        model_prediction = np.matmul (X, weights)
        return np.round(model_prediction, 5)
    
    def get_error(self, model_prediction:NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        error = np.mean(np.square(model_prediction - ground_truth))
        return round(error, 5)
    

if __name__ == "__main__":
    # Test the class with the given inputs
    X=[[0.3745401188473625, 0.9507143064099162, 0.7319939418114051]]
    weights=[1.0, 2.0, 3.0]
    ground_truth=[[0.59865848],[0.15601864],[0.15599452]]

    solution = Solution()
    model_prediction = solution.get_model_prediction(X, weights)
    print(f"The prediction is: {model_prediction}")

    error = solution.get_error(model_prediction, ground_truth)
    print(f"The error is: {error}")
    
    