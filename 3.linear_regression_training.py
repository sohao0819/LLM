import numpy as np
from numpy.typing import NDArray

class Solution: 
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate 
    
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64],desired_weight: int) -> float:
        return - 2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N #get the partial derivative of weight array 
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64])-> NDArray[np.float64]:
         return np.squeeze(np.matmul(X, weights))
    

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int, 
        initial_weights: NDArray[np.float64]) -> NDArray[np.float64]:
    
        weights = initial_weights 
        N = len(X)
        learning_rate = self.learning_rate 

        for _ in range(num_iterations):
            # calcuate the current model prediction witht the current weight 
            model_prediction = self.get_model_prediction (X, weights)
            for i in range (len(weights)): # iterate through each weight 
                derivate = self.get_derivative(model_prediction, Y, N, X, i) 
                weights[i] = weights[i] - learning_rate * derivate # update the weight by substructing the learning rate times the derivative 
    
        return np.round(weights,5)


if __name__ == "__main__":
    # Test the class with the given inputs
    X = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float64)  # Ensure dtype is float64
    Y = np.array([6.0, 3.0], dtype=np.float64)  # Ensure dtype is float64
    num_iterations = 10 
    initial_weights = np.array([0.2, 0.1, 0.6], dtype=np.float64)  # Ensure dtype is float64

    solution = Solution()
    result = solution.train_model(X, Y, num_iterations, initial_weights)
    print(f"The prediction is: {result}")