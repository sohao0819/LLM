class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimizer = init 

        for _ in range(iterations):
            derivative = 2 * minimizer 
            minimizer = minimizer - learning_rate * derivative

        return round(minimizer, 5)
    
if __name__ == "__main__":
    # Test the class with the given inputs
    iterations = 10
    learning_rate = 0.01
    init = 5

    solution = Solution()
    result = solution.get_minimizer(iterations, learning_rate, init)
    print(f"The minimized value is: {result}")