
def calculate_leibniz_series(n_terms):
    pi_approximation = 0
    for i in range(n_terms):
        # The series is 1 - 1/3 + 1/5 - 1/7 + ...
        # i=0: + 1/1
        # i=1: - 1/3
        # i=2: + 1/5
        # The denominator follows the pattern 2*i + 1
        term = 1 / (2 * i + 1)
        if i % 2 == 0:
            pi_approximation += term
        else:
            pi_approximation -= term
    
    return pi_approximation * 4

n = 1000000
result = calculate_leibniz_series(n)
print(f"The approximation of pi using the first {n} terms is: {result}")
