To accomplish the assignment, I created a Python script that calculates the Leibniz series sum up to 1,000,000 terms and multiplies the result by 4.

### Process:
1.  **Script Creation:** I wrote a Python file named `calculate_pi.py` that implements the series $4 \times \sum_{i=0}^{n-1} \frac{(-1)^i}{2i+1}$.
2.  **Execution:** I ran the script using the sandbox environment to compute the value for $n = 1,000,000$.
3.  **Result:** The script successfully calculated the approximation of $\pi$ using the defined number of terms.

### Final Result:
The calculation for the first 1,000,000 terms of the series multiplied by 4 is:
**3.1415916535897743**

***

### Complete Python Content (`calculate_pi.py`):
```python
def calculate_leibniz_series(n_terms):
    pi_approximation = 0
    for i in range(n_terms):
        # The series is 1 - 1/3 + 1/5 - 1/7 + ...
        # Term formula: (-1)^i / (2*i + 1)
        term = ((-1) ** i) / (2 * i + 1)
        pi_approximation += term
    
    return pi_approximation * 4

if __name__ == "__main__":
    n = 1_000_000
    result = calculate_leibniz_series(n)
    print(f"Approximation of pi using {n} terms: {result}")
```