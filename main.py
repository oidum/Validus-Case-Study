import numpy as np

# Adam Schneider
# Approximate time taken: 8 hours

# Constant variables
s_0 = 1
N = 2
p = 0.5


# Question 1
def EurCall1(v, K):
    # Up and down factors
    u = 1+v
    d = 1-v

    # Step 1: generate stock prices
    stock_prices = np.zeros([N+1, N+1])
    for i in range(N+1):
        for j in range(i+1):
            stock_prices[j, i] = s_0 * (u ** (i-j)) * (d**j)
    # print("Stock prices: \n", stock_prices)

    # Step 2: Find option prices
    option_prices = np.zeros([N+1, N+1])
    # Terminal nodes
    option_prices[:, N] = np.maximum(np.zeros(N+1), stock_prices[:, N] - K)
    # Work backwards
    for i in np.arange(N - 1, -1, -1):
        for j in np.arange(0, i+1):
            option_prices[j, i] = p*option_prices[j, i+1] + (1-p)*option_prices[j+1, i+1] # Find expected value (in this case the probabilities are the same)
    # print("Option prices: \n", option_prices)
    return option_prices[0, 0]


# Question 2
def EurCall2(K, V):
    # Initial conditions
    epsilon = 0.0001
    delta = 1
    v = 0.5
    jump = 0.25
    max_iter = 1000
    i = 0

    # Run loop while difference between V and V_temp is > epsilon
    while delta > epsilon and i < max_iter:
        V_temp = EurCall1(v, K)
        delta = abs(V-V_temp)

        # Check success conditions
        if delta < epsilon:
            break
        else:

            # Update v value and reduce jump size
            if V_temp < V:
                v = v + jump
            else:
                v = v - jump
            jump = jump/2
            i = i+1

        # Warning if maximum iterations reached
        if i == max_iter:
            print("Maximum iterations reached.")

    return v

    # Scrapped first pass where I would solve directly for v, instead I opted to approximate using EurCall1

    # option_prices = np.zeros([N+1, N+1])
    # prob = np.zeros(N+1)
    # stock_prices = np.zeros([N+1, N+1])
    # # Step 1: Work backwards form root node of option value tree to find terminal node values
    # for i in np.arange(1, N+1):
    #     for j in np.arange(i+1):
    #         option_prices[j, i]
    #
    # # Calculate probabilities of the terminal nodes
    # for j in range(N+1):
    #     prob[j] = 1/2**i * nCr(N, j)
    #
    # print(prob)
    #
    # # Step 2: Calculate stock prices
    return v

# Question 3
def USACall(v, K):
    # Up and down factors
    u = 1+v
    d = 1-v

    # Step 1: generate stock prices
    stock_prices = np.zeros([N+1, N+1])
    for i in range(N+1):
        for j in range(i+1):
            stock_prices[j, i] = s_0 * (u ** (i-j)) * (d**j)
    # print("Stock prices: \n", stock_prices)

    # Step 2: Find option prices
    option_prices = np.zeros([N+1, N+1])
    # Terminal nodes

    option_prices[:, N] = np.maximum(np.zeros(N+1), stock_prices[:, N] - K)
    # Work backwards
    # We have the option to exercise the option at every step
    for i in np.arange(N - 1, -1, -1):
        for j in np.arange(0, i+1):
            x_value = max(0, stock_prices[j, i] - K)  # Exercise value
            bin_value = p*option_prices[j, i+1] + (1-p)*option_prices[j+1, i+1]  # Binomial value
            option_prices[j, i] = np.maximum(bin_value, x_value)
    # print("Option prices: \n", option_prices)
    return option_prices[0, 0]


# Question 4
def expectation(v):
    # Up and down factors
    u = 1+v
    d = 1-v

    stock_prices = np.zeros([N+1, N+1])
    stock_prob = np.zeros([N+1, N+1])
    stock_expectation = np.zeros(N+1)

    for i in range(N+1):
        for j in range(i+1):
            stock_prices[j, i] = s_0 * (u ** (i-j)) * (d**j)
            stock_prob[j, i] = 1/2**i * nCr(i, j)

    for i in range(N+1):
        stock_expectation[i] = np.dot(stock_prices[:, i], stock_prob[:, i])
    return np.amax(stock_expectation)


#Question 5
def EurCall3():
    # My approach for this question was to vectorize the inputs for 2 and then calibrate a v for each option.
    return


# Factorial utility function
def fact(n):
    res = 1
    for i in range(2, n+1):
        res = res*i
    return res

# utility function to calculate probability in the tree
def nCr(n, r):
    return fact(n) / (fact(r) * fact(n - r))


if __name__ == '__main__':
    print("Question 1: ")
    print("Option value: $%.2f" % EurCall1(0.9, 0.5))
    print("Question 2: ")
    print("v Value: %.2f" % EurCall2(0.5, 0.78))
    print("Question 3: ")
    print("Option value: $%.2f" % USACall(0.9, 0.5))
    print("Question 4: ")
    print("Expectation is: $%.2f" % expectation(0.1))



