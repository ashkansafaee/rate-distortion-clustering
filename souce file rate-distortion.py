# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(
    "/Users/ash/Documents/Dokument – Ashkans MacBook Air/MT7037 - Statistical Information Theory/Data-Exercise2-2025",
    sep=r"\s+",
    header=None
)

df.columns = ["X", "Y"]
print(df.head())

# %%
fig, ax = plt.subplots()
ax.scatter(df['X'], df['Y'], alpha = 0.8)
ax.set_title('Data Points')
plt.show()

# %%
from scipy.spatial import distance_matrix

# Extract the coordinates as a (200, 2) numpy array
coords = df[['X', 'Y']].values

# Compute the distance matrix.
# Returns the matrix of all pair-wise distances.
dist_matrix = distance_matrix(coords, coords, p=2)

# %% [markdown]
# Finding the rate distortion function is a variation problem that can be solved by introducing a Lagrange multiplier, $\beta$, for the constrained expected distortion (Tishby et al. 2019). We want to use Rate Distortion theory for clustering were we solve this problem by minimizing $I(T,X)$ with regard to $P(T|X)$ subject to $D$. Our expression is $L = I(X, T)+ \beta D$, with $D$ being the average distortion over clusters and within-cluster distances (cohesion, compactness) $E_{p(x,t)}[d(x,t)]$.
# 
# We implement the self-consistent equations: 
# $$
# p_n(t|x)=\frac{p(t)}{Z_n(x,\beta)}\exp[-\beta d(x,t)] \quad 
# p_{n+1}(t) = \sum_x p(x)p_t(t|x)
# $$
# that need to be satisfied simultaneously. We repeat the procedure until the algorithm converges at $L^{now} - L^{old}$ at an acceptable treshold. 

# %%
def blahut_arimoto(dist_matrix, beta, Nc, thresh: 1e-12, max_iter = 3000):
    """
    Implements the Blahut-Arimoto algorithm to find the optimal clustering membership 
    probability p(t|x) that minimizes the Rate-Distortion functional.

    - Initializes p(t|x) with random values to break symmetry and explore local optima (as per Exercise 2b).
    - Iteratively updates the cluster marginals p(t) and membership p(t|x) until the 
    change in probabilities falls below the threshold.
    - Normalizes conditional probabilities to ensure they form a valid PMF (sum to 1).

    Args:
        dist_matrix (list)): pairwise distances in a 200x200 matrix
        beta (integer): values of constrained expected distortion
        Nc (integer): Number of classes
        thresh (1e): treshold
        max_iter (int): set to 3000.

    Returns:
        updated membership probabilites 
    """
    N = dist_matrix.shape[0]
    d = dist_matrix[ :, :Nc]

    # Initialize p(x) where later when we run the algorithm
    # we will give it uniform probability 1/200
    p_x = np.ones(N) / N

    # Initialize p(t|x) at random
    p_t_given_x = np.random.rand(N, Nc)
    # Normalize to ensure rows sum to 1
    p_t_given_x /= p_t_given_x.sum(axis=1, keepdims=True)

    for i in range(max_iter):
        old = p_t_given_x.copy()

        # Update marginals p(t)
        p_t = np.sum(p_x[:, np.newaxis] * p_t_given_x, axis=0)

        # Blahut-Arimoto Update 
        exponent = -beta * d
        # Calculate the numerator : P(t)*exp(-beta * d(x,t))
        p_t_given_x = p_t[None, :] * np.exp(exponent)
        # Normalize to ensure rows sum to 1
        p_t_given_x /= p_t_given_x.sum(axis = 1, keepdims = True)

        # Check for convergence
        if np.linalg.norm(p_t_given_x - old) < thresh:
            print(f"Converged at iteration {i}")
            break

    return p_t_given_x, p_t

# According to Rate Distortion Theory, we must minimize $\mathcal{L} = I(\tilde{X}; X) - \beta D, with $\beta \geq 0$. 
def mutual_info_XT(p_x, p_t_given_x, p_t):
    """
    Calculates the Mutual Information I(X; T) between data points (X) and clusters (T).
    - Represents the 'Rate' or complexity of the compressed representation.
    - Measures how much information about the original coordinates is preserved in the 
    cluster mapping.
    - Uses a small epsilon (1e-12) to avoid log(0) errors during computation.

    Args:
        p_x (array): Uniform input probability distribution (1/200 for each point).
        p_t_given_x (matrix): The conditional membership probability identified by BA algorithm.
        p_t (array): The marginal probability of each cluster.

    Returns:
        The mutual information in bits (i.e. complexity)
    """
    eps = 1e-12 #adding epsilon to avoid singular values at log(0)
    p_xt = p_x[:, np.newaxis] * p_t_given_x
    inner_term = np.log((p_t_given_x + eps) / (p_t + eps))
    mutual_info = np.sum(p_xt * inner_term)

    return max(0, mutual_info) #the self consistent equations are satisfied simul. at the minima of the functional

# %% [markdown]
# Computing the Average distortion $$D = \sum_x \sum_t p(x, t) d(x, t) = \sum_x p(x) \sum_t p(t|x) d(x, t)$$ 
# we get the following function (lecture notes, https://web.stanford.edu/class/ee368b/Handouts/04-RateDistortionTheory.pdf):

# %%
# Computing D, average distortion (average over clusters, and within-cluster distances (cohesion, compactness))
def average_distortion(p_x, p_t_given_x, dist_matrix):
    """
    Computes the expected Euclidean distortion D for the given clustering.
    - Quantifies the distance between data points and their cluster representatives.
    - Calculated as expectation E[d(x, t)] over the joint probability p(x, t).
    - Uses the distance matrix from exercise 2a. 

    Args:
        p_x (array): Uniform probability distribution (1/200).
        p_t_given_x (marix): Membership probability p(x|t).
        dist_matrix (matrix): Distance matrix d(x, t). 

    Returns:
        The average distortion. 
    """
    return np.sum(p_x[ :, None] * p_t_given_x * dist_matrix)

# %%
# Initializing p_x lenght 200, uniform for all i
N = 200
p_x_uni = np.ones(N) / N

# Create a list of betas starting at 0.1 and multiplying with 2 until max 40 is reached 
betas = []
b = 1
while b <= 40:
    betas.append(b)
    b *= 2

# Run simulation looping over different betas and Nc 2, 3, 5
Nc_values = [2, 3, 4]
n_runs = 10 #multiple run each starting with random 
results = []

for nc in Nc_values:
    # Slice dist_matrix to match number of symbols nc used in BA-algorithm
    d_sliced = dist_matrix[:, :nc]
    
    # Loop over different betas and save result in a list 
    for b in betas:
        best_I = None
        best_D = None
        min_lagrangian = float('inf') #unbounded upper value 'inf'

        # Run code n_runs to ensure we run code multiple times each starting at random
        for run in range(n_runs):        
            p_tx, p_t = blahut_arimoto(d_sliced, b, nc, thresh=1e-12)
            Ixt = mutual_info_XT(p_x_uni, p_tx, p_t)
            D = average_distortion(p_x_uni, p_tx, d_sliced)
            
            # Save current Lagrangian
            current_lagrangian = Ixt + b * D

            # Add condition to store min Lagrangian
            if current_lagrangian < min_lagrangian:
                min_lagrangian = current_lagrangian
                best_I = Ixt
                best_D = D

        # Save variables from min Lagrangian in results
        results.append({'Nc': nc, 
                        'beta': b, 
                        'I_XT' : best_I, 
                        'D': best_D})

# %%
import seaborn as sns

# Store results in data frame for plotting
df_ic = pd.DataFrame(results)

# To mimic the info-curve from class we invert the graph
df_ic['Negative_D'] = -df_ic['D']

plt.figure(figsize=(8, 6))
sns.lineplot(data = df_ic, x = 'I_XT', y = 'Negative_D', hue = 'Nc', marker = 'o')
plt.title("Information Curve: Growth from Compressed to Detailed")
plt.xlabel("Complexity -> Less Compressed")
plt.ylabel("Distortion -> Less Distorted")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


