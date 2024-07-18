import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
start_time = time.time()

# Load data
df = pd.read_csv('Book3.csv')
df = df.apply(pd.to_numeric, errors='coerce')
columns_to_drop = ["Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"]
df = df.drop(columns=columns_to_drop)
#print(df.isnull())
df = df.drop(10800)

# Define time parameters
total_samples = len(df)  # Total samples per bus
total_time = 180  # Total time in seconds
sampling_rate = 60  # Samples per second

# Create time array
times = np.linspace(0, total_time, total_samples)

# Plotting
plt.figure(figsize=(12, 6))
for bus in df.columns:
    if bus == 'Domain':
        continue
    plt.plot(times, df[bus], label=bus)

# Highlight disturbance at t=9 sec
disturbance_time = 53 # Time of disturbance in seconds
plt.axvline(x=disturbance_time, color='red', linestyle='--', label='Disturbance')

plt.title('Voltage magnitudes for IEEE 14-bus System')
plt.xlabel('Time (seconds)')
plt.ylabel('V(in p.u.)')
plt.legend()
plt.grid(True)
plt.show()

df.drop(columns={'Domain'}, inplace=True)

# Function to add AWGN to a given signal
def add_awgn(signal, target_snr_db):
    sig_avg_watts = np.mean(signal ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Add AWGN to each column of your DataFrame
target_snr_db = 92 # Desired target SNR in dB
rows_to_add_noise = range(0, 9000)
for column in df.columns:
    df.loc[rows_to_add_noise, column] = add_awgn(df.loc[rows_to_add_noise, column], target_snr_db)

"""=============================================================================
Randomized SVD. See Halko, Martinsson, Tropp's 2011 SIAM paper:

"Finding structure with randomness: Probabilistic algorithms for constructing
approximate matrix decompositions"
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

def rsvd(A, rank, n_oversamples=None, n_subspace_iters=None,
         return_range=False):
    """Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
    """
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A.
    Q = find_range(A, n_samples, n_subspace_iters)

    # Stage B.
    B = Q.T @ A
    U_tilde, S, Vt = np.linalg.svd(B)
    U = Q @ U_tilde

    # Truncate.
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    # This is useful for computing the actual error of our approximation.
    if return_range:
        return U, S, Vt, Q
    return U, S, Vt

# ------------------------------------------------------------------------------

def find_range(A, n_samples, n_subspace_iters=None):
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    """
    m, n = A.shape
    O = np.random.randn(n, n_samples)
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)

# ------------------------------------------------------------------------------

def subspace_iter(A, Y0, n_iters):
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T @ Q)
        Q = ortho_basis(A @ Z)
    return Q

# ------------------------------------------------------------------------------

def ortho_basis(M):
    """Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    """
    Q, _ = np.linalg.qr(M)
    return Q


ne = df.shape[1] # this is dimensionality reduction function
def output_rho(Y, e=3.29e-4):
    h = Y.shape[0]
    n = Y.shape[1]
    U, S, Vt = rsvd(Y, min(h, n))
    error = ne*e;
    mode = 0
    for i in range(len(S)):
        if(S[i] > error * math.sqrt(Y.size)):
            mode = max((i+1), mode)
        else:
            break
    return mode


Y_final = pd.DataFrame()

# Compression function
def compress(Y, r):
    global Y_final
    h = Y.shape[0]
    n = Y.shape[1]
    U, S, Vt = rsvd(Y, r)
    S = np.diag(S)
    Y_approx = U[:, :r] @ S[0:r, :r] @ Vt[:r, :]
    Y_final = pd.concat([Y_final, pd.DataFrame(Y_approx)], ignore_index=True)
    return Y_approx

sui = 0
# Compression Ratio function
def CR(Y, rho):
    global sui
    h = Y.shape[0]
    n = Y.shape[1]
    CR = (h * n) / (rho * (h + n + 1))
    sui += rho *(h + n + 1)
    return CR

sui1 = 0
def RMSE(Y, Y_approx):
    global sui1
    h = Y.shape[0]
    n = Y.shape[1]
    frobenius_norm = np.linalg.norm(Y - Y_approx, 'fro')
    sui1 += (frobenius_norm * frobenius_norm)
    ans = frobenius_norm / math.sqrt(h * n)
    return ans

sui2 = 0

def MAE(Y, Y_approx):
    global sui2
    max_abs_deviation = np.abs(Y - Y_approx).max().max()
    sui2 = max(sui2, max_abs_deviation)
    ans = max_abs_deviation
    return ans



# Progressive Partitioning Algorithm
fs = 60
n = 14
l = fs
h = 200
rho_n = 1
phi = 0
n_phi = 0
phi = 0
rho_max = 0
alpha = 0.5
rho_n = 1
sigma = 0
ctr = 0
rho = [1] * df.shape[0]  # Increase the length of rho to accommodate the maximum value of time_stamp
Y = pd.DataFrame()  # Initialize a new buffer

new_buffer = True
for time_stamp in range(1, len(rho)):
    if new_buffer:
        Y = pd.DataFrame()  # Initialize a new buffer
        Y = pd.concat([Y, df.iloc[time_stamp:time_stamp + 1]], ignore_index=True)
        rho[time_stamp] = output_rho(Y)
    else:
        Y = pd.concat([Y, df.iloc[time_stamp:time_stamp + 1]], ignore_index=True)
        rho[time_stamp] = output_rho(Y)
    if phi == 0:
        rho[time_stamp] = output_rho(Y)
        if rho[time_stamp] > rho[time_stamp - 1]:
            phi = 1
            n_phi = 0
            sigma += rho[time_stamp]
            ctr += 1
            rho_max = max(rho_max, rho[time_stamp])
            rho_avg = sigma / ctr
            Y_approx = compress(Y, rho[time_stamp])
            print("The compression ratio is ")
            print(CR(Y, rho[time_stamp]))
            print("RMSE error")
            print(RMSE(Y, Y_approx))
            print("MADE error")
            print(MAE(Y, Y_approx))
            new_buffer = True
            continue
        elif rho[time_stamp] <= rho[time_stamp - 1]:
            if len(Y) >= h:
                Y_approx = compress(Y, rho[time_stamp])
                print("The compression ratio is ")
                print(CR(Y, rho[time_stamp]))
                print("RMSE error")
                print(RMSE(Y, Y_approx))
                print("MADE error")
                print(MAE(Y, Y_approx))
                new_buffer = True
                continue
            elif len(Y) < h:
                new_buffer = False
                continue
    elif phi == 1:
        sigma += rho[time_stamp]
        ctr += 1
        rho_max = max(rho_max, rho[time_stamp])
        rho_avg = sigma / ctr
        if rho_avg < alpha * rho_max:
            phi = 2
            Y_approx = compress(Y, rho[time_stamp])
            print("The compression ratio is ")
            print(CR(Y, rho[time_stamp]))
            print("RMSE error")
            print(RMSE(Y, Y_approx))
            print("MADE error")
            print(MAE(Y, Y_approx))
            new_buffer = True
            continue
        elif rho_avg >= alpha * rho_max:
            if rho[time_stamp] == rho_n:
                n_phi += 1
                if n_phi >= l:
                    phi = 0
                    n_phi = 0
                    Y_approx = compress(Y, rho[time_stamp])
                    print("The compression ratio is ")
                    print(CR(Y, rho[time_stamp]))
                    print("RMSE error")
                    print(RMSE(Y, Y_approx))
                    print("MADE error")
                    print(MAE(Y, Y_approx))
                    new_buffer = True
                    continue
                else:
                    if len(Y) >= h:
                        Y_approx = compress(Y, rho[time_stamp])
                        print("The compression ratio is ")
                        print(CR(Y, rho[time_stamp]))
                        print("RMSE error")
                        print(RMSE(Y, Y_approx))
                        print("MADE error")
                        print(MAE(Y, Y_approx))
                        new_buffer = True
                        continue
                    elif len(Y) < h:
                        new_buffer = False
                        continue
            else:
                n_phi = 0
                if len(Y) >= h:
                    Y_approx = compress(Y, rho[time_stamp])
                    print("The compression ratio is ")
                    print(CR(Y, rho[time_stamp]))
                    print("RMSE error")
                    print(RMSE(Y, Y_approx))
                    print("MADE error")
                    print(MAE(Y, Y_approx))
                    new_buffer = True
                    continue
                elif len(Y) < h:
                    new_buffer = False
                    continue

# Plot Columns of Y_final
num_rows = len(Y_final)
step = total_time / num_rows
x_values = np.arange(0, total_time, step)

for column in Y_final.columns:
    plt.plot(x_values, Y_final[column], label=column)

plt.xlabel('Time (s)')
plt.ylabel('V(p.u.)')
plt.title('Plot of Columns of Y_final')
plt.legend()
plt.show()
def compression_ratio(df, Y_final):
    # Calculate the size of the original matrix
    size_df = df.size
    
    # Calculate the size of the compressed matrix
    size_Y_final = Y_final.size
    
    # Calculate compression ratio
    compression_ratio = size_df / size_Y_final
    
    return compression_ratio

cr = compression_ratio(df, Y_final)
print("Compression Ratio:", df.size/sui)

print("Our RMSE error is ")
print(math.sqrt(sui1/df.size))

print("Our MADE error is", sui2)
# Plot Estimated Rank vs. Timestamp
plt.figure()
timestamps = list(range(len(rho)))
plt.plot(times, rho, label='Estimated Rank (rho[t])')
plt.title('Estimated Rank vs. Timestamp')
plt.xlabel('Timestamp (t)')
plt.ylabel('Estimated Rank (rho[t])')
plt.legend()
plt.grid(True)
plt.show()
print("--- {} seconds ---".format(time.time() - start_time))