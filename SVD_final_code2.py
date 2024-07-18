import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('Frequency_Load_Outage.csv')
df = df.apply(pd.to_numeric, errors='coerce')

total_samples = len(df)
sampling_rate = 60
total_time = len(df)/sampling_rate

time = np.linspace(0, total_time, total_samples)

plt.figure(figsize=(12, 6))
for bus in df.columns:
    if bus == 'Time':
        continue
    plt.plot(time, df[bus], label=bus)

disturbance_time = 1.9
plt.axvline(x=disturbance_time, color='red', linestyle='--', label='Disturbance')

plt.title('Voltage magnitudes for IEEE 14-bus System')
plt.xlabel('Time (seconds)')
plt.ylabel('V(in p.u.)')
plt.legend()
plt.grid(True)
plt.show()

df.drop(columns={'Time'}, inplace=True)

def add_awgn(signal, target_snr_db):
    sig_avg_watts = np.mean(signal ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    noisy_signal = signal + noise
    return noisy_signal

target_snr_db = 92
rows_to_add_noise = range(115, 170)
for column in df.columns:
    df.loc[rows_to_add_noise, column] = add_awgn(df.loc[rows_to_add_noise, column], target_snr_db)

ne = df.shape[1]
def output_rho(Y, e=3.29e-4):
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
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
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
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
    sui += rho * (h + n + 1)
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
l = fs
h = 100
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

print("Compression Ratio:", df.size/sui)

print("Our RMSE error is ")
print(math.sqrt(sui1/df.size))

print("Our MADE error is", sui2)

plt.figure()
timestamps = list(range(len(rho)))
plt.plot(time, rho, label='Estimated Rank (rho[t])')
plt.title('Estimated Rank vs. Timestamp')
plt.xlabel('Timestamp (t)')
plt.ylabel('Estimated Rank (rho[t])')
plt.legend()
plt.grid(True)
plt.show()

df_columns = df.columns
Y_final.columns = df_columns

num_rows = len(df)
step = total_time / num_rows
x_values = np.arange(0, total_time, step)

last_column_df = df.columns[-1] 
plt.plot(x_values, df[last_column_df], label="Original data", linestyle='-', color='blue')

num_rows = len(Y_final)
step = total_time / num_rows
x_values = np.arange(0, total_time, step)

last_column_Y_final = Y_final.columns[-1] 
plt.plot(x_values, Y_final[last_column_Y_final], label="Reconstructed data (progressive partitioning)", linestyle='--', color='red')

plt.xlabel('Time (s)')
plt.ylabel('V (p.u.)')
plt.title('Comparison of last PMU data between Original and Reconstructed Data')
plt.legend()
plt.show()