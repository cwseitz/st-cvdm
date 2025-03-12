import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator

path = '/N/slate/cwseitz/st_cvdm/Sim/4x/N-100_N0-500-1000/eval_data/'
df100 = pd.read_csv(path + 'N100-error.csv')
path = '/N/slate/cwseitz/st_cvdm/Sim/4x/N-200_N0-500-1000/eval_data/'
df200 = pd.read_csv(path + 'N200-error.csv')
path = '/N/slate/cwseitz/st_cvdm/Sim/4x/N-500_N0-500-1000/eval_data/'
df500 = pd.read_csv(path + 'N500-error.csv')

def compute_stats(df):
    grouped = df.groupby(['label', 'prefix', 'idx'])
    xacc = [np.mean(group['x_err'].values) for _, group in grouped]
    yacc = [np.mean(group['y_err'].values) for _, group in grouped]
    N0 = [group['N0'].values[0] for _, group in grouped]
    return np.array(N0), np.array(xacc), np.array(yacc)

N0_100, xacc_100, yacc_100 = compute_stats(df100)
N0_200, xacc_200, yacc_200 = compute_stats(df200)
N0_500, xacc_500, yacc_500 = compute_stats(df500)

pixel_size = 25.0
bins = np.arange(500, 1000, 100)

def bin_std(N0, acc):
    return np.array([np.std(acc[(N0 >= bins[i]) & (N0 < bins[i+1])]) for i in range(len(bins) - 1)])

xstd_binned_100 = bin_std(N0_100, xacc_100)
ystd_binned_100 = bin_std(N0_100, yacc_100)
xstd_binned_200 = bin_std(N0_200, xacc_200)
ystd_binned_200 = bin_std(N0_200, yacc_200)
xstd_binned_500 = bin_std(N0_500, xacc_500)
ystd_binned_500 = bin_std(N0_500, yacc_500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

ax1.plot(bins[:-1], xstd_binned_100 * pixel_size, 'x', color='red', label=r'$\rho = 100$', markersize=7)
ax1.plot(bins[:-1], xstd_binned_200 * pixel_size, 'x', color='blue', label=r'$\rho = 200$', markersize=7)
ax1.plot(bins[:-1], xstd_binned_500 * pixel_size, 'x', color='gray', label=r'$\rho = 500$', markersize=7)

ax1.set_xscale('log')
ax1.set_xticks(bins[:-1])
ax1.set_xlabel('Photons', fontsize=16)
ax1.set_ylabel(r'$\sigma_{u}$ (nm)', fontsize=16)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid()

ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.get_major_formatter().set_scientific(True)
ax1.xaxis.get_major_formatter().set_powerlimits((0, 0))
ax1.yaxis.set_major_locator(MultipleLocator(5))

ax2.plot(bins[:-1], ystd_binned_100 * pixel_size, 'x', color='red', label=r'$\rho = 100$', markersize=7)
ax2.plot(bins[:-1], ystd_binned_200 * pixel_size, 'x', color='blue', label=r'$\rho = 200$', markersize=7)
ax2.plot(bins[:-1], ystd_binned_500 * pixel_size, 'x', color='gray', label=r'$\rho = 500$', markersize=7)

ax2.set_xscale('log')
ax2.set_xticks(bins[:-1])
ax2.set_xlabel('Photons', fontsize=16)
ax2.set_ylabel(r'$\sigma_{v}$ (nm)', fontsize=16)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.xaxis.get_major_formatter().set_scientific(True)
ax2.xaxis.get_major_formatter().set_powerlimits((0, 0))
ax2.yaxis.set_major_locator(MultipleLocator(5))

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,fontsize=12)
ax2.grid()
plt.tight_layout()
plt.savefig('/N/slate/cwseitz/st_cvdm/Sim/4x/Figure-1.png',dpi=300)
plt.show()

