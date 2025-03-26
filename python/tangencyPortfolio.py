
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.optimize as sco

# %%
##########################
#       PART 1           #
# Data Collection and    #
# Return Calculation     #
##########################

# Veriler artık API'den değil, mevcut CSV dosyasından yüklenecek.
csv_file_path = "data.csv"  # data.csv dosyanızın yolunu belirtin

# İlgilenilen hisseler (CSV dosyasında sütun isimleri olarak bulunmalıdır)
tickers = ['BRSAN.IS', 'ANSGR.IS', 'TURSG.IS', 'BTCIM.IS', 'OYAKC.IS']

# CSV dosyasını oku
data = pd.read_csv(csv_file_path)

# 'Date' sütununu datetime formatına çevir ve DataFrame'in indeksine ata
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 2023-01-01 ile 2024-12-31 arasındaki verileri filtrele ve yalnızca seçili hisselere odaklan
data_filtered = data.loc['2023-01-01':'2024-12-31', tickers]

print("Filtrelenmiş Veri Örneği:")
print(data_filtered.head())

# Filtrelenmiş veriyi 'combined_data' olarak kullanıyoruz
combined_data = data_filtered.copy()

# Eksik veriler varsa ileri doldurma yöntemiyle tamamla
combined_data = combined_data.ffill()

# Günlük log getirileri hesapla: R_t = ln(P_t/P_t-1)
log_returns = np.log(combined_data / combined_data.shift(1)).dropna()

# Günlük log getirilerin tanımlayıcı istatistikleri
desc_stats = log_returns.describe().T[['min', 'max', 'mean', '50%', 'std']]
desc_stats.rename(columns={'50%': 'median'}, inplace=True)

# Günlük log getiriler için varyans-kovaryans matrisi
var_cov_matrix = log_returns.cov()

# PART 1 çıktıları
print("\nTicker Sembolleri:", tickers)
print("\nGünlük Log Getirilerin Tanımlayıcı İstatistikleri:")
print(desc_stats)
print("\nGünlük Log Getirilerin Varyans-Kovaryans Matrisi:")
print(var_cov_matrix)

# Kapanış fiyatlarını zaman serisi grafiği olarak çizdir
# plt.figure(figsize=(14, 7))
# for ticker in combined_data.columns:
#     plt.plot(combined_data.index, combined_data[ticker], label=ticker)
# plt.xlabel('Tarih')
# plt.ylabel('Fiyat')
# plt.title('Seçili Hisselerin Kapanış Fiyatları (2023-2024)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Günlük log getiriler grafiği
# plt.figure(figsize=(14, 7))
# for ticker in log_returns.columns:
#     plt.plot(log_returns.index, log_returns[ticker], label=ticker)
# plt.xlabel('Tarih')
# plt.ylabel('Günlük Log Getiri')
# plt.title('Seçili Hisselerin Günlük Log Getirileri (2023-2024)')
# plt.legend()
# plt.grid(True)
# plt.show()

# %%
print("Toplam Gün Sayısı:", log_returns.shape[0])

# %%
##########################
#       PART 2           #
# Portfolio Construction #
# and Risk-Return        #
# Analysis               #
##########################

# Günlük getirileri yıllıklaştırma (işlem günü sayısı üzerinden)
trading_days = log_returns.shape[0]
asset_mean_daily = log_returns.mean()
asset_std_daily  = log_returns.std()

asset_mean_annual = asset_mean_daily * trading_days
asset_std_annual  = asset_std_daily * np.sqrt(trading_days)
cov_daily = log_returns.cov()
cov_annual = cov_daily * trading_days

# Örnek portföy ağırlık kombinasyonları (tüm ağırlıklar toplamı 1 olacak şekilde)
weight_combinations = [
    np.array([0.40, 0.20, 0.20, 0.10, 0.10]),
    np.array([0.30, 0.30, 0.20, 0.10, 0.10]),
    np.array([0.25, 0.25, 0.25, 0.15, 0.10]),
    np.array([0.20, 0.20, 0.20, 0.20, 0.20]),  # Eşit dağılım
    np.array([0.50, 0.20, 0.10, 0.10, 0.10]),
    np.array([0.35, 0.25, 0.15, 0.15, 0.10]),
    np.array([0.45, 0.25, 0.15, 0.10, 0.05]),
    np.array([1/5, 1/5, 1/5, 1/5, 1/5]),       # Eşit dağılım (kesirli ifade)
]

# Yıllık risk free oran (örnek olarak %26)
rf = 0.26

# Portföy istatistiklerini hesapla ve DataFrame'e kaydet
portfolio_stats = pd.DataFrame(columns=["Expected Return", "Variance", "Std Dev", "Sharpe Ratio"])
for i, w in enumerate(weight_combinations):
    port_return = np.dot(w, asset_mean_annual)
    port_variance = np.dot(w.T, np.dot(cov_annual, w))
    port_std = np.sqrt(port_variance)
    sharpe_ratio = (port_return - rf) / port_std
    portfolio_stats.loc[f"Portfolio {i+1}"] = [port_return, port_variance, port_std, sharpe_ratio]

print("\nPortföy İstatistikleri (Yıllıklaştırılmış):")
print(portfolio_stats)

# %%
##########################
#       PART 3           #
# Efficient Frontier and #
#     Interpretation     #
##########################

n = len(asset_mean_annual)  # Varlık sayısı

# Portföy performansını (beklenen getiri ve varyans) hesaplayan yardımcı fonksiyon
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    var = np.dot(weights.T, np.dot(cov_matrix, weights))
    return ret, var

# Portföy varyansını minimize eden fonksiyon (optimizasyonda kullanılacak)
def minimize_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Kısıt: Ağırlıkların toplamı 1 olmalı
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Ağırlık sınırları: 0 ile 1 arasında (açığa satış yok)
bounds = tuple((0, 1) for asset in range(n))

# --- Minimum Varyanslı Portföy ---
min_var_solution = sco.minimize(minimize_variance, n*[1./n],
                                args=(cov_annual,),
                                method='SLSQP', bounds=bounds, constraints=constraints)
min_var_weights = min_var_solution.x
min_var_return, min_var_variance = portfolio_performance(min_var_weights, asset_mean_annual, cov_annual)
min_var_std = np.sqrt(min_var_variance)
min_var_sharpe = (min_var_return - rf) / min_var_std

print("\nMinimum Varyanslı Portföy:")
print("Ağırlıklar:", min_var_weights)
print("Beklenen Getiri:", min_var_return)
print("Standart Sapma:", min_var_std)
print("Sharpe Oranı:", min_var_sharpe)

# --- Tangency Portföyü (Sharpe Oranını Maksimize Eden) ---
def negative_sharpe(weights, mean_returns, cov_matrix, rf):
    ret, var = portfolio_performance(weights, mean_returns, cov_matrix)
    std = np.sqrt(var)
    return - (ret - rf) / std

tan_solution = sco.minimize(negative_sharpe, n*[1./n],
                            args=(asset_mean_annual, cov_annual, rf),
                            method='SLSQP', bounds=bounds, constraints=constraints)
tan_weights = tan_solution.x
tan_return, tan_variance = portfolio_performance(tan_weights, asset_mean_annual, cov_annual)
tan_std = np.sqrt(tan_variance)
tan_sharpe = (tan_return - rf) / tan_std

print("\nTangency Portföyü:")
print("Ağırlıklar:", tan_weights)
print("Beklenen Getiri:", tan_return)
print("Standart Sapma:", tan_std)
print("Sharpe Oranı:", tan_sharpe)

# --- Efficient Frontier ---
# Farklı hedef getiriler için varyansı minimize ederek efficient frontier oluşturuluyor
target_returns = np.linspace(min(asset_mean_annual), max(asset_mean_annual), 50)
efficient_portfolios = []
for target in target_returns:
    constraints_target = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, asset_mean_annual) - target}
    )
    result = sco.minimize(minimize_variance, n*[1./n],
                          args=(cov_annual,), method='SLSQP',
                          bounds=bounds, constraints=constraints_target)
    if result.success:
        efficient_portfolios.append(result)
    else:
        efficient_portfolios.append(None)

frontier_returns = []
frontier_std = []
for opt in efficient_portfolios:
    if opt is not None:
        ret, var = portfolio_performance(opt.x, asset_mean_annual, cov_annual)
        frontier_returns.append(ret)
        frontier_std.append(np.sqrt(var))

# Efficient Frontier, minimum varyanslı ve tangency portföyü ile risk-free oranı grafiği
plt.figure(figsize=(10, 6))
plt.plot(frontier_std, frontier_returns, 'b--', label='Efficient Frontier')
plt.scatter(min_var_std, min_var_return, marker='*', s=200, label='Minimum Varyanslı')
plt.scatter(tan_std, tan_return, marker='*', s=200, label='Tangency Portföyü')
plt.scatter(0, rf, marker='o', s=100, label='Risk-Free Oran (26%)')
plt.xlabel('Yıllıklaştırılmış Standart Sapma (Risk)')
plt.ylabel('Yıllıklaştırılmış Beklenen Getiri')
plt.title('Efficient Frontier')
#plt.ylim((-1, 2))  # veya 
plt.ylim((0, 2))   # gibi
plt.legend()
plt.grid(True)
plt.show()
