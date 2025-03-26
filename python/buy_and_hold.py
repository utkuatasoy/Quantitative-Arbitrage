import pandas as pd
import numpy as np

def calculate_portfolio_profit(csv_file_path, tickers, weights, start_date, end_date):
    """
    Belirtilen tarih aralığında (start_date, end_date),
    seçili hisselerin (tickers) fiyatlarını 'data.csv' dosyasından okur.
    Tangency portföyü ağırlıklarını (weights) kullanarak
    toplam portföy getirisini hesaplar.
    
    Parametreler:
    ------------
    csv_file_path : str
        CSV dosyasının yolu (örneğin "data.csv")
    tickers : list
        Portföydeki hisse sembolleri (sütun isimleri)
    weights : array-like
        Portföy ağırlıkları (toplamı 1 olacak şekilde)
    start_date : str
        Başlangıç tarihi, format: "YYYY-MM-DD"
    end_date : str
        Bitiş tarihi, format: "YYYY-MM-DD"
    
    Geri Dönüş:
    -----------
    float
        Bu tarih aralığındaki portföyün toplam getiri (ör. 0.35 = %35 getiri)
    """
    
    # CSV dosyasını oku
    data = pd.read_csv(csv_file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Belirtilen tarihler arasındaki veriyi filtrele
    data_filtered = data.loc[start_date:end_date, tickers].copy()
    
    # Eksik veriler varsa ileri doldurma
    data_filtered.ffill(inplace=True)
    
    # Günlük log getirileri hesapla
    log_returns = np.log(data_filtered / data_filtered.shift(1)).dropna()
    
    # Her bir hissenin toplam log getirisini hesapla
    # (start_date'ten end_date'e kadar kümülatif log getiri)
    total_log_return = log_returns.sum()
    
    # Log getiriden toplam basit getiriye geç: e^(toplam log getiri) - 1
    total_return = np.exp(total_log_return) - 1
    
    # Portföy getirisini ağırlıklarla hesapla
    portfolio_return = np.dot(weights, total_return)
    
    return portfolio_return

# Örnek kullanım:
if __name__ == "__main__":
    # data.csv dosyanızın yolu
    csv_file_path = "data/data.csv"
    
    # Tangency portföyünde kullanacağınız hisseler
    tickers = ['BRSAN.IS', 'ANSGR.IS', 'TURSG.IS', 'BTCIM.IS', 'OYAKC.IS']
    
    # Tangency portföyünden elde edilen örnek ağırlıklar
    tangency_weights = np.array([0.10827083, 0.28059675, 0.32216874, 0.14116987, 0.1477938])
    
    # 2024-01-01 ile 2025-12-31 arasındaki getiri
    profit_2024_2025 = calculate_portfolio_profit(
        csv_file_path=csv_file_path,
        tickers=tickers,
        weights=tangency_weights,
        start_date="2024-01-01",
        end_date="2025-12-31"
    )
    
    print(f"Portföyün 2024-2025 arasındaki toplam getiri: %{profit_2024_2025 * 100:.2f}")
