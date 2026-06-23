import pandas as pd
import numpy as np
import os
import time
import logging 
import yfinance as yf
from moexalgo import Market, Ticker   
from functools import cache

logging.basicConfig(level=logging.INFO)
data_logger = logging.getLogger(__name__)
        
def get_tickers_from_market(market_name:str="NASDAQ"):
    """
    Получить список тикеров для заданного рынка.

    Для NASDAQ загружаются топ-200 акций по капитализации из внешнего CSV-файла.
    Для MOEX возвращаются все доступные акции через библиотеку moexalgo.

    Args:
        market_name (str): Название рынка. Допустимые значения: "NASDAQ" или "MOEX".
                           По умолчанию "NASDAQ".

    Returns:
        list: Список строк – тикеров акций.
    """

    url = "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/all.csv"
    try:
        if market_name == "NASDAQ":
            data = pd.read_csv(url)
            tickers = data['symbol'].tolist()
            for i, t in enumerate(tickers):
                if '/' in t:
                    tickers[i] = t.replace('/', '-')

            data_logger.info(f"Loaded {len(tickers)} tickers from NASDAQ.")

        elif market_name == "MOEX":
            stocks = Market("stocks")
            tickers = stocks.tickers()['ticker']

            data_logger.info(f"Loaded {len(tickers)} tickers from MOEX.")
        else:
            raise ValueError(f"Can not get tickers from {market_name} market. Support only NASDAQ, MOEX.")
        return tickers
    except Exception as e:
        data_logger.exception(f"Exception while load tickers list: {e}")
        return []

def get_data_by_tickers(
        tickers: list, 
        start:str, 
        end:str, 
        market_name:str="NASDAQ",
        chunk_size:int = 50,
        sleep_time: float = 0.01) -> pd.DataFrame: 
    """
    Загрузить дневные цены закрытия для заданных тикеров за указанный период.

    Для NASDAQ используется yfinance с разбивкой на чанки для избежания перегрузки API.
    Для MOEX используются данные через moexalgo (по одному тикеру).

    Args:
        tickers (list): Список строк – тикеры акций.
        start (str): Дата начала периода в формате 'YYYY-MM-DD'.
        end (str): Дата окончания периода в формате 'YYYY-MM-DD'.
        market_name (str): Рынок: "NASDAQ" или "MOEX". По умолчанию "NASDAQ".
        chunk_size (int): Количество тикеров в одном запросе для yfinance.
                           По умолчанию 50.
        sleep_time (float): Пауза (в секундах) между запросами для yfinance.
                            По умолчанию 0.01.

    Returns:
        pd.DataFrame: DataFrame с ценами закрытия.
                      Индексы – даты, столбцы – тикеры.
    """
    if tickers is None:
        data_logger.error("Can not get any data: tickers list is empty")
        return pd.DataFrame()
    
    try:
        N = len(tickers)
        if market_name == "NASDAQ":
            all_data = []
            for i in range(0, N, chunk_size):
                try:
                    chunk_data = yf.download(tickers=tickers[i: i+chunk_size], start=start, end=end, progress=False, group_by='ticker')
                    time.sleep(sleep_time)

                    if chunk_data.empty:
                        data_logger.warning(f'Chunk {i // chunk_size} is empty.')
                        continue

                    if isinstance(chunk_data, pd.Series):
                        chunk_data = chunk_data.to_frame()
                    
                    chunk_close = chunk_data.xs("Close", axis=1, level=1)
                    all_data.append(chunk_close)         

                except Exception as e:
                    data_logger.error(f'Error while downloading chunk {i // chunk_size}: {e}')

            if not all_data:
                data_logger.error("No data downloaded from NASDAQ")
                return pd.DataFrame()
                
            data = pd.concat(all_data, axis=1)
            data = data.dropna(axis=1, how="all")
            data_logger.info(f'Successifully downloaded {data.shape[1]} stocks out of {N}')
            return data

        elif market_name == "MOEX":
            try:
                data = pd.DataFrame()
                for t in tickers:
                    try:
                        ticker =  Ticker(t)
                        df = ticker.candles(
                        start=start,
                        end=end,
                        period='1D'
                        )
                        if df.empty:
                            data_logger.warning(f"No candles for {t}")
                            continue

                        data[t] = df.set_index('begin')['close']
                        time.sleep(0.1) 
                    except Exception as e:
                        data_logger.error(f'Error while downloading stock {t}.')
            except Exception as e:
                data_logger.error(f'Error while downloading stock {t}.')

            return data 
        else:
            raise ValueError(f"Unsupported market: {market_name}. Support only NASDAQ, MOEX.")

    
    except Exception as e:
        data_logger.exception(f'Exception while download data: {e}')
        return pd.DataFrame()

def get_log_returns(data: pd.DataFrame, prefix:str="") -> pd.DataFrame:
    """
    Рассчитать логарифмические доходности и сохранить результат в CSV-файл.

    Вычисляются логарифмические доходности как отношение логарифмов цен,
    затем удаляются бесконечные значения и столбцы, содержащие пропуски.
    Результат сохраняется в файл `data/log_data{_prefix}.csv`.

    Args:
        data (pd.DataFrame): DataFrame с ценами закрытия (индекс – даты).
        prefix (str): Дополнительный суффикс для имени файла (например, название рынка).
                      По умолчанию пустая строка (файл сохраняется как log_data.csv).

    Returns:
        pd.DataFrame: DataFrame с логарифмическими доходностями.
    """
    try:
        log_returns = np.log(data / data.shift(1))[1:]
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
        log_returns = log_returns.dropna(axis=1, how="any")
        data_logger.info(f'Data was successifully prepared')
    except Exception as e:
        data_logger.error(f"Exception while prepare data: {e}")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.dirname(current_dir)
        log_returns.to_csv(path+'/data/log_data' + bool(prefix)*f'_{prefix}' + '.csv')
        data_logger.info(f'Data was successifully saved to ./data')
    except Exception as e:
        data_logger.error(f"Exception while save data: {e}")

    return log_returns

