import numpy as np
import pandas as pd
import cvxpy as cp
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from math import isclose
from typing import Literal
from scipy.optimize import minimize
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Класс для оптимизации портфеля методом минимальной дисперсии.

    Загружает логарифмические доходности из CSV-файла, вычисляет ковариационную
    матрицу и средние доходности, затем находит веса портфеля с минимальной
    дисперсией при ограничении на сумму весов (и, опционально, запрете коротких
    позиций). Поддерживается фильтрация активов по Парето-фронту (по соотношению
    доходность/риск).
    Args:
            path_to_file (str): Путь к CSV-файлу с логарифмическими доходностями
                (столбцы – активы, строки – даты).
            risk_free (float): Годовая безрисковая ставка (используется для
                расчёта коэффициента Шарпа при фильтрации). По умолчанию 0.15.
            allow_shorts (bool): Разрешать ли короткие позиции (отрицательные веса).
                По умолчанию False.
    """

    def __init__(self, path_to_file:str, risk_free:float=0.15, allow_shorts:bool=False):
        try:
            self.log_returns = pd.read_csv(path_to_file)
        except FileNotFoundError:
            logger.error(f"File {path_to_file} is not exist.")
        self.allow_shorts = allow_shorts
        self.risk_free = risk_free

    def get_pareto_tickers(self):
        """
        Возвращает список тикеров, находящихся на Парето-фронте (риск-доходность).

        Для каждого актива вычисляется средняя доходность и стандартное отклонение.
        Активы сортируются по возрастанию риска, затем отбираются те, у которых
        доходность строго больше максимальной доходности среди ранее отобранных.
        Это даёт множество активов, не доминируемых по обоим критериям.

        Returns:
            list: Список строк – тикеров активов, попавших на Парето-фронт.
        """
        agg_result = self.log_returns.agg(["mean", "std"]).transpose().reset_index()
        agg_result.columns = ['Ticker', 'Expectable_Mean', 'Std']
        sorted_df = agg_result.sort_values('Std').reset_index(drop=True)

        pareto_front = []
        max_return_so_far = -np.inf

        for i, row in sorted_df.iterrows():
            if row['Expectable_Mean'] > max_return_so_far:
                pareto_front.append(i)
                max_return_so_far = row['Expectable_Mean']

        pareto_df = sorted_df.loc[pareto_front]
        return pareto_df["Ticker"].tolist()
    
    def _min_variance_portfolio(self, gamma=1e-6):
        """
        Вычисляет веса портфеля с минимальной дисперсией.

        Решается задача квадратичного программирования с ограничением на сумму
        весов (равна 1) и, если self.allow_shorts == False, ограничением
        неотрицательности весов. Для регуляризации к ковариационной матрице
        добавляется малая диагональная матрица.

        Args:
            gamma (float): Коэффициент регуляризации L2 (штраф на квадраты весов).
                По умолчанию 1e-6.

        Returns:
            np.ndarray: Вектор весов портфеля (длины N).
        """
        N = len(self.means)
        weights = cp.Variable(N)
        reg_cov = self.cov_matrix + 1e-6 * np.eye(len(self.means))
        objective = cp.Minimize(cp.quad_form(weights, reg_cov) + gamma * cp.sum_squares(weights))
        constraints = [cp.sum(weights) == 1]
        if not self.allow_shorts:
            constraints.append(weights >= 0)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='OSQP', eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
        return weights.value
    
    
    def run(self, filter_pareto:bool=False):
        """
        Выполняет оптимизацию портфеля.

        Загружает данные, опционально фильтрует активы по Парето-фронту,
        удаляет столбец индекса (если присутствует), вычисляет необходимые
        статистики и находит портфель с минимальной дисперсией. Веса, близкие
        к нулю (по модулю менее 1e-4), удаляются из результата.

        Args:
            filter_pareto (bool): Если True, перед оптимизацией оставляются
                только активы, находящиеся на Парето-фронте. По умолчанию False.

        Returns:
            pd.DataFrame: DataFrame с одной строкой, содержащей веса активов
                в портфеле. Индекс – 0, столбцы – тикеры активов.
        """        
        if filter_pareto:
            pareto_tickers = self.get_pareto_tickers()
            self.log_returns = self.log_returns[pareto_tickers]
        self.log_returns = self.log_returns.drop(["Unnamed: 0"], axis=1)
        self.agg_data = self.log_returns.agg(["mean", "std"]).transpose().reset_index()
        self.agg_data.columns = ['Ticker', 'Expectable_Mean', 'Std']
        self.cov_matrix = self.log_returns.cov().values       
        self.means = self.agg_data["Expectable_Mean"].values
        self.targets = np.linspace(self.means.min(), self.means.max(), 100)
        port_w_df = pd.DataFrame(columns=self.log_returns.columns)
        port_w_df.loc[len(port_w_df)] = self._min_variance_portfolio()
        col_to_drop = [col for col in port_w_df.columns if isclose(port_w_df[col][0], 0.0, abs_tol=1e-4)]
        port_w_df = port_w_df.drop(col_to_drop, axis=1)
        return port_w_df