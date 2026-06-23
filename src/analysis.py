import numpy as np
import pandas as pd
import logging
import os
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from typing import Literal, Optional
from src.optimization import PortfolioOptimizer
from src.build_mis import MIS_Builder
plt.style.use('seaborn-v0_8-darkgrid')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_portfolio_metrics(log_returns: pd.DataFrame, risk_free: float, weights):
    """
    Рассчитывает основные метрики портфеля на основе логарифмических доходностей.

    Args:
        log_returns (pd.DataFrame): DataFrame с логарифмическими доходностями
            активов (столбцы – активы, строки – даты).
        risk_free (float): Годовая безрисковая ставка.
        weights: Объект, содержащий веса портфеля (должен иметь атрибут columns
            с названиями активов и поддерживать транспонирование).

    Returns:
        tuple: (port_ret, annual_ret, annual_risk, sharpe_ratio), где
            port_ret (pd.Series) – дневные доходности портфеля,
            annual_ret (float) – годовая ожидаемая доходность,
            annual_risk (float) – годовая волатильность,
            sharpe_ratio (float) – коэффициент Шарпа.
    """
    returns = log_returns[weights.columns]
    cov = returns.cov().values
    port_ret = returns @ weights.T
    annual_ret = port_ret.mean() * 252
    annual_risk = port_ret.std() * np.sqrt(252)
    sharpe_ratio = (annual_ret - risk_free)/annual_risk
    return port_ret, annual_ret, annual_risk, sharpe_ratio

def sharpe(port_ret, risk_free):
    annual_ret = port_ret.mean() * 252
    annual_risk = port_ret.std() * np.sqrt(252)
    sharpe_ratio = (annual_ret - risk_free)/annual_risk
    return np.float64(sharpe_ratio)
     

class PortfolioComparator:
    """
    Класс для сравнения двух портфелей (MIS и SMIS) по эффективности.

    Загружает доходности активов, оптимизирует портфели на основе сохранённых
    наборов активов (из CSV-файлов), сравнивает их по Шарпу, доходности и риску,
    а также предоставляет методы визуализации (кумулятивная доходность,
    эффективная граница, распределение доходностей и др.).

    Args:
        mis_file (str): Путь к CSV-файлу с активами, отобранными для MIS-портфеля.

        smis_file (str): Путь к CSV-файлу с активами, отобранными для SMIS-портфеля.

        log_returns_file (str): Путь к CSV-файлу с логарифмическими доходностями
        всех активов (используется для расчёта метрик).

        risk_free (float): Годовая безрисковая ставка. По умолчанию 0.15.

    """
    def __init__(self, 
                 mis_file:str, 
                 smis_file:str, 
                 log_returns_file:str, 
                 risk_free:float=0.15):
            
        self.mis_file = mis_file
        self.smis_file = smis_file
        self.risk_free = risk_free
        try:
            self.log_returns = pd.read_csv(log_returns_file)
        except Exception as e:
            logger.error(f"File {log_returns_file} is not exist.")
  

    def compare_portfolios(self):
        """
        Сравнивает портфели MIS и SMIS по годовым метрикам и проводит статистические тесты.

        Для каждого набора активов оптимизируется портфель минимальной дисперсии,
        затем рассчитываются годовая доходность, риск и коэффициент Шарпа.
        Дополнительно вычисляются p-значения для разности Шарпа (бутстрап)
        и для попарного t-теста доходностей.

        Returns:
            tuple: (compare_df, boot_p_value, tt_p_value), где
                compare_df (pd.DataFrame): DataFrame с метриками для каждого портфеля
                    (строки: "mis", "smis"; столбцы: "ann_ret", "ann_risk", "sharpe").
                boot_p_value (float): p-значение бутстрап-теста для разности Шарпа.
                tt_p_value (float): p-значение парного t-теста для доходностей.
        """


        mis_weights = PortfolioOptimizer(path_to_file=self.mis_file, 
                                     risk_free=self.risk_free).run()
        mis_ret, *mis_metrics = get_portfolio_metrics(self.log_returns, self.risk_free, weights=mis_weights)

        smis_weights = PortfolioOptimizer(path_to_file=self.smis_file, 
                                risk_free=self.risk_free).run()
        smis_ret, *smis_metrics = get_portfolio_metrics(self.log_returns, self.risk_free, weights=smis_weights)
        
        compare_df = pd.DataFrame(index=["mis", "smis"], columns=["ann_ret", "ann_risk", "sharpe"])
        compare_df.loc["smis"] = np.array(smis_metrics).squeeze(axis=1)
        compare_df.loc["mis"] = np.array(mis_metrics).squeeze(axis=1)

        sharpe_diff, boot_p_value = self._bootstrap_sharpe_diff(mis_ret, smis_ret)
        t_stat, tt_p_value = ttest_rel(mis_ret, smis_ret)
        return compare_df, boot_p_value, tt_p_value
    
    def _bootstrap_sharpe_diff(self, ret1, ret2, n_bootstrap=10000):
        """
        Оценивает значимость разности коэффициентов Шарпа с помощью бутстрапа.

        Args:
            ret1 (array-like): Доходности первого портфеля.
            ret2 (array-like): Доходности второго портфеля.
            n_bootstrap (int): Количество бутстрап-выборок. По умолчанию 10000.

        Returns:
            tuple: (sharpe_diff, p_value), где
                sharpe_diff (float) – наблюдаемая разность Шарпа,
                p_value (float) – эмпирическое p-значение.
        """
        rng = np.random.default_rng(seed=42)
        sharpe_diff = np.float64(sharpe(ret1, self.risk_free) - sharpe(ret2, self.risk_free))
        combined = np.column_stack([ret1, ret2])
        n = len(combined)
        boot_diffs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            sample = combined[idx]
            s1 = sharpe(sample[:, 0], self.risk_free)
            s2 = sharpe(sample[:, 1], self.risk_free)
            boot_diffs.append(s1 - s2)
        boot_diffs = np.array(boot_diffs)
        p_value = np.mean(np.abs(boot_diffs) >= np.abs(sharpe_diff))
        return sharpe_diff, p_value

    def plot_cumulative_returns(self, mis_weights, smis_weights, save_path: Optional[str] = None):
        """
        Строит график кумулятивной доходности портфелей (начальный капитал = 1).

        Args:
            mis_weights: Веса активов для MIS-портфеля (объект с атрибутом columns).
            smis_weights: Веса активов для SMIS-портфеля.
            save_path (Optional[str]): Путь для сохранения графика (если указан).
        """
        ret_mis = self.log_returns[mis_weights.columns] @ mis_weights.T
        ret_smis = self.log_returns[smis_weights.columns] @ smis_weights.T
        
        cum_mis = (1 + ret_mis).cumprod()
        cum_smis = (1 + ret_smis).cumprod()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cum_mis.index, cum_mis, label='MIS портфель', color='blue')
        plt.plot(cum_smis.index, cum_smis, label='SMIS портфель', color='red')
        plt.title('Кумулятивная доходность портфелей')
        plt.xlabel('Дата')
        plt.ylabel('Накопленный капитал (нач. = 1)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_efficient_frontier(self, mis_weights, smis_weights, 
                                all_returns: Optional[pd.DataFrame] = None,
                                n_points: int = 50, save_path: Optional[str] = None):
        """
        Строит эффективные границы Марковица для портфелей MIS, SMIS и всех активов.

        Args:
            mis_weights: Веса MIS-портфеля.
            smis_weights: Веса SMIS-портфеля.
            all_returns (Optional[pd.DataFrame]): Доходности всех доступных активов.
                Если не указаны, используются self.log_returns.
            n_points (int): Количество точек для построения границы. По умолчанию 50.
            save_path (Optional[str]): Путь для сохранения графика.
        """
        if all_returns is None:
            all_returns = self.log_returns  
        
        frontiers = {}
        labels = {}
        
        ret_mis = self.log_returns[mis_weights.columns]
        frontiers['MIS'] = self._compute_efficient_frontier(ret_mis, n_points)
        labels['MIS'] = f'MIS (Sharpe = {sharpe(ret_mis @ mis_weights.T, self.risk_free)[0]:.3f})'
        
        ret_smis = self.log_returns[smis_weights.columns]
        frontiers['SMIS'] = self._compute_efficient_frontier(ret_smis, n_points)
        labels['SMIS'] = f'SMIS (Sharpe = {sharpe(ret_smis @ smis_weights.T, self.risk_free)[0]:.3f})'
        
        plt.figure(figsize=(10, 6))
        colors = {'MIS': 'blue', 'SMIS': 'red', 'All': 'gray'}
        markers = {'MIS': 'o', 'SMIS': 's', 'All': '^'}
        
        for key in frontiers:
            frontier = frontiers[key]
            plt.plot(frontier['risk'], frontier['return'], 
                     label=labels[key], color=colors[key], 
                     marker=markers[key], markersize=3, linestyle='-', alpha=0.7)
        
        # Добавить точки конкретных портфелей (равные веса, мин. риск, макс. Шарпа) если нужно
        plt.title('Эффективные границы портфелей')
        plt.xlabel('Годовой риск (волатильность)')
        plt.ylabel('Годовая ожидаемая доходность')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def _compute_efficient_frontier(self, returns: pd.DataFrame, n_points: int = 50, gamma:float=1e-6):
        """
        Вычисляет эффективную границу для заданного набора активов.

        Args:
            returns (pd.DataFrame): Доходности активов.
            n_points (int): Количество точек на границе. По умолчанию 50.
            gamma (float): Коэффициент регуляризации. По умолчанию 1e-6.

        Returns:
            dict: Словарь с ключами 'risk' и 'return', содержащими массивы
                соответствующих значений (только для успешно решённых задач).
        """
        mu = np.array(returns.mean())
        cov = returns.cov() 
        n = len(mu)
        
        # Диапазон доходностей
        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        risks = []
        
        for target in target_returns:
            weights = cp.Variable(n)
            reg_cov = cov + 1e-6 * np.eye(len(mu))
            objective = cp.Minimize(cp.quad_form(weights, reg_cov) + gamma * cp.sum_squares(weights))
            constraints = [cp.sum(weights) == 1, weights >= 0, mu @ weights == target]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver='OSQP', eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)  
            if prob.status == 'optimal':
                risk = np.sqrt(weights.value @ cov @ weights.value)
                risks.append(risk)
            else:
                risks.append(np.nan)
        
        valid = ~np.isnan(risks)
        return {'risk': np.array(risks)[valid], 'return': target_returns[valid]}

    def plot_weights_distribution(self, weights_df: pd.DataFrame, title: str = "Веса активов в портфеле", save_path=None):
        """
        Визуализирует распределение весов активов в портфеле (горизонтальная
        столбчатая диаграмма, отсортированная по убыванию веса).

        Args:
            weights_df (pd.DataFrame): DataFrame с весами (одна строка).
            title (str): Заголовок графика. По умолчанию "Веса активов в портфеле".
            save_path (Optional[str]): Путь для сохранения графика.
        """
        weights = weights_df.iloc[0]  
        weights_sorted = weights.sort_values(ascending=False)
        
        plt.figure(figsize=(10, max(6, len(weights)//4)))
        sns.barplot(x=weights_sorted.values, y=weights_sorted.index, palette='viridis')
        plt.title(title)
        plt.xlabel('Вес в портфеле')
        plt.ylabel('Тикер')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_returns_distribution(self, mis_weights, smis_weights, save_path=None):
        """
        Сравнивает распределения дневных доходностей двух портфелей.

        Строит гистограммы с ядерной оценкой плотности и ящики с усами (boxplot)
        для визуального сравнения.

        Args:
            mis_weights: Веса MIS-портфеля.
            smis_weights: Веса SMIS-портфеля.
            save_path (Optional[str]): Путь для сохранения графика.
        """
        ret_mis = self.log_returns[mis_weights.columns] @ mis_weights.T
        ret_mis = np.array(ret_mis).squeeze(axis=1)
        ret_smis = self.log_returns[smis_weights.columns] @ smis_weights.T
        ret_smis = np.array(ret_smis).squeeze(axis=1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(ret_mis, bins=50, kde=True, color='blue', label='MIS')
        sns.histplot(ret_smis, bins=50, kde=True, color='red', label='SMIS')
        plt.title('Распределение дневных доходностей')
        plt.xlabel('Доходность')
        plt.ylabel('Частота')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=[ret_mis, ret_smis], palette=['blue', 'red'])
        plt.xticks([0, 1], ['MIS', 'SMIS'])
        plt.title('Сравнение разброса доходностей')
        plt.ylabel('Доходность')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
            

class PortfolioAnalysis:
        """
        Класс для анализа эффективности одного портфеля, построенного на основе MIS.

        Позволяет строить зависимости метрик от параметра (порога или уровня значимости),
        а также вычислять эффективное число ставок (ENB) и индекс Херфиндаля–Хиршмана
        (HHI) для портфеля.

        Args:
            mis_file (str): Путь к CSV-файлу с активами, отобранными для MIS.
            log_returns_file (str): Путь к CSV-файлу с логарифмическими доходностями.
            risk_free (float): Годовая безрисковая ставка. По умолчанию 0.15.
            top_n (int): Количество акций для отбора по Шарпу (используется,
                если param задан и требуется перестроить MIS). По умолчанию 100.
            procedure (Literal["BH", "Holm"]): Процедура коррекции для S-графа.
                По умолчанию "BH".
            param (Optional[float]): Параметр (порог для T-графа или уровень
                значимости для S-графа). Если указан, MIS перестраивается заново.
                В противном случае используется существующий файл.
        """
        def __init__(self, 
                 mis_file:str, 
                 log_returns_file:str, 
                 risk_free:float=0.15, 
                 top_n:float=100,
                 procedure:Literal["BH", "Holm"]="BH",
                 param:Optional[float]=None):
            
            self.mis_type = "s" if "SG" in mis_file else "t"
            self.mis_file = mis_file
            self.risk_free = risk_free
            self.market = "MOEX" if "MOEX" in mis_file else "NASDAQ"
            self.param = param
            self.top_n = top_n
            self.procedure = procedure

            if self.param:
                mis_builder = MIS_Builder(
                    log_ret_file_name=f"log_data_{self.market}.csv",
                    procedure=self.procedure,
                    market_name=self.market,
                    type_graph=self.mis_type,
                    param=self.param, 
                    top_n=self.top_n)
                mis_builder.run()
                mis_opt = PortfolioOptimizer(f"data/mis_result/{self.market}_MIS_{self.mis_type.upper()}G_{self.param}.csv", risk_free=self.risk_free)
                self.mis_weights = mis_opt.run()

            try:
                self.log_returns = pd.read_csv(log_returns_file)
            except Exception as e:
                logger.error(f"File {log_returns_file} is not exist.")
            
        def plot_sharpe_on(self, start_val:float, end_val:float, step: float):
            """
            Строит графики зависимости средней доходности и коэффициента Шарпа
            от значения параметра (Threshold для T-графа или Alpha для S-графа).

            Args:
                start_val (float): Начальное значение параметра.
                end_val (float): Конечное значение параметра.
                step (float): Шаг изменения параметра.
            """
            if start_val > end_val:
                logger.error("Start value should be lower than end value")
                return
            if step >= (end_val - start_val):
                logger.error("Step can not be bigger than (end_val - start_val)")
                return

            rets = dict()
            sharps = dict()
            for param in np.arange(start_val, end_val + step/2, step):
                mis_builder = MIS_Builder(log_ret_file_name=f"log_data_{self.market}.csv",
                                        market_name=self.market,
                                        procedure=self.procedure,
                                        type_graph=self.mis_type,
                                        param=param, 
                                        top_n=self.top_n)
                mis_size = len(mis_builder.run())
                    
                mis_opt = PortfolioOptimizer(f"data/mis_result/{self.market}_MIS_{self.mis_type.upper()}G_{param}.csv", risk_free=self.risk_free)
                mis_weights = mis_opt.run()

                ret, ann_ret, ann_risk, sharpe = get_portfolio_metrics(
                    log_returns=self.log_returns, 
                    risk_free=self.risk_free, 
                    weights=mis_weights
                )

                if ann_ret[0] > rets.get(param, -np.inf):
                    rets[param] = ann_ret[0]

                if sharpe[0] > sharps.get(param, -np.inf):
                    sharps[param] = ann_risk[0]

            fig, ax = plt.subplots(1, 2)
            sorted_lst_ret = sorted(rets.items(), key=lambda items: items[0])
            x, y = list(zip(*sorted_lst_ret))
            param_name = "Alpha" if self.mis_type=="s" else "Threshold"

            ax[0].plot(x, y)
            ax[0].set_xlabel(param_name)
            ax[0].set_ylabel("Средняя доходность")
            ax[0].set_title(f"Зависимость средней доходности от {param_name}. Рынок {self.market}")

            sorted_lst_sharpe = sorted(sharps.items(), key=lambda items: items[0])
            x, y = list(zip(*sorted_lst_sharpe))
            ax[1].plot(x, y, c='r')
            ax[1].set_xlabel(param_name)
            ax[1].set_ylabel("Коэффициент Шарпа")
            ax[1].set_title(f"Зависимость к. Шарпа от {param_name}. Рынок {self.market}")
            plt.show()
    
        def effective_number_of_bets(self) -> float:
            """
            Вычисляет эффективное число ставок (ENB) портфеля на основе
            распределения вкладов в риск.

            ENB определяется как экспонента энтропии долей рисковых вкладов.

            Returns:
                float: Эффективное число ставок (≥ 1). Если портфель пуст или
                    все вклады неположительны, возвращается 0.0.
            """
            mis_tickers = self.mis_weights.columns
            mis_weights =  np.array(self.mis_weights.T).squeeze(axis=1)
            returns_df = self.log_returns[mis_tickers]
            cov_matrix = returns_df.cov()
            portf_var = mis_weights @ cov_matrix @ mis_weights.T
            marginal_contrib = cov_matrix @ mis_weights
            risk_contrib = mis_weights * marginal_contrib
            p = risk_contrib / portf_var 
            p = p[p > 0] 
            if len(p) == 0:
                return 0.0
            H = -np.sum(p * np.log(p))
            return np.exp(H)
        
        def hhi(self):
            """
            Вычисляет индекс Херфиндаля–Хиршмана (HHI) для весов портфеля.

            HHI = сумма квадратов весов. Используется как мера концентрации.

            Returns:
                float: Индекс HHI (в диапазоне от 0 до 1).
            """
            mis_tickers = self.mis_weights.columns
            mis_weights =  np.array(self.mis_weights.T).squeeze(axis=1)
            return np.sum(mis_weights ** 2)