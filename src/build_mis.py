import pandas as pd
import numpy as np
import logging
import networkx as nx
import os
from typing import Literal
from scipy.stats import pearsonr
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MIS_Builder:
    """
    Класс для построения максимального независимого множества (MIS)
    на основе графа корреляций акций.

    Поддерживаются два типа графов:
    - threshold graph (T-граф): рёбра добавляются, если коэффициент корреляции
        превышает заданный порог.
    - statistical graph (S-граф): рёбра добавляются по результатам
        множественного тестирования (процедуры Беньямини–Хохберга или Холма).

    После построения графа MIS находится как максимальная клика в дополнении графа.
    Результат сохраняется в CSV-файл.
    Args:
        log_ret_file_name (str): Имя CSV-файла с логарифмическими доходностями
            (файл должен находиться в папке data/).
        market_name (str): Название рынка (используется для формирования
            имени выходного файла).
        type_graph (Literal["t", "s"]): Тип графа:
            "t" – пороговый граф, "s" – статистический граф.
        param (float): Параметр, зависящий от типа графа:
            для T-графа – порог корреляции; для S-графа – уровень значимости.
            По умолчанию 0.5.
        procedure (Literal["BH", "Holm"]): Процедура коррекции p-значений
            для S-графа. Игнорируется для T-графа. По умолчанию "BH".
        risk_free (float): Безрисковая ставка (годовая) для расчёта
            коэффициента Шарпа. По умолчанию 0.05.
        top_n (int): Количество акций с наибольшим коэффициентом Шарпа,
            которые будут использованы для построения графа.
            По умолчанию 100.
    """
    def __init__(self, log_ret_file_name:str,
                 market_name:str, 
                 type_graph:Literal["t", "s"], 
                 param:float=0.5,
                 procedure:Literal["BH", "Holm"]="BH",
                 risk_free:float=0.05,
                 top_n:int=100):

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.dirname(current_dir) 
            self.log_returns = pd.read_csv(path+ f"/data/{log_ret_file_name}")
            self.log_returns = self.log_returns.drop([self.log_returns.columns[0]], axis=1).astype(np.float64)
            logger.info(f"Successifully load data from data/{log_ret_file_name}")
        except FileNotFoundError:
            logger.error(f"File {log_ret_file_name} is not exist.")

        self.type_graph = type_graph
        self.risk_free = risk_free
        self.top_n = top_n
        self.param = param
        self.market_name = market_name
        self.procedure = procedure

    def top_n_stocks(self)->pd.DataFrame:
        """
        Отбирает top_n акций с наибольшим коэффициентом Шарпа.

        Для каждой акции рассчитывается годовой коэффициент Шарпа на основе
        логарифмических доходностей. Акции с недостаточной историей (менее 252
        наблюдений) исключаются. Возвращается DataFrame с доходностями
        отобранных акций.

        Returns:
            pd.DataFrame: DataFrame с логарифмическими доходностями
                отобранных акций (индекс – даты, столбцы – тикеры).
                В случае ошибки возвращается пустой DataFrame.
        """
        try:
            if self.top_n < 0: 
                logger.error("Wrong value for n. Only positive integer values are allowed.")
                raise 
            sharpes = dict()
            for s in self.log_returns.columns:
                s_returns = self.log_returns[s].replace([-np.inf, np.inf], np.nan).dropna()
                if len(s_returns) < 252: continue
                sharpes[s] = MIS_Builder.calculate_sharpe_ratio(s_returns, self.risk_free)
            
            try:
                sorted_stocks = sorted(sharpes.items(), key=lambda item: item[1], reverse=True)
                topN_tickers = [t for t, sharpe in sorted_stocks][:self.top_n]
            except IndexError:
                logger.info(f'Failed to extract top {self.top_n} stocks. Instead, {len(self.log_returns.columns[1:])} was extracted.')
            topN_stocks = self.log_returns[topN_tickers]
            logger.info(f"Data Frame of {self.top_n} best by Sharp ratio stocks was successfully loaded.")
            return topN_stocks
        except Exception as e:
            logger.error(f'Exception while get top n stocks DataFrame: {e}')
            return pd.DataFrame()
        
    def _build_threshold_graph(self) -> nx.Graph:
        """
        Строит пороговый граф (T-граф) на основе корреляционной матрицы.

        Рёбра добавляются между парами акций, если абсолютное значение
        коэффициента корреляции превышает значение self.param.
        Петли удаляются.

        Returns:
            nx.Graph: Построенный граф. В случае ошибки возвращается пустой граф.
        """
        try:
            corr_matrix = self.topN_stocks.corr()
            if corr_matrix.empty:
                logger.error(
                    "Exception while getting correlation matrix: correlation matrix is empty.")
                return nx.Graph()

            G = nx.from_pandas_adjacency(corr_matrix, create_using=nx.Graph())
            diag_edges = [(v, u) for v, u in G.edges() if v == u]
            G.remove_edges_from(diag_edges)
            logger.info(f'Successifully create {G}.') 
            edges_to_remove = [(v, u) for v, u, w in G.edges(data='weight') if w <= self.param]
            G.remove_edges_from(edges_to_remove)
            logger.info(f'Successifully delete edges with weight <= {self.param}. New graph sizes: {G}.')
            return G
        except Exception as e:
            logger.error(f'Exception while build threshold graph: {e}')
            return nx.Graph()
        
    def _benjamini_hochberg_procedure(self) -> nx.Graph:
        """
        Строит статистический граф с использованием процедуры Беньямини–Хохберга.

        Для каждой пары акций вычисляется p-значение корреляции Пирсона.
        Затем применяется процедура FDR с уровнем self.param.
        Рёбра добавляются для пар, признанных значимыми.

        Returns:
            nx.Graph: Граф, рёбра которого соответствуют значимым корреляциям.
                В случае ошибки возвращается пустой граф.
        """
        try:
            tickers = self.topN_stocks.columns
            N = len(tickers)
            p_values = np.zeros((N, N))

            #Calcucate p-values
            try:
                for i in range(len(tickers)):
                    for j in range(i+1, len(tickers)):
                        _, p_val = pearsonr(self.topN_stocks[tickers[i]], self.topN_stocks[tickers[j]])
                        if np.isnan(p_val) or np.isinf(p_val):
                            p_val = 1.0
                        p_values[i][j] = p_val
            except Exception as e:
                logger.error(f"Exception while calculate p-values: {e}")

            pairs = []
            for i in range(N):
                for j in range(i+1, N):
                    pairs.append((p_values[i, j], i, j))

            pairs.sort(key=lambda x: x[0])

            matrix_indicies = [(i, j) for _, i, j in pairs]
            M = len(matrix_indicies)   
            p_vals = [pairs[i][0] for i in range(len(pairs))]
            G = nx.Graph()
            reject_to = -1
            for k in range(M-1, -1, -1):
                try:
                    i, j = matrix_indicies[k]
                    if pairs[k][0] <= ((k+1)/M) * self.param:
                        reject_to = k
                        break
                except Exception as e:
                    logger.error(f"Exception in main loop of BHP: {e}")
                    return nx.Graph()
            for k in range(reject_to+1):
                i, j = matrix_indicies[k]
                G.add_edge(tickers[i], tickers[j])
            logger.info("BHP sucessfully ended")
            return G
                
        except Exception as e:
            logger.error(f"Exception in Benjamini-Hochber Procedure: {e}")
            return nx.Graph()
    
    def _holm_procedure(self) -> nx.Graph:
        """
        Строит статистический граф с использованием процедуры Холма.

        Вычисляются p-значения для всех пар корреляций, затем применяется
        step-down процедура Холма с уровнем значимости self.param.
        Рёбра добавляются для пар, прошедших коррекцию.

        Returns:
            nx.Graph: Граф, рёбра которого соответствуют значимым корреляциям.
                В случае ошибки возвращается пустой граф.
        """
        try:
            tickers = self.topN_stocks.columns
            N = len(tickers)
            p_values = np.zeros((N, N))

            try:
                for i in range(len(tickers)):
                    for j in range(i+1, len(tickers)):
                        _, p_val = pearsonr(self.topN_stocks[tickers[i]], 
                                            self.topN_stocks[tickers[j]])
                        if np.isnan(p_val) or np.isinf(p_val):
                            p_val = 1.0
                        p_values[i][j] = p_val
            except Exception as e:
                logger.error(f"Exception while calculate p-values: {e}")

            
            pairs = []
            for i in range(N):
                for j in range(i+1, N):
                    pairs.append((p_values[i, j], i, j))
            pairs.sort(key=lambda x: x[0])

            M = len(pairs) 


            reject_until = -1
            for k in range(M):
                p_val = pairs[k][0]
                
                if p_val <= self.param / (M - k):
                    reject_until = k  
                else:
                    break            


            G = nx.Graph()
            for k in range(reject_until + 1):
                _, i, j = pairs[k]
                G.add_edge(tickers[i], tickers[j])

            logger.info(
                f"Holm procedure ended: {reject_until+1} rejections out of {M}"
            )
            return G

        except Exception as e:
            logger.error(f"Exception in Holm procedure: {e}")
            return nx.Graph()
    
    def mis_to_csv(self):
        """
        Сохраняет найденное максимальное независимое множество (MIS) в CSV-файл.

        Из self.topN_stocks выбираются только те акции, которые входят в self.mis.
        Файл сохраняется в директорию data/mis_result/ с именем вида:
            {market_name}_MIS_{type_graph}G_{param}.csv
        Если директория не существует, она создаётся.
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.dirname(current_dir) 
            result = self.topN_stocks[list(self.mis)]
            mis_name = f"{self.market_name}_MIS_{self.type_graph.upper()}G_{str(self.param)}.csv"
            os.makedirs("/data/mis_result/", exist_ok=True)
            result.to_csv(path+f"/data/mis_result/{mis_name}")
            logger.info(f"MIS size of {len(self.mis)} successfully saved to {path+f"/data/result_mis/{mis_name}"}.")
        except Exception as e:
            logger.error(f"Error while save MIS: {e}")

    def build_mis(self) -> set:
        """
        Находит максимальное независимое множество (MIS) в текущем графе.

        Вычисляется дополнение графа self.G, затем находится максимальная клика
        в дополнении (что эквивалентно MIS в исходном графе).

        Returns:
            set: Множество тикеров, образующих MIS. В случае ошибки возвращается
                пустое множество.
        """
        try:
            if len(self.G.nodes) == 0:
                raise ValueError('Graph is empty.')

            G_complement = nx.complement(self.G)
            max_clique = max(nx.find_cliques(G_complement), key=len) # find_cliques использует детерминированную версию BK
            logger.info(f"Successifully created MIS size of {len(max_clique)}")
            print(max_clique)
            return set(max_clique)
        except Exception as e:
            logger.error(f"Exception while build MIS: {e}")
            return set()  
          
    def run(self):
        """
        Выполняет полный процесс построения MIS.

        Шаги:
            1. Отбор top_n акций по коэффициенту Шарфа.
            2. Построение графа на основе выбранного типа и параметров.
            3. Поиск MIS.
            4. Сохранение результата в CSV.

        Returns:
            set: Множество тикеров, составляющих MIS.
        """
        self.topN_stocks = self.top_n_stocks()
        if self.type_graph == "t":
            self.G = self._build_threshold_graph()
        elif self.type_graph == "s":
            if self.procedure == "BH":
                self.G =  self._benjamini_hochberg_procedure()
            elif self.procedure == "Holm":
                self.G = self._holm_procedure()
        self.mis = self.build_mis()
        self.mis_to_csv()
        return self.mis
        

    @staticmethod
    def calculate_sharpe_ratio(log_returns: pd.Series | np.ndarray, risk_free:float=0.05) -> float:
        annual_return = log_returns.mean() * 252
        annual_risk = log_returns.std() * np.sqrt(252)

        if annual_risk == 0: return 0

        return (annual_return - risk_free) / annual_risk
    


    



    
