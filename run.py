from src.build_mis import MIS_Builder
from src.optimization import PortfolioOptimizer
from src.analysis import PortfolioComparator, PortfolioAnalysis
from typing import Literal
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse


logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def run_process(
        market: Literal["NASDAQ", "MOEX"],
        smis_procedure:Literal["BH", "Holm"],
        mis_p:float=0.5, 
        smis_p:float=0.1,
        top_n:int=100,
        vizualization:bool=False
        ):
      
      risk_free = 0.11 if market == "NASDAQ" else 0.15 #Уточнить, какая fisk_free использовалась

      mis_builder = MIS_Builder(
            log_ret_file_name=f"log_data_{market}.csv",
            market_name=market,
            type_graph="t",
            param=mis_p,
            risk_free=risk_free,
            top_n=top_n
        ).run()
      smis_builder = MIS_Builder(
            log_ret_file_name=f"log_data_{market}.csv",
            market_name=market,
            type_graph="s",
            param=smis_p,
            procedure=smis_procedure,
            risk_free=risk_free,
            top_n=top_n
        ).run()
      
      
      mis_w = PortfolioOptimizer(path_to_file=f"data/mis_result/{market}_MIS_TG_{mis_p}.csv", risk_free=risk_free).run()
      smis_w = PortfolioOptimizer(path_to_file=f"data/mis_result/{market}_MIS_SG_{smis_p}.csv", risk_free=risk_free).run()


      comparator = PortfolioComparator(
            mis_file=f"data/mis_result/{market}_MIS_TG_{mis_p}.csv",
            smis_file=f"data/mis_result/{market}_MIS_SG_{smis_p}.csv",
            log_returns_file=f"data/log_data_{market}.csv",
            risk_free=risk_free
        )

        #comparator.plot_efficient_frontier(mis_w, smis_w)
      compare_df,  boot_p_value, tt_p_value = comparator.compare_portfolios()
      mis_an = PortfolioAnalysis(
            mis_file=f"data/mis_result/{market}_MIS_TG_{mis_p}.csv",
            log_returns_file=f"data/log_data_{market}.csv",
            mis_p=mis_p,
            risk_free=risk_free
        )
      smis_an = PortfolioAnalysis(
            mis_file=f"data/mis_result/{market}_MIS_SG_{smis_p}.csv",
            log_returns_file=f"data/log_data_{market}.csv",
            mis_p=smis_p,
            procedure=smis_procedure,
            risk_free=risk_free
        )

      enb = []
      hhi = []
      enb.append(mis_an.effective_number_of_bets())
      hhi.append(mis_an.hhi())
      enb.append(smis_an.effective_number_of_bets())
      hhi.append(smis_an.hhi())

      compare_df["ENB"]= enb
      compare_df["HHI"]= hhi



      logger.info(f"Compare df:\n{compare_df}")
      logger.info(f"bootstrap p-value = {boot_p_value}")
      logger.info(f"t-test p-value = {tt_p_value}")
      
      if vizualization:
        comparator.plot_cumulative_returns(mis_w, smis_w)
        comparator.plot_returns_distribution(mis_w, smis_w)
        comparator.plot_efficient_frontier(mis_w, smis_w)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запуск эксперимента по построению MIS, SMIS и сравнению портфелей."
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["NASDAQ", "MOEX"],
        default="NASDAQ",
        help="Рынок: NASDAQ или MOEX (по умолчанию NASDAQ)"
    )
    parser.add_argument(
        "--smis_procedure",
        type=str,
        choices=["BH", "Holm"],
        default="BH",
        help="Процедура коррекции p-значений для S-графа: BH или Holm (по умолчанию BH)"
    )
    parser.add_argument(
        "--mis_p",
        type=float,
        default=0.5,
        help="Параметр для T-графа (порог корреляции, по умолчанию 0.5)"
    )
    parser.add_argument(
        "--smis_p",
        type=float,
        default=0.1,
        help="Параметр для S-графа (уровень значимости, по умолчанию 0.1)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="Число акций с наивысшим коэффициентом Шарпа (по умолчанию 100)"
    )

    parser.add_argument(
        "--viz",
        type=bool,
        default=False,
        help="Активировать визуализацию сравнения портфелей (по умолчанию False)"
    )

    args = parser.parse_args()

    # Вызов основной функции с переданными аргументами
    run_process(
        market=args.market,
        smis_procedure=args.smis_procedure,
        mis_p=args.mis_p,
        smis_p=args.smis_p,
        top_n=args.top_n,
        vizualization=args.viz
    )