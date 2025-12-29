import os
from config import Config, parse_args, get_basic_name
from trainer import train
import time


def get_name(test_time, valid_period, market_name): #ALL2023q1-validperiod1-nn-facx20240519--label1--dropout0.2
    return '-'.join(filter(None, [
        f'{market_name}{test_time}',
        f'{valid_period}',
        f'{get_basic_name()}',
    ])).replace(' ', '')


if __name__ == "__main__":
    start_time = time.time()
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    config = Config()
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    season_list = ["2023q1", "2023q2", "2023q3", "2023q4", "2024q1", "2024q2"]
    market = "ALL"

    for season in season_list:
        valid_period = rf"validperiod{args.valid_period_idx}"
        print(f"[INFO] Start training: season={season}, valid_period={valid_period}, model_name={get_name(season, valid_period, market)})")
        train(config, args, get_name(season, valid_period, market), market, season)
    end_time = time.time()
    print(f"Script ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")