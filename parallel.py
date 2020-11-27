import argparse
import os
from multiprocessing import Pool
import psutil
import datetime as dt
import numpy as np
from dateutil.parser import parse
import pyDOE

def run_in_parallel(num_simulations, num_parallel, config, log_folder, verbose, book_freq, hist_date, mkt_start_time, mkt_end_time):

    global_seeds = np.random.randint(0, 2 ** 31, num_simulations)
    print(f'Global Seeds: {global_seeds}')
    lhc_unit = np.array(pyDOE.lhs(3, num_simulations, "m"))
    lhc = np.round(lhc_unit*100)

    processes = []

    for i in range(num_simulations):

        seed        = global_seeds[i]

        zip_count   = int(lhc[i][0])
        mmt_count   = int(lhc[i][1])
        mr_count    = int(lhc[i][2])
        mm_count    = 1

        zi_count    = int(1000 - zip_count - mmt_count - mr_count - mm_count)

        print(f"current config: {zi_count}, {zip_count}, {mmt_count}, {mr_count}")


        processes.append(f'python -u abides.py -c {config} -l {log_folder}_config_{zi_count}_{zip_count}_{mmt_count}_{mr_count}_{mm_count} \
                        {"-v" if verbose else ""} -s {seed} -b {book_freq} -d {hist_date} -st {mkt_start_time} -et {mkt_end_time} \
                        -zi {zi_count} -zip {zip_count} -mmt {mmt_count} -mr {mr_count} -mm {mm_count}')

    print(processes)  
    pool = Pool(processes=num_parallel)
    pool.map(run_process, processes)


def run_process(process):
    os.system(process)


if __name__ == "__main__":
    start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description='Main config to run multiple ABIDES simulations in parallel')
    parser.add_argument('-c', '--config', 
                        required=True,
                        help='Name of config file to execute'
                        )
    parser.add_argument('-ns', '--num_simulations', 
                        type=int,
                        default=1,
                        help='Total number of simulations to run')
    parser.add_argument('-np', '--num_parallel', 
                        type=int,
                        default=None,
                        help='Number of simulations to run in parallel')
    parser.add_argument('-l', '--log_folder',
                        required=True,
                        help='Log directory name')
    parser.add_argument('-b', '--book_freq', 
                        default=None,
                        help='Frequency at which to archive order book for visualization'
                        )
    parser.add_argument('-n', '--obs_noise', 
                        type=float, 
                        default=1000000,
                        help='Observation noise variance for zero intelligence agents (sigma^2_n)'
                        )
    parser.add_argument('-o', '--log_orders',
                        action='store_true',
                        help='Log every order-related action by every agent.'
                        )
    parser.add_argument('-s', '--seed', 
                        type=int, 
                        default=None,
                        help='numpy.random.seed() for simulation'
                        )
    parser.add_argument('-v', '--verbose', 
                        action='store_true',
                        help='Maximum verbosity!'
                        )
    parser.add_argument('--config_help', 
                        action='store_true',
                        help='Print argument options for this config file'
                        )
    parser.add_argument('-d', '--historical_date',
                        required=True,
                        help='historical date being simulated in format YYYYMMDD.'
                        )
    parser.add_argument('-st', '--start_time',
                        default='09:30:00',
                        help='Starting time of simulation.'
                        )
    parser.add_argument('-et', '--end_time',
                        default='16:00:00',
                        help='Ending time of simulation.'
                        )

    args, remaining_args = parser.parse_known_args()

    seed            = args.seed
    num_simulations = args.num_simulations
    num_parallel    = args.num_parallel if args.num_parallel else psutil.cpu_count() # count of the CPUs on the machine
    config          = args.config
    log_folder      = args.log_folder
    verbose         = args.verbose
    book_freq       = args.book_freq
    hist_date       = args.historical_date
    mkt_start_time  = args.start_time
    mkt_end_time    = args.end_time


    print(f'Total number of simulation: {num_simulations}')
    print(f'Number of simulations to run in parallel: {num_parallel}')
    print(f'Configuration: {config}')

    np.random.seed(seed)

    run_in_parallel(num_simulations = num_simulations,
                    num_parallel    = num_parallel,
                    config          = config,
                    log_folder      = log_folder,
                    verbose         = verbose,
                    book_freq       = book_freq,
                    hist_date       = hist_date,
                    mkt_start_time  = mkt_start_time,
                    mkt_end_time    = mkt_end_time)

    end_time = dt.datetime.now()
    print(f'Total time taken to run in parallel: {end_time - start_time}')
