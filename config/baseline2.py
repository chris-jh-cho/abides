# Baseline 2
# - 1     Exchange Agent
# - 1     Market Maker Agent
# - 100   ZIP Agent
# - 25    Momentum Agents
# - 25    Mean-Reversion Agents


import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.ZIP import ZeroIntelligencePlus
from agent.market_makers.SpreadBasedMarketMakerAgent import SpreadBasedMarketMakerAgent
from agent.MomentumAgent import MomentumAgent
from agent.MeanReversionAgent import MeanReversionAgent
from model.LatencyModel import LatencyModel

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for RMSC03 config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='10:30:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('-b', '--book_freq', 
                    default=None,
                    help='Frequency at which to archive order book for visualization')
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    action='store_true',
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
# market maker config
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-4,
                    help='Volatility of fundamental time series.'
                    )

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 31 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = True
log_orders = False
book_freq = args.book_freq

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

r_bar = 1e5
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11

# Oracle
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'sigma_s': 0,
                    'fund_vol': args.fund_vol,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1, dtype='uint64'))}}

oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1


# 2) ZIP
num_value = 20
agents.extend([ZeroIntelligencePlus(id=j,
                                    name="ZIP {}".format(j),
                                    type="ZeroIntelligencePlus",
                                    symbol=symbol,
                                    starting_cash=starting_cash,
                                    lambda_a = 5e-11,
                                    log_orders=log_orders,
                                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])


# 3) Market Maker Agents

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

# each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)

num_mm_agents = 1

agents.extend([SpreadBasedMarketMakerAgent(id=j,
                                            name="Market_MakerAgent_{}".format(j),
                                            type='SpreadBasedMarketMakerAgent',
                                            symbol=symbol,
                                            starting_cash=starting_cash,
                                            lambda_a = 5e-11,
                                            log_orders=log_orders,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1,
                                                                                                    dtype='uint64')))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend('MarketMakerAgent')


# 4) Momentum Agents
num_momentum_agents = 25

agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             lambda_a = 5e-11,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")


# 5) Mean Reversion Agents
num_mean_reversion_agents = 25

agents.extend([MomentumAgent(id=j,
                             name="Mean_Reversion_{}".format(j),
                             type="MeanReversionAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             lambda_a = 5e-11,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_mean_reversion_agents)])
agent_count += num_mean_reversion_agents
agent_types.extend("ZIPAgent")


# 6) ZI
num_value = 500
agents.extend([ZeroIntelligencePlus(id=j,
                                    name="Zero Intelligence {}".format(j),
                                    type="ZeroIntelligenceAgent",
                                    symbol=symbol,
                                    starting_cash=starting_cash,
                                    lambda_a = 5e-11,
                                    log_orders=log_orders,
                                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ZeroIntelligenceAgent'])



########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("Baseline 2 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 31 - 1,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1))
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args
                             )
# KERNEL

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=args.log_dir)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
