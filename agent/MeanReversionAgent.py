from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np


class MeanReversionAgent(TradingAgent):
    """
    Simple Trading Agent that compares the 20 past mid-price observations with the 50 past observations and places a
    buy limit order if the 20 mid-price average >= 50 mid-price average or a
    sell limit order if the 20 mid-price average < 50 mid-price average
    """

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000,
                 min_size=50, max_size=100, lambda_a=0.05,
                 log_orders=False, random_state=None, short_duration=20,
                 long_duration=40, margin=0):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        # received information
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.short_duration = short_duration
        self.long_duration = long_duration
        self.margin = margin
        self.lambda_a = lambda_a

        self.size = self.random_state.randint(self.min_size, self.max_size)
        self.mid_list, self.avg_20_list, self.avg_50_list = [], [], []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        """ Mean reversion agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            if bid and ask:
                self.mid_list.append((bid + ask) / 2)
                if len(self.mid_list) > 20: self.avg_20_list.append(MeanReversionAgent.ma(self.mid_list, n=self.short_duration)[-1].round(2))
                if len(self.mid_list) > 50: self.avg_50_list.append(MeanReversionAgent.ma(self.mid_list, n=self.long_duration)[-1].round(2))
                if len(self.avg_20_list) > 0 and len(self.avg_50_list) > 0:
                    # 20200928 Chris Cho: Added the margin function
                    if self.avg_20_list[-1] <= self.avg_50_list[-1] + self.margin:
                        self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=True, limit_price=ask)
                    elif self.avg_20_list[-1] > self.avg_50_list[-1] - self.margin:
                        self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=False, limit_price=bid)
            #set wakeup time
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def getWakeFrequency(self):
        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        return pd.Timedelta('{}ns'.format(int(round(delta_time))))


    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n