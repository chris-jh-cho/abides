from agent.TradingAgent import TradingAgent
from util.util import log_print

from math import sqrt
import numpy as np
import pandas as pd
import copy


class ZeroIntelligencePlus(TradingAgent):

    # setup
    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000, sigma_n=1000,
                 q_max=10, R_min=0, R_max=250, eta=1.0,
                 lambda_a=0.005, log_orders=False, random_state=None):

        # Base class init.
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        # determine whether the agent is a "buy" agent or a "sell" agent
        self.buy = bool(self.random_state.randint(0, 2))
        log_print("Coin flip: this agent is a {} agent", "BUY" if self.buy else "SELL")

        # Store important parameters particular to the ZI agent
        self.symbol = symbol  # symbol to trade
        self.sigma_n = sigma_n  # observation noise variance
        self.q_max = q_max  # max unit holdings
        self.R_min = R_min  # min requested surplus
        self.R_max = R_max  # max requested surplus
        self.eta = eta  # strategic threshold
        self.lambda_a = lambda_a  # mean arrival rate of ZI agents - this is the exp. distribution parameter to be tuned
        self.order_size = 100 #order size is fixed at 100 to start - the order size needs to be tuned
        # std of 500 should be plenty
        self.limit_price = 100000 + np.random.uniform(-5000, 5000)

        # we are not querying from an oracle right now
        #self.limit_price = self.oracle.observePrice(self.symbol, startTime, sigma_n=self.sigma_n,random_state=self.random_state)


        # ZIP update parameters
        self.target_price           = 0 #target price just needs to exist
        self.momentum_target_price  = 0 #initial condition
        self.hoff_delta             = 0 #hoff delta needs to exist
        self.learning_rate          = 0.1 + 0.4*np.random.rand() #uniform [0.1,0.5] fixed per agent
        self.momentum               = 0.2 + 0.6*np.random.rand() #uniform [0.2,0.8] fixed per agent
        self.abs_change_up          = 10*np.random.rand() #uniform [0,0.05] fixed per agent - temporarily changed to 10 cents
        self.rel_change_up          = 1 + 0.05*np.random.rand() #uniform [1,1.05] fixed per agent
        self.abs_change_down        = -10*np.random.rand() #uniform [-0.05,0] fixed per agent - temporarily changed to 10 cents
        self.rel_change_down        = 1 - 0.05*np.random.rand() #uniform [0.95,1] fixed per agent
        self.profit_margin          = 0.3 + 0.05*np.random.rand() #uniform [0.3,0.35] fixed per agent


        # the agent uses this to determine which data to query from the orderbook
        self.data_dummy = 0

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state = 'AWAITING_WAKEUP'

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time = None

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        # self.exchangeID is set in TradingAgent.kernelStarting()

        super().kernelStarting(startTime)

        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        H = int(round(self.getHoldings(self.symbol), -2) / 100)

        # May request real fundamental value from oracle as part of final cleanup/stats.
        if self.symbol != 'ETF':
            #rT = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)
            # for the time being, rT is 100000 and does not move
            rT = 100000

        # this is only for a portfolio evaluation
        else:
            portfolio_rT, rT = self.oracle.observePortfolioPrice(self.symbol, self.portfolio, self.currentTime,
                                                                 sigma_n=0,
                                                                 random_state=self.random_state)

        # Add final (real) fundamental value times shares held.
        surplus = rT * H + self.holdings['CASH'] - self.starting_cash

        self.logEvent('FINAL_VALUATION', surplus, True)

        log_print(
            "{} final report.  Holdings {}, end cash {}, start cash {}, final fundamental {}, surplus {}",
            self.name, H, self.holdings['CASH'], self.starting_cash, rT, surplus)

    def wakeup(self, currentTime):
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                log_print("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        # Schedule a wakeup for the next time this agent should arrive at the market
        # (following the conclusion of its current activity cycle).
        # We do this early in case some of our expected message responses don't arrive.

        # Agents should arrive according to a Poisson process.  This is equivalent to
        # each agent independently sampling its next arrival time from an exponential
        # distribution in alternate Beta formation with Beta = 1 / lambda, where lambda
        # is the mean arrival rate of the Poisson process.
        #delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        #self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(int(round(delta_time)))))

        # If the market has closed and we haven't obtained the daily close price yet,
        # do that before we cease activity for the day.  Don't do any other behavior
        # after market close.
        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return

        # Issue cancel requests for any open orders.  Don't wait for confirmation, as presently
        # the only reason it could fail is that the order already executed.  (But requests won't
        # be generated for those, anyway, unless something strange has happened.)
        self.cancelOrders()

        # The ZI agent doesn't try to maintain a zero position, so there is no need to exit positions
        # as some "active trading" agents might.  It might exit a position based on its order logic,
        # but this will be as a natural consequence of its beliefs.

        # In order to use the "strategic threshold" parameter (eta), the ZI agent needs the current
        # spread (inside bid/ask quote).  It would not otherwise need any trade/quote information.

        # If the calling agent is a subclass, don't initiate the strategy section of wakeup(), as it
        # may want to do something different.

        # 20200304 Chris Cho - data_dummy added to send out different messages at each wakeup
        if type(self) == ZeroIntelligencePlus:

            """
            # this is the msssage to query parameter
            if self.data_dummy == 0:

                self.getCurrentParameter(self.symbol)
                self.state = 'AWAITING_PARAMETER'

                # set wakeup to earliest time possible to query order stream
                self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(1e9)))

                self.data_dummy = 1
            """

            # this is the msssage to query spread
            if self.data_dummy == 0:

                self.getCurrentSpread(self.symbol)
                self.state = 'AWAITING_SPREAD'

                # set wakeup to earliest time possible to query order stream
                self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(1e9)))

                self.data_dummy = 1

            # this is the msssage to query order stream
            elif self.data_dummy == 1:

                self.getOrderStream(self.symbol, length = 20)
                self.state = 'AWAITING_ORDER_STREAM'

                # Order arrival time can be fit into exponential distribution
                self.setWakeup(currentTime + self.getWakeFrequency())

                self.data_dummy = 0

        else:
            self.state = 'ACTIVE'

    def placeOrder(self):
        # Called when it is time for the agent to determine a limit price and place an order.
        limit_price = self.limit_price
        history = self.getKnownStreamHistory(self.symbol)

        # Determine the limit price.
        shout_price = limit_price * (1 - self.profit_margin) if self.buy else limit_price * (1 + self.profit_margin)

        def upwards_adjustment(last_shout):

            # set target price
            self.target_price = self.rel_change_up * last_shout + self.abs_change_up

            # using target price and limit price, calculate widrow WH delta
            self.hoff_delta = self.learning_rate * (self.target_price - shout_price)

            # Add momentum based update
            self.momentum_target_price = self.momentum * self.momentum_target_price + (1 - self.momentum) * self.hoff_delta

            # update margin based on this information
            self.profit_margin = (shout_price + self.momentum_target_price) / limit_price - 1

            return None

        def downwards_adjustment(last_shout):

            # set target price
            self.target_price = self.rel_change_down * last_shout + self.abs_change_down

            # using target price and limit price, calculate widrow WH delta
            self.hoff_delta = self.learning_rate * (self.target_price - shout_price)

            # Add momentum based update
            self.momentum_target_price = self.momentum * self.momentum_target_price + (1 - self.momentum) * self.hoff_delta

            # update margin based on this information
            self.profit_margin = (shout_price + self.momentum_target_price) / limit_price - 1

            return None

        # Either place the constructed order, or if the agent could secure (eta * R) surplus
        # immediately by taking the inside bid/ask, do that instead.
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

        if self.buy and ask_vol > 0:
            R_ask = limit_price - ask
            if R_ask >= (limit_price * self.profit_margin):
                log_print("{} desired R = {}, but took R = {} at ask = {} due to eta", self.name, self.profit_margin, R_ask, ask)
                shout_price = ask
            else:
                log_print("{} demands R = {}, limit price {}", self.name, self.profit_margin, shout_price)
        elif (not self.buy) and bid_vol > 0:
            R_bid = bid - limit_price
            if R_bid >= (limit_price * self.profit_margin):
                log_print("{} desired R = {}, but took R = {} at bid = {} due to eta", self.name, self.profit_margin, R_bid, bid)
                shout_price = bid
            else:
                log_print("{} demands R = {}, limit price {}", self.name, self.profit_margin, shout_price)

        # Place the order.
        self.placeLimitOrder(self.symbol, self.order_size, self.buy, int(shout_price))

        # Update margin
        if history.empty == False:

            last_shout = history.loc[0, 'limit_price'] # note that "limit_price" here refers to the column name. Therefore the "last_shout" is the first entry of this column

            # for a seller
            if self.buy == False:
                
                # if the transaction value is not empty - i.e. if the transaction occurred
                if history.loc[0, 'transactions'] != []:

                    # if the ask price is too low, seller raises the ask price
                    if shout_price < last_shout:

                        upwards_adjustment(last_shout)

                    # otherwise, if the last shout was a bid, seller lowers the ask price
                    elif history.loc[0, 'is_buy_order'] == True:
                        
                        downwards_adjustment(last_shout)
                        

                # if the last shout did not undergo transaction
                else:

                    # if the last shout was an ask
                    if history.loc[0, 'is_buy_order'] == False:
                       
                        downwards_adjustment(last_shout)


            # for a buyer
            elif self.buy == True:

                # if the transaction value is not empty - i.e. if the transaction occurred
                if history.loc[0, 'transactions'] != []:

                    # if the bid price is too high, buyer lowers bid price
                    if shout_price > last_shout:

                        downwards_adjustment(last_shout)

                    # otherwise, if the last shout was an ask, buyer raises bid price
                    elif history.loc[0, 'is_buy_order'] == False:
                        
                        upwards_adjustment(last_shout)

                # if the last shout did not undergo transaction
                else:

                    # if the last shout was a bid, buyer raises bid price
                    if history.loc[0, 'is_buy_order'] == True:

                        upwards_adjustment(last_shout)

    def receiveMessage(self, currentTime, msg):
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receiveMessage(currentTime, msg)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        # 20200304 Chris Cho:
        # Note that we only need one of the two if statements below, as this section is only
        # used to execute, not to query the information which we've received. the "super receive message"
        # does the lifting i.e. receiving the message for us to query

        if self.state == 'AWAITING_SPREAD':
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if msg.body['msg'] == 'QUERY_SPREAD':
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed: return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.

                self.state = 'AWAITING_WAKEUP'


        elif self.state == 'AWAITING_ORDER_STREAM':
            
            if msg.body['msg'] == 'QUERY_ORDER_STREAM':

                if self.mkt_closed: return

                self.placeOrder()

                self.state = 'AWAITING_WAKEUP'

        

    # Internal state and logic specific to this agent subclass.

    # Cancel all open orders.
    # Return value: did we issue any cancellation requests?
    def cancelOrders(self):
        if not self.orders: return False

        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True

    def getWakeFrequency(self):
        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        return pd.Timedelta('{}ns'.format(int(round(delta_time))))