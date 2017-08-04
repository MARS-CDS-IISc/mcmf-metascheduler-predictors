from pandas import read_csv, DataFrame, Series

import statsmodels.api as sm
import rpy2.robjects as R
from rpy2.robjects.packages import importr
import pandas.rpy.common as com

from pandas import date_range
from sklearn.metrics import mean_squared_error
from math import sqrt
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure, show
import sys
import numpy as np

class PricePredictor:
    stats = importr('stats')
    tseries = importr('tseries')
    forecast = importr('forecast')
    def __init__(self, location):
        self.full_data = []
        self.training_data = []
        self.forecast_period = 24 # default forecast for next day
        self.sarimaModel = None
        self.dataset = None       # Pandas DataFrame
        self.location = location
        self.zero_time = 0

    def get_beautiful_plot(self):
        fig = figure()
        ax = fig.add_subplot(111, autoscale_on=True)                
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('legend', fontsize=14)
        return fig, ax
            
    def load_data(self, datasetPath, datasetName):
        datafile = datasetPath + datasetName
        df = open(datafile, 'r')
        # Expecting the file to be list of price values one after the other on each line
        ldf = df.readlines()
        for l in ldf:
            if l.split()[0] == "\"price\"":
                continue
            self.full_data.append(float(l.split(',')[0]))

        self.dataset = read_csv(datafile)
        self.dataset = self.dataset.price

        uQuartile = np.percentile(np.array(self.full_data), 90.0)
        lQuartile = np.percentile(np.array(self.full_data), 10.0)
        #print 'Upper quartile:', uQuartile, 'Lower quartile:', lQuartile
        self.square_data = []
        # 12am to 12 pm is off peak
        # 12 pm to 12 am is on peak
        for i in range(len(self.full_data)/24):
            for i in range(12):
                self.square_data.append(lQuartile)
            for i in range(12):
                self.square_data.append(uQuartile)
    def get_average_price(self):
        return np.mean(self.full_data)

    def initialize_start_time(self, ts):
        self.zero_time = ts
    '''
    * Find current hour
    * We know prices till end of the day
    * If we require a prediction beyond that, predict for appropriate number of hours
    * Calculate the total energy cost as watts*time*unit cost
    * current_time is in seconds
    * Need some exception handling here because we don't have endless price data
    '''
    def get_price_prediction(self, current_time, power_watts, wait_time, duration, siteid):
        current_time = current_time - self.zero_time
        
        # power_watts convert to megawatt - we'll lose accuracy if we do this
        # power_watts = float(power_watts)/10**6

        current_day = current_time/(60*60*24)
        current_hour = current_time/(60*60)
        current_day_end_time = (current_day + 1)*(60*60*24)
        
        start_time = current_time + wait_time
        end_time = current_time + wait_time + duration
        end_hour = (end_time)/(60*60)

        cost = 0
        required_prices = []
        last_hour_price = 0

        # Case: All prices are known
        if 1 or end_time < current_day_end_time or siteid == 2: # Tennessee is a case where we know the prices beforehand - ARIMA crashes for the on-peak off-peak cycle
            #print 'End hour is:', end_hour+1
            required_prices = self.full_data[current_hour:end_hour+1]
            last_hour_price = self.full_data[end_hour+1]
        else:
            required_prices = []
            predict_begin_hour = start_time/(60*60)
            # Some (if any) prices are known
            if start_time < current_day_end_time:
                today_start_hour = start_time/(60*60)
                today_end_hour = (current_day_end_time)/(60*60)
                required_prices.extend(self.full_data[today_start_hour:today_end_hour])
                predict_begin_hour = today_end_hour

            # Predict for the rest - from predict_begin_hour for duration hours
            prediction_count = 1 + duration/(60*60) 
            train_period_end = predict_begin_hour
            train_period_begin = min(0, predict_begin_hour-3*24)
            pred_values = self.get_predictions(train_period_begin, train_period_end, prediction_count)
            required_prices.extend(pred_values)
            last_hour_price = required_prices[-1]
            required_prices = required_prices[:-1]

        for price in required_prices:
            cost += price*power_watts
            
        remaining_seconds = (duration - (duration/(60*60))*60*60)
        remaining_hours = (remaining_seconds)/(60.0*60.0)
        cost += int(remaining_hours*last_hour_price)

        return cost

    def calculate_price_from_batch_forecast(self, power_watts, wait_time, duration, forecast):
        start_time = wait_time
        end_time = wait_time + duration
        
        start_hour = (start_time)/(60*60)
        end_hour = (end_time)/(60*60)
    
        required_prices = forecast[start_hour:end_hour]
        #last_hour_price = forecast[end_hour]
        last_hour_price = forecast[-1]


        remaining_seconds = (duration - (duration/(60*60))*60*60)
        remaining_hours = (remaining_seconds)/(60.0*60.0)

        cost = 0.0
        for price in required_prices:
            cost += price*power_watts
        cost += remaining_hours*last_hour_price
        cost = float(cost/10**6)
        return cost

    def get_batch_forecast(self, current_time, max_time, siteid):
        #print 'Getting batch forecast for site', siteid, 'ctime:', current_time, 'max_time:', max_time
        
        current_time = current_time - self.zero_time
        
        current_day = current_time/(60*60*24)
        current_hour = current_time/(60*60)
        current_day_end_time = (current_day + 1)*(60*60*24)
        
        start_time = current_time
        end_time = current_time + max_time
        end_hour = (end_time)/(60*60)

        cost = 0
        required_prices = []
        last_hour_price = 0
        
        duration = max_time
        #print 'Duration hours = ', duration/(60*60), 'total secs:', duration
        adjusted_duration = (duration/(60*60) + 1)*3600
        duration = adjusted_duration
        #print 'Adjusted duration hours = ', duration/(60*60), 'total secs:', duration

        if end_hour+1 > len(self.full_data)-1:
            #print 'Reached end of month at site', siteid, 'end_hour:', end_hour, 'current_time=', current_time, 'max_time=', max_time
            return None

        # Case: All prices are known
        # 
        #
        # PUTTING a 1 or clause to stop predicting and return true values always - for bug fix help
        #
        #
        if  end_time < current_day_end_time or siteid == 2: # Tennessee is a case where we know the prices beforehand - ARIMA crashes for the on-peak off-peak cycle
            #print 'End hour is:', end_hour+1
            required_prices = self.full_data[current_hour:end_hour+2]
        else:
            required_prices = []
            predict_begin_hour = start_time/(60*60)
            # Some (if any) prices are known
            if start_time < current_day_end_time:
                today_start_hour = start_time/(60*60)
                today_end_hour = (current_day_end_time)/(60*60)
                required_prices.extend(self.full_data[today_start_hour:today_end_hour])
                predict_begin_hour = today_end_hour
                if duration/(60*60) <= today_end_hour-current_hour:
                    return required_prices
                duration = duration - 3600*(today_end_hour-today_start_hour)
                if duration < 0:
                    duration = 3600
                
                #print 'Current hour:', current_hour, 'Today end hour:', today_end_hour, 'reduced duration:', duration
            # Predict for the rest - from predict_begin_hour for duration hours
            
            prediction_count = 1 + duration/(60*60) 
            train_period_end = predict_begin_hour
            train_period_begin = min(0, predict_begin_hour-3*24)
            #print 'Requesting predictions from ARIMA using', 'begin:',train_period_begin, 'end:',train_period_end, 'count:',prediction_count
            pred_values = self.get_predictions(train_period_begin, train_period_end, prediction_count)
            required_prices.extend(pred_values)
        return required_prices



    def get_predictions(self, train_period_begin, train_period_end, prediction_count):
        #print 'Getting training data from', train_period_begin, 'to', train_period_end
        self.training_data = self.full_data[train_period_begin: train_period_end]
        self.forecast_period = prediction_count
        #print self.dataset.iloc[train_period_begin:train_period_end]
        r_df = com.convert_to_r_dataframe(DataFrame(self.dataset.iloc[train_period_begin:train_period_end]))
        y = PricePredictor.stats.ts(r_df)
        orderR = R.IntVector((1,0,1))
        season = R.ListVector({'order': R.IntVector((1,0,1)), 'period' : 24})
        model = PricePredictor.stats.arima(y, order = orderR, seasonal=season,method="ML")
        f = PricePredictor.forecast.forecast(model, h=self.forecast_period)
        #print f
        #print "\n"
        predValues = []
        for item in f.items():
            if item[0] == 'mean':
                for value in item[1].items():
                    predValues.append(value[1])
        return predValues

    '''
    * Calculates actual power consumption of a job.
    * Here duration is the actual runtime of the job and wait time is actual wait time.
    * So, we can calculate this only in the job termination event.
    '''
    def get_actual_price(self, current_time, power_watts, wait_time, duration):
        current_time = current_time - self.zero_time
        
        # power_watts convert to megawatt
        power_watts = float(power_watts)/10**6

        current_day = current_time/(60*60*24)
        current_hour = current_time/(60*60)
        current_day_end_time = (current_day + 1)*(60*60*24)
        
        start_time = current_time + wait_time
        end_time = current_time + wait_time + duration
        end_hour = (end_time)/(60*60)

        cost = 0
        required_prices = []
        last_hour_price = 0

        # Case: All prices are known
        required_prices = self.full_data[current_hour:end_hour+1]
        last_hour_price = self.full_data[end_hour+1]

        for price in required_prices:
            cost += price*power_watts
            
        remaining_seconds = (duration - (duration/(60*60))*60*60)
        remaining_hours = (remaining_seconds)/(60.0*60.0)
        cost += remaining_hours*last_hour_price

        return cost

    def get_current_price(self, start_time):
        cost = 0
        try:
            start_time = start_time - self.zero_time
            start_hour = start_time/(60*60)
            cost = self.full_data[start_hour]
            return cost
        except Exception:
            return cost

    def get_actual_running_cost(self, start_time, end_time, power_watts, square_price=False):
        cost = 0
        try:
            start_time = start_time - self.zero_time
            end_time = end_time - self.zero_time
            duration = end_time - start_time

            power_watts = float(power_watts)/10**6

            start_hour = start_time/(60*60)
            end_hour = end_time/(60*60)

            start_hour = int(start_hour)
            end_hour = int(end_hour)

            required_prices = []
            last_hour_price = 0

            if square_price == False:
                required_prices = self.full_data[start_hour:end_hour+1]
            else:
                required_prices = self.square_data[start_hour:end_hour+1]

            for price in required_prices:
                cost += price*power_watts

            if end_hour+1 <= len(self.full_data):
                if square_price == False:
                    last_hour_price = self.full_data[end_hour+1]            
                else:
                    last_hour_price = self.square_data[end_hour+1]

                remaining_seconds = (duration - (duration/(60*60))*60*60)
                remaining_hours = (remaining_seconds)/(60.0*60.0)
                cost += remaining_hours*last_hour_price
            return cost
        except Exception:
            return cost
    '''
    def get_actual_running_cost(self, start_time, end_time, power_watts):
        print 'get_actual_running_cost:', start_time, end_time, power_watts
        cost = 0
        try:
            start_time = start_time - self.zero_time
            end_time = end_time - self.zero_time
            duration = end_time - start_time

            power_watts = float(power_watts)/10**6

            start_hour = start_time/(60*60)
            end_hour = end_time/(60*60)


            required_prices = []
            last_hour_price = 0

            start_hour = int(start_hour)
            end_hour = int(end_hour)
            #print 'Data end points:', start_hour, end_hour+1, 'data=', self.full_data[start_hour:end_hour+1]

            required_prices = self.full_data[start_hour:end_hour+1]


            for price in required_prices:
                cost += price*power_watts#*1 hour

            if end_hour+1 <= len(self.full_data):
                last_hour_price = self.full_data[end_hour+1]            
                remaining_seconds = (duration - (duration/(60*60))*60*60)
                remaining_hours = (remaining_seconds)/(60.0*60.0)
                cost += remaining_hours*last_hour_price
            print 'price:', cost
            return cost
        except Exception:
            return cost
    '''

def main():
    pp = PricePredictor('test')
    dirname = sys.argv[1]
    fname = sys.argv[2]
    #pp.load_data('/home/prakash/work/metaSched/priceSuite/price_data/', 'Illinois.csv')
    pp.load_data(dirname, fname)
    listActual = []
    listPred = []
    limit = 24
    for train_step in range(27):
        train_start = train_step*24
        train_end = train_start + 24*3
        pred = pp.get_predictions(train_start, train_end, 24)
        actual = pp.full_data[train_end:train_end+24]
        listActual.extend(actual)
        listPred.extend(pred)
        '''
        fig, ax = pp.get_beautiful_plot()
        X = [i+(train_step-1)*24 for i in range(train_end-train_start+1+24)]
        #print len(X[:train_end-train_start+1])
        plt.plot(X, pp.full_data[:train_end-train_start+1+24], color='r')
        #plt.plot(X[train_end-train_start+1:], actual, color='r', linestyle='--')
        plt.plot(X[train_end-train_start+1:], pred, color='g', linestyle='--')
        #plt.plot(X[train_end-train_start+1:], pred, color='b')

        plt.xlabel("Time", fontweight='bold', fontsize=14)
        plt.ylabel(r"Price (\$)", fontweight='bold', fontsize=14)
        plt.savefig('samplePred.pdf', filetype="pdf")
        '''
        #sys.exit(0)
        #print len(actual), len(pred), train_end
        rms = sqrt(mean_squared_error(actual, pred))
        #print 'Train', train_step
        #print 'RMSE ($):', rms 
        perc_error = 0.0
        for i in range(len(actual)):
            perc_error += abs(actual[i]-pred[i])/actual[i]
        perc_error = perc_error*100.0/len(actual)
        #print 'Average percentage error:', perc_error
    #print 'Net statistics:'
    rms = sqrt(mean_squared_error(listActual, listPred))
    print 'RMSE ($):', rms 
    perc_error = 0.0
    #print len(listActual)
    for i in range(len(listActual)):
        perc_error += abs(listActual[i]-listPred[i])/listActual[i]
    perc_error = perc_error*100.0/len(listActual)
    print 'Average percentage error:', perc_error
    '''
    fig = figure()
    ax = fig.add_subplot(111, autoscale_on=True)                
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize=14)
    
    X = [1+i for i in range(len(listActual))]


    plt.plot(X, listPred, label="Predicted", color='g')
    plt.plot(X, listActual, label="True", color='r')

    plt.xlabel("Time", fontweight='bold', fontsize=14)
    plt.ylabel(r"Price (\$)", fontweight='bold', fontsize=14)
    plt.title("Illinois Electricity Price Prediction", fontweight="bold", fontsize=14)
    legend = ax.legend(loc='upper left',  fancybox=True)



    plt.savefig('samplePred.pdf', filetype="pdf")
    '''
if __name__=="__main__":
    main()
