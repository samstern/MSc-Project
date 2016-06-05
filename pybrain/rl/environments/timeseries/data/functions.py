
# Helper function for data processing

from pandas import ewma, ewmvar, rolling_mean, rolling_var


""" Creating Technical Indicators
"""

def sampleMovingAverage(rets, lookbackHorizon):
    return rolling_mean(rets,lookbackHorizon)

def exponentialMovingAverage(rets, hl):
    return ewma(rets,halflife=hl)

def movingVariance(rets, lookbackHorizon):
    return rolling_var(rets,lookbackHorizon)

def exponentialMovingVariance(rets, hl):
    return ewmvar(rets,halflife=hl)

def difference(rets, degrees):
    return rets.diff(periods=degrees)

def lag(rets,p):
    return rets.shift(periods=p)

def momentum(rets, start_lag, end_lag):
    return rets.shift(start_lag)-rets.shift(end_lag)
