import numpy as np

def maxmin(time, swe):
    """
    return the maximum, the minimum and the date of each one of a series of values

    :param time: time serie
    :param swe: snow water equivalent series
    :return:
    """
    assert time.ndim == 1 and swe.ndim == 1 and len(time) == len(swe)
    try :
        maximum = np.max(swe)
        index_of_maximum = np.where(swe == maximum)[0][0]
        date_max = time[index_of_maximum]
    except :
        maximum = 'No data'
        date_max = 'No data'
    try :
        minimum = np.min(swe)
        index_of_minimum = np.where(swe == minimum)[0][0]
        date_min = time[index_of_minimum]
    except :
        minimum = 'No data'
        date_min = 'No data'
    return maximum, minimum, date_max, date_min

def amount_snowmelt(maximum, time, swe):
    """
       return the amount of the snow melt knowing the maximum of the swe

       :param maximum: maximum of snow water equivalent
       :param time: time serie
       :param swe: snow water equivalent series
       :return:
       """
    assert time.ndim == 1 and swe.ndim == 1 and len(time) == len(swe)

    sum = 0
    index_of_max = np.where(swe == maximum)[0][0]
    for i in range (index_of_max,len(swe)-1) :
        sum = sum +(swe[i]-swe[i+1])
    return sum

def amount_precipitation(maximum,swe,cum_precip):
    """
          return the amount of the rain fell from the beginning of the year to the maximum of snow

          :param cum_sum: cumulated precipitation of the rain
          :return:
          """
    index_max = np.where(swe == maximum)[0][0]
    try :
        precipitation = cum_precip[index_max]
    except :
        precipitation = 'No data'
    return precipitation

