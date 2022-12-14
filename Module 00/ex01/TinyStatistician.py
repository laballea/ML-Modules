import math


class TinyStatistician():
    def __init__(self):
        pass

    def sum(self, x):
        return sum(x)

    def mean(self, x):
        """
            computes the mean of a given non-empty list or array x, using a for-loop.
            The method returns the mean as a float, otherwise None if x is an empty list or
            array.
        """
        try:
            return float(sum(x) / len(x))
        except Exception:
            return None

    def median(self, x):
        """
            computes the median of a given non-empty list or array x. The method
            returns the median as a float
        """
        try:
            x.sort()
            if (len(x) % 2 == 1):
                return float(x[int((len(x) + 1) / 2 - 1)])
            else:
                return float((x[int(len(x) / 2)] + x[int(len(x) / 2 - 1)]) / 2)
        except Exception:
            return None

    def quartile(self, x):
        """
            computes the 1st and 3rd quartiles of a given non-empty array x.
            The method returns the quartile as a float
        """
        try:
            x.sort()
            return [float(x[int((len(x) + 3) / 4) - 1]), float(x[int((3 * len(x) + 1) / 4) - 1])]
        except Exception:
            return None

    def percentile(self, x, p):
        """
            computes the expected percentile of a given non-empty list or
            array x. The method returns the percentile as a float, otherwise None if x is an
            empty list or array or a non expected type object. The second parameter is the
            wished percentile. This method should not raise any Exception
        """
        try:
            x = sorted(x)
            p = p / 100
            k = (len(x)-1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return x[int(k)]
            d0 = x[int(f)] * (c-k)
            d1 = x[int(c)] * (k-f)
            return d0+d1
        except Exception:
            return None

    def var(self, x):
        """
            computes the variance of a given non-empty list or array x, using a for-
            loop. The method returns the variance as a float
        """
        try:
            var = 0
            sumX = sum(x)
            for el in x:
                var += (el - (1/len(x)*sumX))**2
            return var / len(x)
        except Exception:
            return None

    def std(self, x):
        """
            computes the standard deviation of a given non-empty list or array x,
            using a for-loop. The method returns the standard deviation as a float,
        """
        try:
            return math.sqrt(self.var(x))
        except Exception:
            return None
