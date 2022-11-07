from TinyStatistician import TinyStatistician

tstat = TinyStatistician()
a = [1, 42, 300, 10, 59]
print(tstat.mean(a))
# Expected result: 82.4
print(tstat.median(a))
# Expected result: 42.0
print(tstat.quartile(a))
# Expected result: [10.0, 59.0]
print(tstat.var(a))
# Expected result: 12279.439999999999
print(tstat.std(a))
# Expected result: 110.8126346586886
print(tstat.percentile(a, 3))
print(tstat.percentile(a, 10))
# Output 4.6
print(tstat.percentile(a, 15))
# Output: 6.4
print(tstat.percentile(a, 20))
# Output: 8.2
print(tstat.percentile(a, 50))
# Output: 42
