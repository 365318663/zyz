# coding:utf-8

# ##############################
# 测试opp_hour文件中，是否有一个志愿者多次进行同一个项目的志愿活动
# 输入：opp_hour
# 输出：志愿者id，项目id，出现次数
# ############################


from pyspark import SQLContext, SparkContext
from pyspark import Row
import numpy as np
import pandas as pd
from pyspark import StorageLevel

sc = SparkContext()
sqlContext = SQLContext(sc)

# 读取特称工程第一步的输出数据
rdd = sc.textFile("/zyzdata/opp_hour.csv") \
    .map(lambda line: line.split(",").strip('\"')) \
    .filter(lambda line: int(line[5]) == 1) \
    .map(lambda line: (int(line[2]), int(line[3])))

data = sqlContext.createDataFrame(rdd, ['opp_id', 'vol_id'])
output = data.groupBy('opp_id', 'vol_id').count()

output.repartition(1).write.json('/zyzdata_output/text')