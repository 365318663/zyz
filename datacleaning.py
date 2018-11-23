# coding:utf-8

# ##############################
# 数据清理，得出70个省市区的志愿项目数和项目总时长
# 输入：opp
# 输出：得出60多个省市区的志愿项目数和项目总时长
# ############################

from pyspark import SQLContext,SparkContext
from pyspark import Row
from pyspark import StorageLevel
import pandas as pd
import numpy as np
import xlrd
# import time

sc = SparkContext()
sqlContext = SQLContext(sc)

# 提取数据
rdd = sc.textFile("/zyzdata_analyze/opp.csv") \
    .map(lambda line: line.split(",")) \
    .filter(lambda line:int(line[0])>1514736000 and int(line[0])<1530374399 and line[2]>0.0) \
    .map(lambda line:(int(line[0]),int(line[1]),float(line[2]),int(line[3])))
data = sqlContext.createDataFrame(rdd, ['opp_end_date', 'opp_district', 'opp_hour', 'opp_province'])

# 把省一级信息取出
province_count = data.groupby('opp_province').count().withColumnRenamed('opp_province', 'opp_district')
province_hour = data.groupby('opp_province').sum('opp_hour').withColumnRenamed('opp_province', 'opp_district')
province = sqlContext.createDataFrame([[6490], [2], [5900], [7907]], ['opp_district'])
# 仅要北京、重庆、天津、上海四个直辖市的信息
province_count2 = province.join(province_count, province.opp_district == province_count.opp_district, 'left_outer').drop(province_count.opp_district)
province_hour2 = province.join(province_hour, province.opp_district == province_hour.opp_district, 'left_outer').drop(province_hour.opp_district)
# 把市一级信息取出
count = data.groupby('opp_district').count()
hour = data.groupby('opp_district').sum('opp_hour')
# 把省市信息union在一起
count = count.unionAll(province_count2)
hour = hour.unionAll(province_hour2)
# 把hour和count信息union在一起
data2 = count.join(hour, count.opp_district == hour.opp_district, 'left_outer').drop(count.opp_district)

data3 = data2.toPandas()

# 读取城市，城市代码，总人口
# workbook = xlrd.open_workbook('G:/WSPN/zyz/new/city_info.xls')
workbook = xlrd.open_workbook('/home/ldt/city_info.xls')
booksheet = workbook.sheet_by_name(u'Sheet1')
nrows = booksheet.nrows
ncols = booksheet.ncols
xcols = [3,4,14]
city_data = []
for i in range(1,nrows):
    city_data.append([booksheet.cell(i,3).value.encode('utf8'), int(booksheet.cell(i,4).value), booksheet.cell(i,14).value])

city = pd.DataFrame(data = city_data, columns = ['city','opp_district','people_num'])

output = pd.merge(city, data3, on='opp_district', how='left')
output.rename(columns={'opp_district':'opp_district\\province'}, inplace = True)
output = output.fillna(0)

output['count'] =  output['count']/output['people_num']
output['sum(opp_hour)'] = output['sum(opp_hour)']/output['people_num']
output.columns = ['市/省','市/省代码','人口数（百万）','志愿项目数（每百万人）','志愿项目总时长(每百万人)']

output.to_csv('/home/ldt/result/opp_count_hourSum.csv')

