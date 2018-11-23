# coding:utf8

# ############################
# 对54个城市进行评分并画图
# 输入：每个城市的志愿者人数和志愿团体数，以及项目数和项目总时长
# 输出：54个城市评分靠前的10个
# ###########################

from pyspark import SQLContext, SparkContext
from pyspark import Row
from pyspark import StorageLevel
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
from pylab import mpl

reload(sys)
sys.setdefaultencoding('utf8')

sc = SparkContext()
sqlContext = SQLContext(sc)

# 读取每个城市志愿者人次和志愿团体数的数据
org_mem = []
with open('G:/pythontest/myfirstexample/src/zyz/data/region_mem&org_pop.json') as f:
    for line in f.readlines():
        data = json.loads(line)
        data['region_id'] = float(data['region_id'])
        data['org'] = float(data['org'])
        data['member'] = float(data['member'])
        org_mem.append(data)
org_mem = pd.DataFrame(org_mem)

# 读取每个城市的项目数和项目总时长的数据
opp_hour = pd.read_csv('G:/pythontest/myfirstexample/src/zyz/output/opp_count_hourSum.csv')
opp_hour.drop(opp_hour.columns[[0]], axis=1, inplace=True)
print
type(opp_hour)
opp_hour.columns = ['city', 'region_id', 'pop', 'opp', 'hour']

data = pd.merge(opp_hour, org_mem, on='region_id', how='left')

a = []
for i in [3, 4, 5, 6]:
    a.append(max(data.iloc[:, i].values))

# 归一化
data['opp'] = data['opp'] / a[0]
data['hour'] = data['hour'] / a[1]
data['member'] = data['member'] / a[2]
data['org'] = data['org'] / a[3]

# 乘以40加60
data['score'] = (0.25 * data['opp'] + 0.25 * data['hour'] + 0.25 * data['member'] + 0.25 * data['org']) * 40 + 60

output = data.sort_values(by='score', ascending=False)

index_x = []
index_y = []

for i in range(10):
    index_y.append(round(output['score'].values[i], 2))
for i in range(10):
    index_x.append(output.iloc[i, 0])

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)

# 为了防止乱序，这里输入的是list，下面设置ticks改名
ax.bar(x=np.arange(len(index_x)), height=index_y, width=0.8, color='red')

# ax.set_xticks(np.arange(n2) + 0.5 * width)
# ax.set_xticklabels(clusters, fontsize=23, fontproperties = 'Times New Roman')
# ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85])
# ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=20, fontproperties = 'Times New Roman')

ax.grid(ls=':', lw=1.5)

ax.set_xlabel('城市', fontsize=25)
ax.set_ylabel('活跃度', fontsize=25)
ax.set_title('城市活跃度中排名前10的城市的活跃度', fontsize=30)
# 加入文本
x = range(10)
for i in range(len(index_x)):
    ax.text(x[i], index_y[i], index_y[i], ha='center', va='bottom', fontsize=20)

ax.set_xticklabels(index_x)

plt.xticks(np.arange(len(index_x)), index_x, fontsize=20)
plt.yticks(fontsize=25)

plt.savefig('G:/pythontest/myfirstexample/src/zyz/data/city_score.png')
plt.show()