import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


oecd_bli = pd.read_csv('oecd_bli_2015.csv', thousands=',')
oecd_bli = oecd_bli[oecd_bli['INEQUALITY'] == 'TOT']
oecd_bli = oecd_bli.pivot(index='Country', columns='Indicator', values='Value')
print(oecd_bli.head(2))
print(oecd_bli['Life satisfaction'].head()) # head()默认取出5个


gdp_per_capita = pd.read_csv('gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')
gdp_per_capita.rename(columns={'2015': 'GDP per capita'}, inplace=True)
gdp_per_capita.set_index('Country', inplace=True)
print(gdp_per_capita.head())


# 合并上边两条的数据
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
full_country_stats.sort_values(by='GDP per capita', inplace=True)
print(full_country_stats.tail())


# 制造测试数据，在原有数据上分离
remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))
print(keep_indices)
sample_data = full_country_stats[["GDP per capita", 'Life satisfaction', 'Water quality']].iloc[keep_indices]
missing_data = full_country_stats[["GDP per capita", 'Life satisfaction', 'Water quality']].iloc[remove_indices]
print(sample_data)
print(sample_data['Water quality'].describe())

# 画出数据的样式
# sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3), c='Water quality')
# plt.axis([0, 60000, 0, 10])
# position_text = {
#     "Hungary": (5000, 1),
#     "Korea": (18000, 1.7),
#     "France": (29000, 2.4),
#     "Australia": (40000, 3.0),
#     "United States": (52000, 3.8),
# }
# for country, pos_text in position_text.items():
#     pos_data_x, pos_data_y, z = sample_data.loc[country]
#     country = "U.S." if country == "United States" else country
#     plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
#             arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
#     plt.plot(pos_data_x, pos_data_y, "ro")
# plt.show()

# 使用sklearn求线性模型的解
line1 = linear_model.LinearRegression()
x_sample = np.c_[sample_data['GDP per capita']]
y_sample = np.c_[sample_data['Life satisfaction']]
line1.fit(x_sample, y_sample)
t0, t1 = line1.intercept_[0], line1.coef_[0][0]
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.axis([0, 60000, 0, 10])
X=np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1*X, "b")
plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
plt.show()



if __name__ == '__main__':
    pass

