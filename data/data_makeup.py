# 生成全国数据（年份、肥胖率、GDP）
import numpy as np

years = np.arange(2010, 2024)
obesity_rate = 0.15 + 0.01*(years - 2010) + np.random.normal(0, 0.02, len(years))
gdp = 3000 + 800*(years - 2010)

national_data = np.column_stack([years, obesity_rate, gdp])
np.savetxt("data/national/national.csv", national_data, delimiter=',', header="year,obesity_rate,gdp", fmt=['%d', '%.3f', '%.1f'], encoding='utf-8')

# 生成基层数据（月份、肥胖率）
months = np.arange(1, 13)
local_obesity = 0.25 + 0.05*np.sin(2*np.pi*months/12) + np.random.normal(0, 0.03, 12)

local_data = np.column_stack([months, local_obesity])
np.savetxt("data/local/local.csv", local_data, delimiter=',', header="month,obesity_rate", fmt=['%d', '%.3f'], encoding='utf-8')