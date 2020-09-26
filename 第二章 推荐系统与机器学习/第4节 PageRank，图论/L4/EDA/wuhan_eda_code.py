import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def line_chart(provinceName):
    result = clean_df[clean_df['provinceName']==provinceName]
    result = result.sort_values(by="province_confirmedCount", ascending=True)
    print(result)
    plt.plot(result['updateTime'], result['province_confirmedCount'])
    plt.show()
    sns.lineplot(x='updateTime', y='province_confirmedCount', data=result)
    plt.show()

data = pd.read_csv('DXYArea.csv')
df = data[['provinceName', 'province_confirmedCount', 'updateTime']]
df['updateTime'] = df['updateTime'].str[0:10]
clean_df = df.drop_duplicates(['provinceName', 'updateTime'], keep='first')
line_chart('北京市')