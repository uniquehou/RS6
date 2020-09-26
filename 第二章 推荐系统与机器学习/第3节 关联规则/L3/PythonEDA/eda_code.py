import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.font_manager import FontProperties

def scatter():
    N = 500
    x = np.random.randn(N)
    y = np.random.randn(N)
    plt.scatter(x, y, marker='x')
    plt.show()
    df = pd.DataFrame({'x':x, 'y': y})
    sns.jointplot(x='x', y='x', data=df, kind='scatter')
    plt.show()

def box_plots():
    data = np.random.normal(size=(10,4))
    labels = ['A', 'B', 'C', 'D']
    plt.boxplot(data, labels=labels)
    plt.show()
    df = pd.DataFrame(data, columns=labels)
    sns.boxplot(data=df)
    plt.show()

def pie_chart():
    nums = [25, 33, 37]
    labels = ['ADB', 'APC', 'TK']
    plt.pie(x=nums, labels=labels)
    plt.show()

def pie_chart2():
    data = {
        'ADC': 25,
        'APC': 33,
        'TK': 37
    }
    data = pd.Series(data)
    data.plot(kind='pie', label='heros')
    plt.show()

def thermodynamic():
    np.random.seed(33)
    data = np.random.rand(3,3)
    heatmap = sns.heatmap(data)
    plt.show()

def spider_chart():
    labels = np.array(['推进', 'KDA', '生存', '团战', '发育', '输出'])
    stats = [76, 58, 67, 97, 86, 58]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    ax.set_thetagrids(angles[:-1]*180/np.pi, labels, FontProperties=font)
    plt.show()

# 二元变量分布图
def jointplot():
    flights = sns.load_dataset('flights')
    sns.jointplot(x='year', y='passengers', data=flights, kind='scatter')
    sns.jointplot(x='year', y='passengers', data=flights, kind='kde')
    sns.jointplot(x='year', y='passengers', data=flights, kind='hex')
    plt.show()

# 成对关系图
def pairplot():
    flights = sns.load_dataset('flights')
    sns.pairplot(flights)
    plt.show()

def thermodynamic2():
    flights = sns.load_dataset('flights')
    # pivot：行转列
    flights = flights.pivot('month', 'year', 'passengers')
    sns.heatmap(flights)
    sns.heatmap(flights, linewidths=5, annot=True, fmt='d')
    plt.show()


if __name__ == '__main__':
    # scatter()
    # box_plots()
    # pie_chart()
    # pie_chart2()
    # thermodynamic()
    # spider_chart()
    jointplot()
    pairplot()
    thermodynamic2()