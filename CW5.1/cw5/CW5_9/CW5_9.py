import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate



df = pd.read_csv('iris_dataset.csv') 
print('The Original data frame:') 
print(tabulate(df.head(5),headers='keys',tablefmt='fancy_grid')) 
print("\n"+"*"*100+"\n")
print('The information of data frame:') 
print(df.info()) 
print(df.shape) 
print(f'\nCheck NAN  values:') 
print(df.isnull().sum())
print(f"\n Is there any NAN values:{df.isnull().sum().any()}\n") 
print("\n"+"*"*100+"\n")


X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
Y = df['target']
class_name = ['setosa', 'versicolor', 'virginica']


stats = ['min', 'max', 'mean', 'var']

result = df.groupby('target').agg({
    'sepal length (cm)': stats,
    'sepal width (cm)': stats, 
    'petal length (cm)': stats,
    'petal width (cm)': stats
})

stats = ['min', 'max', 'mean', 'var']
result = df.groupby('target').agg({col: stats for col in X.columns})

for idx in range(3):
    print(f"{'='*30}")
    print(f"\n{class_name[idx]} -:")
    print(f"{'='*30}")
    print(result.loc[idx].round(2))
print("\n"+"*"*100+"\n")

#---->>>    #grouped = df.groupby('target')
            #features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

            #for idx in range(3):
            #    print(f"\n{'='*30}")
            #    print(f"==== {class_name[idx]} ====")
            #    print(f"{'='*30}")
            #    group_data = grouped.get_group(idx)
            #    
            #    for feature in features:
            #        print(f"\n{feature}:")
            #        print(f"  min: {group_data[feature].min():.2f}")
            #        print(f"  max: {group_data[feature].max():.2f}")
            #        print(f"  range: {group_data[feature].max() - group_data[feature].min():.2f}")
            #        print(f"  mean: {group_data[feature].mean():.2f}")
            #        print(f"  var: {group_data[feature].var():.2f}")
