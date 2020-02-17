

```python
##
```


```python
import numpy as np
```


```python
irises = pd.read_csv("/home/bartek/Documents/ROCHE_zadanie/rochepolskajuniordatascience/Graduate - IRISES dataset (2019-06).csv", sep = '|', header = 0)
```


```python
irises.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 150 entries, 1 to 150
    Data columns (total 5 columns):
    Sepal.Length    150 non-null float64
    Sepal.Width     149 non-null float64
    Petal.Length    150 non-null float64
    Petal.Width     150 non-null object
    Species         150 non-null object
    dtypes: float64(3), object(2)
    memory usage: 7.0+ KB



```python
irises['Petal.Width'] = irises['Petal.Width'].str.replace(',' , '.')

```


```python
irises['Petal.Width'] = irises['Petal.Width'].astype(float)

```


```python
irises.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 150 entries, 1 to 150
    Data columns (total 5 columns):
    Sepal.Length    150 non-null float64
    Sepal.Width     149 non-null float64
    Petal.Length    150 non-null float64
    Petal.Width     150 non-null float64
    Species         150 non-null object
    dtypes: float64(4), object(1)
    memory usage: 7.0+ KB



```python
irises.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>149.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.061745</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.433963</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
irises.isnull().sum()
```




    Sepal.Length    0
    Sepal.Width     1
    Petal.Length    0
    Petal.Width     0
    Species         0
    dtype: int64




```python
irises['Sepal.Width'].fillna(irises['Sepal.Width'].mean(), inplace=True)
```


```python
irises.isnull().sum()
```




    Sepal.Length    0
    Sepal.Width     0
    Petal.Length    0
    Petal.Width     0
    Species         0
    dtype: int64




```python
irises.Species
```




    1         setosa
    2         setosa
    3         setosa
    4         setosa
    5         setosa
    6         setosa
    7         setosa
    8         setosa
    9         setosa
    10        setosa
    11        setosa
    12        setosa
    13        setosa
    14        setosa
    15        setosa
    16        setosa
    17        setosa
    18        setosa
    19        setosa
    20        setosa
    21        setosa
    22        setosa
    23        setosa
    24        setosa
    25        setosa
    26        setosa
    27        setosa
    28        setosa
    29        setosa
    30        setosa
             ...    
    121    virginica
    122    virginica
    123    virginica
    124    virginica
    125    virginica
    126    virginica
    127    virginica
    128    virginica
    129    virginica
    130    virginica
    131    virginica
    132    virginica
    133    virginica
    134    virginica
    135    virginica
    136    virginica
    137    virginica
    138    virginica
    139    virginica
    140    virginica
    141    virginica
    142    virginica
    143    virginica
    144    virginica
    145    virginica
    146    virginica
    147    virginica
    148    virginica
    149    virginica
    150    virginica
    Name: Species, Length: 150, dtype: object




```python
X = irises.drop('Species',axis='columns')
y = irises.Species
```


```python
from sklearn.model_selection import train_test_split

```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
```


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
model = RandomForestClassifier(n_estimators = 75, random_state = 42)
```


```python
model.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=75,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)




```python
model.score(X_test, y_test)
```




    0.9666666666666667




```python
from sklearn.metrics import accuracy_score 
acc = accuracy_score(y_test, y_pred)
print("acc: ", acc)
```


```python
y_predicted = model.predict(X_test)
```


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
```




    array([[ 9,  0,  0],
           [ 0, 10,  0],
           [ 0,  1, 10]])




```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
```




    Text(69.0, 0.5, 'Truth')




![png](output_23_1.png)



```python
from sklearn import metrics
```


```python
print(metrics.classification_report(y_predicted, y_test))
```

                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00         9
      versicolor       1.00      0.91      0.95        11
       virginica       0.91      1.00      0.95        10
    
        accuracy                           0.97        30
       macro avg       0.97      0.97      0.97        30
    weighted avg       0.97      0.97      0.97        30
    



```python

```
