```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/titanic/train.csv
    
    /kaggle/input/titanic/test.csv
    
    /kaggle/input/titanic/gender_submission.csv



```python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
```

    % of women who survived: 0.7420382165605095



```python
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
```

    % of men who survived: 0.18890814558058924



```python
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```

    Your submission was successfully saved!



```python
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
```

데이터는 이미 훈련용과 테스트용이 나눠져 있음. 학습을 위해 따로 분류할 필요는 없음.

데이터는 타이타닉호 탑승객들의 신상정보를 series로 갖는 dataframe으로 구성되어있다. 승객별로 생존여부가 모두 라벨링되어 있으므로 타이타닉 문제는 지도학습에 해당한다. 최종적으로 생존(1)과 사망(0)으로 분류하였다.

```python
train.shape
```




    (891, 12)




```python
test.shape
```




    (418, 11)



훈련데이터는 총 891개의 행과 12개의 열로 되어있다.
테스트는 418개의 행과 11개의 열로 되어있는데 이는 Survived의 열이 빠진 상태이기 때문이다.


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 891 entries, 0 to 890
    
    Data columns (total 12 columns):
    
     #   Column       Non-Null Count  Dtype  
    
    ---  ------       --------------  -----  
    
     0   PassengerId  891 non-null    int64  
    
     1   Survived     891 non-null    int64  
    
     2   Pclass       891 non-null    int64  
    
     3   Name         891 non-null    object 
    
     4   Sex          891 non-null    object 
    
     5   Age          714 non-null    float64
    
     6   SibSp        891 non-null    int64  
    
     7   Parch        891 non-null    int64  
    
     8   Ticket       891 non-null    object 
    
     9   Fare         891 non-null    float64
    
     10  Cabin        204 non-null    object 
    
     11  Embarked     889 non-null    object 
    
    dtypes: float64(2), int64(5), object(5)
    
    memory usage: 83.7+ KB



```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 418 entries, 0 to 417
    
    Data columns (total 11 columns):
    
     #   Column       Non-Null Count  Dtype  
    
    ---  ------       --------------  -----  
    
     0   PassengerId  418 non-null    int64  
    
     1   Pclass       418 non-null    int64  
    
     2   Name         418 non-null    object 
    
     3   Sex          418 non-null    object 
    
     4   Age          332 non-null    float64
    
     5   SibSp        418 non-null    int64  
    
     6   Parch        418 non-null    int64  
    
     7   Ticket       418 non-null    object 
    
     8   Fare         417 non-null    float64
    
     9   Cabin        91 non-null     object 
    
     10  Embarked     418 non-null    object 
    
    dtypes: float64(2), int64(4), object(5)
    
    memory usage: 36.0+ KB


info()를 통해 데이터프레임에 대한 더 자세한 정보를 얻을 수 있다.


```python
def chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
```

각 항목들을 시각화하여 대략적인 상태를 파악하고 계수들끼리의 상관관계를 알아보기 위해 막대 그래프로 시각화하였다. 위의 함수는 막대 그래프로 그려주는 함수다.


```python
chart('Sex')
chart('Pclass')
chart('Embarked')
chart('SibSp')
```
    
<img src = "https://raw.githubusercontent.com/coachhknu/coachhknu.github.io/master/_posts/output_16_0.png">
<img src = "https://raw.githubusercontent.com/coachhknu/coachhknu.github.io/master/_posts/output_16_1.png">
<img src = "https://raw.githubusercontent.com/coachhknu/coachhknu.github.io/master/_posts/output_16_2.png">
<img src = "https://raw.githubusercontent.com/coachhknu/coachhknu.github.io/master/_posts/output_16_3.png">

일단 성별, 사회적 지위, 탑승지역, 타이타닉에 탑승한 형제,자매 수를 시각화 하였다. 그 결과 여성보다 남성이 더 많이 생존했고, 지위가 높거나 가족이 있는 승객이 더 높은 생존율을 보이고 있다.


```python
train_test_data = [train, test]

sex_map = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_map)
```

훈련데이터와 테스트 데이터를 한번에 전처리 하기 위해서 하나의 리스트로 묶어서 한번에 처리하였다.


```python
train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Alone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['Family'] > 1, 'Alone'] = 0
```

SibSp & Parch 항목은 결국 같이 탑승한 가족의 수로 볼 수 있으므로 하나로 묶어주었다. Family 항목으로 묶고 아닐경우 Alone으로 묶어보았다.


```python
chart('Alone')

```


    
<img src = "https://raw.githubusercontent.com/coachhknu/coachhknu.github.io/master/_posts/output_23_0.png">
    


확인 결과 혼자 탄 사람이 더 많이 사망한 것을 확인했다.


```python
train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Family</th>
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_list=[]
for i in range(1,4):
    series = train[train['Pclass'] == i]['Embarked'].value_counts()
    class_list.append(series)

df = pd.DataFrame(class_list)
df.index = ['1st', '2nd', '3rd']
df.plot(kind="bar", figsize=(10,5))
```




    <AxesSubplot:>




    
<img src = "https://raw.githubusercontent.com/coachhknu/coachhknu.github.io/master/_posts/output_26_1.png">
    


막대 그래프를 보면 낮은 지위의 사람들이 더 낮은 생존율을 보였다. 이를 확실히 확인해보기 위해 거주 지역의 차이를 시각화하였다. 그 결과 Q지역이 상대적으로 빈곤한 지역으로 보였다.


```python
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
```

확인 결과 대부분의 사람들이 S지역에서 탑승했으므로 결측치는 S로 채워줬다.


```python
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    
train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Family</th>
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



탑승지역을 숫자로 치환해서 입력


```python
for dataset in train_test_data:
    dataset['Marry'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
```

이름은 항목에 있지만 이름이 생존율에 유의미한 영향을 미쳤다고 생각하기는 어렵다. 그러나 이름에 무엇이 붙었냐에 따라 결혼/미혼 여부 판단이 가능할 것으로 판단된다. 따라서 이름 옆에 특정 부분을 추출하여 Marry에 저장했다.


```python
train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Family</th>
      <th>Alone</th>
      <th>Marry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['Marry'].value_counts()
```




    Mr          517
    Miss        182
    Mrs         125
    Master       40
    Dr            7
    Rev           6
    Mlle          2
    Major         2
    Col           2
    Countess      1
    Capt          1
    Ms            1
    Sir           1
    Lady          1
    Mme           1
    Don           1
    Jonkheer      1
    Name: Marry, dtype: int64




```python
test['Marry'].value_counts()
```




    Mr        240
    Miss       78
    Mrs        72
    Master     21
    Col         2
    Rev         2
    Ms          1
    Dr          1
    Dona        1
    Name: Marry, dtype: int64



value_counts() 함수를 사용하여 호칭을 분류하여 그 숫자를 카운트해보았다.


```python
for dataset in train_test_data:
    dataset['Marry'] = dataset['Marry'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)
```

서양에서 주로 사용하는 mr,miss,mrs,master를 제외한 다른 항목들은 하나로 묶어서 취급했다.


```python
train['Marry'].value_counts()
```




    0    517
    1    182
    2    125
    3     40
    4     27
    Name: Marry, dtype: int64




```python
test['Marry'].value_counts()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /opt/conda/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3360             try:
    -> 3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:


    /opt/conda/lib/python3.7/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    /opt/conda/lib/python3.7/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'Marrye'

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    /tmp/ipykernel_19/2940886444.py in <module>
    ----> 1 test['Marrye'].value_counts()
    

    /opt/conda/lib/python3.7/site-packages/pandas/core/frame.py in __getitem__(self, key)
       3456             if self.columns.nlevels > 1:
       3457                 return self._getitem_multilevel(key)
    -> 3458             indexer = self.columns.get_loc(key)
       3459             if is_integer(indexer):
       3460                 indexer = [indexer]


    /opt/conda/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:
    -> 3363                 raise KeyError(key) from err
       3364 
       3365         if is_scalar(key) and isna(key) and not self.hasnans:


    KeyError: 'Marrye'



```python
chart('Marry')
```

이후 막대 그래프를 확인해보니 mr의 사망율이 압도적으로 높았고 mrs,miss가 생존율이 높은 반면 가족이 없는 miss의 경우 상대적으로 낮은 것을 확인했다.


```python
train['Cabin'].value_counts()
```


```python
train['Cabin'] = train['Cabin'].str[:1]
```

cabin은 방번호 임으로 숫자를 제외하고 알파벳만 추출한다. 


```python
class_list=[]
for i in range(1,4):
    a = train[train['Pclass'] == i]['Cabin'].value_counts()
    class_list.append(a)

df = pd.DataFrame(class_list)
df.index = ['1st', '2nd', '3rd']
df.plot(kind="bar", figsize=(10,5))
```

막대 그래프 확인결과 1등급의 생존율이 가장 높고 그다음 2등급, 3등급이 가장 낮았다.


```python
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("marry")["Age"].transform("median"), inplace=True)
```

나이에 대한 정보에는 누락된 값이 존재했다. 따라서 Marry에 해당하는 그룹의 가운데값을 결측치로 바꿔준다.


```python
g = sns.FacetGrid(train, hue="Survived", aspect=4)
g = (g.map(sns.kdeplot, "Age").add_legend())
```

나이대별 생존율을 비교하기 위해 선 그래프를 이용해보았다.


```python
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
```


```python
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 15, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 25), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 60), 'Age'] = 4
    dataset.loc[dataset['Age'] > 60, 'Age'] = 5
```

각 데이터를 더 잘 비교하기 위해 나잇대의 그룹을 더 세밀하게 나눠보았다.


```python
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
```

Fare은 탑승요금인데 높은 등급일수록 높다. 따라서 Fare에서 누락된 값은 Pclass의 중간값을 사용해준다.


```python
g = sns.FacetGrid(train, hue="Survived", aspect=4)
g = (g.map(sns.kdeplot, "Fare")
     .add_legend()
     .set(xlim=(0, train['Fare'].max()))) # x축 범위 설정
```

승객별로 탑승요금의 편차가 굉장히 크고 분포는 우측 꼬리가 길게 편향되어 있다. 따라서 데이터를 그룹화할때 길이가 아닌 개수를 기준으로 나눈다음 Farebin이라는 열에 저장한다.


```python
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
```


```python
pd.qcut(train['Fare'], 4)
```


```python
drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
```

이제 훈련에서 제외할 항목들을 삭제해준다.


```python
train.head()
```


```python
drop_column2 = ['PassengerId', 'Survived']
train_data = train.drop(drop_column2, axis=1)
target = train['Survived']
```

PassengerId는 필요없으므로 삭제해주고 survived는 결과값임으로 마찬가지로 삭제해준다. 이 둘은 따로 target에 저장후 훈련데이터에서는 삭제해주었다.


```python
from sklearn.linear_model import LogisticRegression
```


```python
clf = LogisticRegression()
clf.fit(train_data, target)
test_data = test.drop("PassengerId", axis=1)
predict = clf.predict(test_data)
```

로지스틱 회귀의 모델을 생성하여 점수를 측정해보았다.


```python
last = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : predict})

last.to_csv('last.csv', index=False)
```

예측 결과를 PassengerId와 매치시켜 데이터프레임으로 묶은 뒤 csv파일로 저장하였다.


```python
last = pd.read_csv("last.csv")
last.head()
```
