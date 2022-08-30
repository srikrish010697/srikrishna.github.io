# NASA Turbofan dataset challenge

**Kaggle version of the very well known public data set for asset degradation modeling from NASA. It includes Run-to-Failure simulated data from turbo fan jet engines.**

### Prediction goal

**In this dataset the goal is to predict the remaining useful life (RUL) of each engine in the test dataset. 
RUL is equivalent of number of flights remained for the engine after the last datapoint in the test dataset.**

### Experimental scenario

**Datasets consist of multiple multivariate time series. Time series data is is extracted from a engines of the same type. There are 22 sensor variables extracted from the engine with three operational settings. The engine is operating normally at the start of each time series and starts to degrade at some point during the series. Hence , Predict the remaining useful life (RUL) of the engines for every operational cycle.**


### Datasets

**Training set : the fault grows in magnitude until system failure. 
Testing set : time series ends some time prior to system failure.**

### Dataset description 

**The data are provided as a zip-compressed text file with 27 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:**
1) unit number 

2) time, in cycles

3) operational setting 1

4) operational setting 2

5) operational setting 3

6) sensor measurement 1

7) sensor measurement 2

…
27) sensor measurement 22


```python
import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost
import catboost
plt.rcParams['figure.figsize'] = 20, 20
```

    C:\Users\peesri\Anaconda3\envs\py38\lib\site-packages\tqdm\auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    


```python
#Each row is a snapshot of data taken during a single operational cycle
#No time information given so we don't know operational cycle time and its sampling frequency

index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names
directory = 'C:/Work/Machine_learning_projects/Data/PdM/data/NASA turbofan/CMaps/'
train_df = pd.read_csv(directory+r'\train_FD003.txt', 
                     sep='\s+', 
                     header=None,
                     index_col=False,
                     names=col_names)
train = train_df.copy()
test_df = pd.read_csv(directory+r'\test_FD003.txt', 
                     sep='\s+', 
                     header=None,
                     index_col=False,
                     names=col_names)
test = test_df.copy()
y_test = pd.read_csv(directory+r'\RUL_FD003.txt', 
                      sep='\s+', 
                      header=None,
                      index_col=False,
                      names=['RUL'])
```


```python
train_df
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
      <th>unit_number</th>
      <th>time_cycles</th>
      <th>setting_1</th>
      <th>setting_2</th>
      <th>setting_3</th>
      <th>s_1</th>
      <th>s_2</th>
      <th>s_3</th>
      <th>s_4</th>
      <th>s_5</th>
      <th>...</th>
      <th>s_12</th>
      <th>s_13</th>
      <th>s_14</th>
      <th>s_15</th>
      <th>s_16</th>
      <th>s_17</th>
      <th>s_18</th>
      <th>s_19</th>
      <th>s_20</th>
      <th>s_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>-0.0005</td>
      <td>0.0004</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.36</td>
      <td>1583.23</td>
      <td>1396.84</td>
      <td>14.62</td>
      <td>...</td>
      <td>522.31</td>
      <td>2388.01</td>
      <td>8145.32</td>
      <td>8.4246</td>
      <td>0.03</td>
      <td>391</td>
      <td>2388</td>
      <td>100.0</td>
      <td>39.11</td>
      <td>23.3537</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.0008</td>
      <td>-0.0003</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.50</td>
      <td>1584.69</td>
      <td>1396.89</td>
      <td>14.62</td>
      <td>...</td>
      <td>522.42</td>
      <td>2388.03</td>
      <td>8152.85</td>
      <td>8.4403</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.99</td>
      <td>23.4491</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>-0.0014</td>
      <td>-0.0002</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.18</td>
      <td>1582.35</td>
      <td>1405.61</td>
      <td>14.62</td>
      <td>...</td>
      <td>522.03</td>
      <td>2388.00</td>
      <td>8150.17</td>
      <td>8.3901</td>
      <td>0.03</td>
      <td>391</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.85</td>
      <td>23.3669</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>-0.0020</td>
      <td>0.0001</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.92</td>
      <td>1585.61</td>
      <td>1392.27</td>
      <td>14.62</td>
      <td>...</td>
      <td>522.49</td>
      <td>2388.08</td>
      <td>8146.56</td>
      <td>8.3878</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.96</td>
      <td>23.2951</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0.0016</td>
      <td>0.0000</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>641.68</td>
      <td>1588.63</td>
      <td>1397.65</td>
      <td>14.62</td>
      <td>...</td>
      <td>522.58</td>
      <td>2388.03</td>
      <td>8147.80</td>
      <td>8.3869</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>39.14</td>
      <td>23.4583</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24715</th>
      <td>100</td>
      <td>148</td>
      <td>-0.0016</td>
      <td>-0.0003</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>643.78</td>
      <td>1596.01</td>
      <td>1424.11</td>
      <td>14.62</td>
      <td>...</td>
      <td>519.66</td>
      <td>2388.30</td>
      <td>8138.08</td>
      <td>8.5036</td>
      <td>0.03</td>
      <td>394</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.44</td>
      <td>22.9631</td>
    </tr>
    <tr>
      <th>24716</th>
      <td>100</td>
      <td>149</td>
      <td>0.0034</td>
      <td>-0.0003</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>643.29</td>
      <td>1596.38</td>
      <td>1429.14</td>
      <td>14.62</td>
      <td>...</td>
      <td>519.91</td>
      <td>2388.28</td>
      <td>8144.36</td>
      <td>8.5174</td>
      <td>0.03</td>
      <td>395</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.50</td>
      <td>22.9746</td>
    </tr>
    <tr>
      <th>24717</th>
      <td>100</td>
      <td>150</td>
      <td>-0.0016</td>
      <td>0.0004</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>643.84</td>
      <td>1604.53</td>
      <td>1431.41</td>
      <td>14.62</td>
      <td>...</td>
      <td>519.44</td>
      <td>2388.24</td>
      <td>8135.95</td>
      <td>8.5223</td>
      <td>0.03</td>
      <td>396</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.39</td>
      <td>23.0682</td>
    </tr>
    <tr>
      <th>24718</th>
      <td>100</td>
      <td>151</td>
      <td>-0.0023</td>
      <td>0.0004</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>643.94</td>
      <td>1597.56</td>
      <td>1426.57</td>
      <td>14.62</td>
      <td>...</td>
      <td>520.01</td>
      <td>2388.26</td>
      <td>8141.24</td>
      <td>8.5148</td>
      <td>0.03</td>
      <td>395</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.31</td>
      <td>23.0753</td>
    </tr>
    <tr>
      <th>24719</th>
      <td>100</td>
      <td>152</td>
      <td>0.0000</td>
      <td>0.0003</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>643.64</td>
      <td>1599.04</td>
      <td>1436.06</td>
      <td>14.62</td>
      <td>...</td>
      <td>519.48</td>
      <td>2388.24</td>
      <td>8136.98</td>
      <td>8.5150</td>
      <td>0.03</td>
      <td>396</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.56</td>
      <td>23.0847</td>
    </tr>
  </tbody>
</table>
<p>24720 rows × 26 columns</p>
</div>




```python
#remaining useful life is grouped by unit no
# there are 100 unit nos. so each unit no. has a different R2F cycle time

# RUL_by_unit = R2F_by_unit - current_operational_cycle_no

def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by='unit_number') 
    max_cycle = grouped_by_unit['time_cycles'].max() 
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), 
                            left_on='unit_number',
                            right_index=True)
    # Calculate remaining useful life for each row 
    remaining_useful_life = result_frame['max_cycle'] - result_frame['time_cycles']
    result_frame['RUL'] = remaining_useful_life 
    # drop max_cycle as it's no longer needed 
    result_frame = result_frame.drop("max_cycle", axis=1) 
    return result_frame
```


```python
train = add_remaining_useful_life(train)
test = add_remaining_useful_life(test)
```


```python
train.iloc[:,5:].head()
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
      <th>s_1</th>
      <th>s_2</th>
      <th>s_3</th>
      <th>s_4</th>
      <th>s_5</th>
      <th>s_6</th>
      <th>s_7</th>
      <th>s_8</th>
      <th>s_9</th>
      <th>s_10</th>
      <th>...</th>
      <th>s_13</th>
      <th>s_14</th>
      <th>s_15</th>
      <th>s_16</th>
      <th>s_17</th>
      <th>s_18</th>
      <th>s_19</th>
      <th>s_20</th>
      <th>s_21</th>
      <th>RUL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>518.67</td>
      <td>642.36</td>
      <td>1583.23</td>
      <td>1396.84</td>
      <td>14.62</td>
      <td>21.61</td>
      <td>553.97</td>
      <td>2387.96</td>
      <td>9062.17</td>
      <td>1.3</td>
      <td>...</td>
      <td>2388.01</td>
      <td>8145.32</td>
      <td>8.4246</td>
      <td>0.03</td>
      <td>391</td>
      <td>2388</td>
      <td>100.0</td>
      <td>39.11</td>
      <td>23.3537</td>
      <td>258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>518.67</td>
      <td>642.50</td>
      <td>1584.69</td>
      <td>1396.89</td>
      <td>14.62</td>
      <td>21.61</td>
      <td>554.55</td>
      <td>2388.00</td>
      <td>9061.78</td>
      <td>1.3</td>
      <td>...</td>
      <td>2388.03</td>
      <td>8152.85</td>
      <td>8.4403</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.99</td>
      <td>23.4491</td>
      <td>257</td>
    </tr>
    <tr>
      <th>2</th>
      <td>518.67</td>
      <td>642.18</td>
      <td>1582.35</td>
      <td>1405.61</td>
      <td>14.62</td>
      <td>21.61</td>
      <td>554.43</td>
      <td>2388.03</td>
      <td>9070.23</td>
      <td>1.3</td>
      <td>...</td>
      <td>2388.00</td>
      <td>8150.17</td>
      <td>8.3901</td>
      <td>0.03</td>
      <td>391</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.85</td>
      <td>23.3669</td>
      <td>256</td>
    </tr>
    <tr>
      <th>3</th>
      <td>518.67</td>
      <td>642.92</td>
      <td>1585.61</td>
      <td>1392.27</td>
      <td>14.62</td>
      <td>21.61</td>
      <td>555.21</td>
      <td>2388.00</td>
      <td>9064.57</td>
      <td>1.3</td>
      <td>...</td>
      <td>2388.08</td>
      <td>8146.56</td>
      <td>8.3878</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.96</td>
      <td>23.2951</td>
      <td>255</td>
    </tr>
    <tr>
      <th>4</th>
      <td>518.67</td>
      <td>641.68</td>
      <td>1588.63</td>
      <td>1397.65</td>
      <td>14.62</td>
      <td>21.61</td>
      <td>554.74</td>
      <td>2388.04</td>
      <td>9076.14</td>
      <td>1.3</td>
      <td>...</td>
      <td>2388.03</td>
      <td>8147.80</td>
      <td>8.3869</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>39.14</td>
      <td>23.4583</td>
      <td>254</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
plt.figure(figsize=(25,5))
plt.plot(np.arange(0,len(train)),train['RUL'])
plt.xlabel('Cycle',fontsize=14)
plt.ylabel('RUL',fontsize=14)
plt.title('Remaining useful life',fontsize=14)
plt.grid()
plt.show()
```


![png](output_7_0.png)



```python
plt.figure(figsize=(10,5))
plt.hist(train['RUL'],bins=50)
plt.xlabel('RUL')
plt.ylabel('frequency')
plt.show()

#histogrm of RUL looks like a log-normal distribution function
```


![png](output_8_0.png)



```python
# plotting RUL vs sensor signal data

for cols in train.columns[5:-1]:
    plt.figure(figsize=(8,5))
    plt.scatter(train[cols],train['RUL'])
    plt.xlabel(cols, fontsize=14)
    plt.ylabel('RUL')
    plt.show()
```


![png](output_9_0.png)



![png](output_9_1.png)



![png](output_9_2.png)



![png](output_9_3.png)



![png](output_9_4.png)



![png](output_9_5.png)



![png](output_9_6.png)



![png](output_9_7.png)



![png](output_9_8.png)



![png](output_9_9.png)



![png](output_9_10.png)



![png](output_9_11.png)



![png](output_9_12.png)



![png](output_9_13.png)



![png](output_9_14.png)



![png](output_9_15.png)



![png](output_9_16.png)



![png](output_9_17.png)



![png](output_9_18.png)



![png](output_9_19.png)



![png](output_9_20.png)



```python
plt.figure(figsize=(20,20))
sns.heatmap(train.corr().iloc[5:,5:], annot=True)

#There are some weak to moderate correlations between sensors and RUL
```




    <AxesSubplot:>




![png](output_10_1.png)



```python
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
```


```python
X_train = train.iloc[:,5:-1]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,5:-1]
y_test = test.iloc[:,-1]
```


```python
rf = RandomForestRegressor(max_features="sqrt", random_state=42)
rf.fit(X_train, y_train)

# predict and evaluate
y_hat_train = rf.predict(X_train)
evaluate(y_train, y_hat_train, 'train')

y_hat_test = rf.predict(X_test)
evaluate(y_test, y_hat_test)
```

    train set RMSE:21.075686924717434, R2:0.9545371233623917
    test set RMSE:103.18631193534473, R2:-0.535030314310083
    


```python
plt.figure(figsize=(10,10))
plt.scatter(y_test,y_hat_test,label='test')
plt.scatter(y_train,y_hat_train,label='train')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()
```


![png](output_14_0.png)



```python
plt.figure(figsize=(25,5))
plt.plot(np.arange(0,len(y_test)),y_test,label='actual RUL cycle')
plt.plot(np.arange(0,len(y_hat_test)),y_hat_test,label='predicted RUL cycle')
plt.xlabel('Cycle',fontsize=14)
plt.ylabel('RUL',fontsize=14)
plt.title('Remaining useful life prediction comparison',fontsize=14)
plt.grid()
plt.legend()
plt.show()
```


![png](output_15_0.png)


# Neural Survival analysis


```python
from pysurvival.models.multi_task import NeuralMultiTaskModel
from pysurvival.utils.display import display_loss_values
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score
from pysurvival.utils.display import compare_to_actual
from pysurvival.utils.display import create_risk_groups
from pysurvival.utils import save_model
from sklearn.preprocessing import normalize
```


```python
X_train_normalized = normalize(X_train)
X_test_normalized = normalize(X_test)
```


```python
#adding an event indicator column
#event indicator is 0 or 1. 
#1 indicates failure occured

train.loc[np.array(train[train['time_cycles']==1].index-1)[1:],'Event_ind'] = 1
train = train.fillna(0)
event_train = train.iloc[:,-1]
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
      <th>unit_number</th>
      <th>time_cycles</th>
      <th>setting_1</th>
      <th>setting_2</th>
      <th>setting_3</th>
      <th>s_1</th>
      <th>s_2</th>
      <th>s_3</th>
      <th>s_4</th>
      <th>s_5</th>
      <th>...</th>
      <th>s_14</th>
      <th>s_15</th>
      <th>s_16</th>
      <th>s_17</th>
      <th>s_18</th>
      <th>s_19</th>
      <th>s_20</th>
      <th>s_21</th>
      <th>RUL</th>
      <th>Event_ind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>-0.0005</td>
      <td>0.0004</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.36</td>
      <td>1583.23</td>
      <td>1396.84</td>
      <td>14.62</td>
      <td>...</td>
      <td>8145.32</td>
      <td>8.4246</td>
      <td>0.03</td>
      <td>391</td>
      <td>2388</td>
      <td>100.0</td>
      <td>39.11</td>
      <td>23.3537</td>
      <td>258</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.0008</td>
      <td>-0.0003</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.50</td>
      <td>1584.69</td>
      <td>1396.89</td>
      <td>14.62</td>
      <td>...</td>
      <td>8152.85</td>
      <td>8.4403</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.99</td>
      <td>23.4491</td>
      <td>257</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>-0.0014</td>
      <td>-0.0002</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.18</td>
      <td>1582.35</td>
      <td>1405.61</td>
      <td>14.62</td>
      <td>...</td>
      <td>8150.17</td>
      <td>8.3901</td>
      <td>0.03</td>
      <td>391</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.85</td>
      <td>23.3669</td>
      <td>256</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>-0.0020</td>
      <td>0.0001</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>642.92</td>
      <td>1585.61</td>
      <td>1392.27</td>
      <td>14.62</td>
      <td>...</td>
      <td>8146.56</td>
      <td>8.3878</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>38.96</td>
      <td>23.2951</td>
      <td>255</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0.0016</td>
      <td>0.0000</td>
      <td>100.0</td>
      <td>518.67</td>
      <td>641.68</td>
      <td>1588.63</td>
      <td>1397.65</td>
      <td>14.62</td>
      <td>...</td>
      <td>8147.80</td>
      <td>8.3869</td>
      <td>0.03</td>
      <td>392</td>
      <td>2388</td>
      <td>100.0</td>
      <td>39.14</td>
      <td>23.4583</td>
      <td>254</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
test.loc[np.array(test[test['time_cycles']==1].index-1)[1:],'Event_ind'] = 1
test = test.fillna(0)
event_test = test.iloc[:,-1]
```


```python
neural_mltr = NeuralMultiTaskModel(structure = [ {'activation': 'ReLU', 'num_units': 128}, ],bins=np.max(train['time_cycles']),auto_scaler=False)
neural_mltr.fit(X_train_normalized, y_train, event_train, init_method = 'glorot_uniform', optimizer ='adam',
    lr = 1e-3, num_epochs = 1000, dropout = 0.2, l2_reg=1e-2,
    l2_smooth=1e-2, batch_normalization=False, bn_and_dropout=False,
    verbose=True, extra_pct_time = 0.1, is_min_time_zero=True)
```

    % Completion: 100%|***********************************************|Loss: 650.02
    




    NeuralMultiTaskModel( Layer(1): activation = ReLU, units = 128 )




```python
display_loss_values(neural_mltr, figure_size=(10, 10))
```


![png](output_22_0.png)



```python
c_index = concordance_index(neural_mltr, X_test_normalized, y_test, event_test)
print('C-index: {:.2f}'.format(c_index))
```

    C-index: 0.74
    


```python
integrated_brier_score(neural_mltr, X_test_normalized, y_test, event_test, figure_size=(20, 6.5) )
```


![png](output_24_0.png)





    0.005935936831634108




```python
#compute risk score based on survival risk probabilities

risk = neural_mltr.predict_risk(X_test_normalized)
normalized_risk = (risk-min(risk))/(max(risk) - min(risk))
plt.figure(figsize=(20,5))
plt.scatter(y_test,normalized_risk*100)
#plt.xticks(np.arange(0, 474, 1.0))
plt.title('Predicted normalized risk factor by day',fontsize=14)
plt.xlabel('RUL days',fontsize=14)
plt.ylabel('Normalized risk score (%)',fontsize=14)
plt.grid()
plt.show()


#observe a downward trend
#As RUL decreases , the risk score increases. i.e. lower the remaining useful life, higher the risk score.
```


![png](output_25_0.png)



```python
test_risk = pd.concat([test,pd.DataFrame(normalized_risk,columns=['risk_score'])],axis=1)
```


```python
failure_index = np.array(test_risk[test_risk['Event_ind']==1].index)[0:10]
start = 0
plt.figure(figsize=(20,10))
for i in failure_index:  
    plt.plot(test_risk.loc[start:i,'time_cycles'],test_risk.loc[start:i,'risk_score'],label=test_risk.loc[start,'unit_number'])
    plt.scatter(test_risk.loc[i,'time_cycles'],test_risk.loc[i,'risk_score'],marker='*',s=100,c='r')
    start = i+1
plt.xlabel('time cycles',fontsize=14)
plt.ylabel('Normalized risk index',fontsize=14)
plt.grid()
plt.legend()
plt.title('Risk index trend by time cycles : Showing for 10 units',fontsize=14)
plt.show()
```


![png](output_27_0.png)


# conclusion

### We can use survival based analysis to compute a risk score, however, PySurvival package does not support prediction of RUL.

### Currently, we are using RUL days as our output labels directly, instead we need to use the neural network model to predict the shape of the survival density function. 

### Explore the usage of Deep time-to-failure model in https://github.com/gm-spacagna/deep-ttf This model uses gated recurrent units (GRU) to predict the shape of the survival function (modelled as weibull distribution) based on RUL. 


```python

```
