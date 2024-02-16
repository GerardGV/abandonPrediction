```python
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier


path="dataset.csv"
```

## DATA CLEANING

To detect automatically the delimiter or separator, it is specified that separator is equal to None to make python find it. See the explanation of sep parameter on:
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html


```python
dataset=pd.read_csv(path, sep=None, engine="python")
```


```python
dataset.head()
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
      <th>rev_Mean</th>
      <th>mou_Mean</th>
      <th>totmrc_Mean</th>
      <th>da_Mean</th>
      <th>ovrmou_Mean</th>
      <th>ovrrev_Mean</th>
      <th>vceovr_Mean</th>
      <th>datovr_Mean</th>
      <th>roam_Mean</th>
      <th>change_mou</th>
      <th>...</th>
      <th>forgntvl</th>
      <th>ethnic</th>
      <th>kid0_2</th>
      <th>kid3_5</th>
      <th>kid6_10</th>
      <th>kid11_15</th>
      <th>kid16_17</th>
      <th>creditcd</th>
      <th>eqpdays</th>
      <th>Customer_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23,9975</td>
      <td>219,25</td>
      <td>22,5</td>
      <td>0,2475</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-157,25</td>
      <td>...</td>
      <td>0.0</td>
      <td>N</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>Y</td>
      <td>361.0</td>
      <td>1000001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57,4925</td>
      <td>482,75</td>
      <td>37,425</td>
      <td>0,2475</td>
      <td>22,75</td>
      <td>9,1</td>
      <td>9,1</td>
      <td>0</td>
      <td>0</td>
      <td>532,25</td>
      <td>...</td>
      <td>0.0</td>
      <td>Z</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>Y</td>
      <td>240.0</td>
      <td>1000002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16,99</td>
      <td>10,25</td>
      <td>16,99</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-4,25</td>
      <td>...</td>
      <td>0.0</td>
      <td>N</td>
      <td>U</td>
      <td>Y</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>Y</td>
      <td>1504.0</td>
      <td>1000003</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>7,5</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1,5</td>
      <td>...</td>
      <td>0.0</td>
      <td>U</td>
      <td>Y</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>Y</td>
      <td>1812.0</td>
      <td>1000004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55,23</td>
      <td>570,5</td>
      <td>71,98</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>38,5</td>
      <td>...</td>
      <td>0.0</td>
      <td>I</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>U</td>
      <td>Y</td>
      <td>434.0</td>
      <td>1000005</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 100 columns</p>
</div>



Balance in the objective variable:


```python
dataset['churn'].value_counts()
```




    0    50438
    1    49562
    Name: churn, dtype: int64




```python
sns.countplot(x='churn', data=dataset, palette='hls')
plt.show()
```


![png](img/output_7_0.png)
​    


Now, It is necessary to analyse if there is any NaN value. For see the number of NaN values for each attribute is necessary to extend the maximum number of rows that pandas let to show. It is used "with" to do it only for this code block.


```python
print("Dataset size:", dataset.shape)

#Modifying the maximum number of rows that pandas let to show
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   print(dataset.isnull().sum())
```

    Dataset size: (100000, 100)
    rev_Mean              357
    mou_Mean              357
    totmrc_Mean           357
    da_Mean               357
    ovrmou_Mean           357
    ovrrev_Mean           357
    vceovr_Mean           357
    datovr_Mean           357
    roam_Mean             357
    change_mou            891
    change_rev            891
    drop_vce_Mean           0
    drop_dat_Mean           0
    blck_vce_Mean           0
    blck_dat_Mean           0
    unan_vce_Mean           0
    unan_dat_Mean           0
    plcd_vce_Mean           0
    plcd_dat_Mean           0
    recv_vce_Mean           0
    recv_sms_Mean           0
    comp_vce_Mean           0
    comp_dat_Mean           0
    custcare_Mean           0
    ccrndmou_Mean           0
    cc_mou_Mean             0
    inonemin_Mean           0
    threeway_Mean           0
    mou_cvce_Mean           0
    mou_cdat_Mean           0
    mou_rvce_Mean           0
    owylis_vce_Mean         0
    mouowylisv_Mean         0
    iwylis_vce_Mean         0
    mouiwylisv_Mean         0
    peak_vce_Mean           0
    peak_dat_Mean           0
    mou_peav_Mean           0
    mou_pead_Mean           0
    opk_vce_Mean            0
    opk_dat_Mean            0
    mou_opkv_Mean           0
    mou_opkd_Mean           0
    drop_blk_Mean           0
    attempt_Mean            0
    complete_Mean           0
    callfwdv_Mean           0
    callwait_Mean           0
    churn                   0
    months                  0
    uniqsubs                0
    actvsubs                0
    new_cell                0
    crclscod                0
    asl_flag                0
    totcalls                0
    totmou                  0
    totrev                  0
    adjrev                  0
    adjmou                  0
    adjqty                  0
    avgrev                  0
    avgmou                  0
    avgqty                  0
    avg3mou                 0
    avg3qty                 0
    avg3rev                 0
    avg6mou              2839
    avg6qty              2839
    avg6rev              2839
    prizm_social_one     7388
    area                   40
    dualband                1
    refurb_new              1
    hnd_price             847
    phones                  1
    models                  1
    hnd_webcap          10189
    truck                1732
    rv                   1732
    ownrent             33706
    lor                 30190
    dwlltype            31909
    marital              1732
    adults              23019
    infobase            22079
    income              25436
    numbcars            49366
    HHstatin            37923
    dwllsize            38308
    forgntvl             1732
    ethnic               1732
    kid0_2               1732
    kid3_5               1732
    kid6_10              1732
    kid11_15             1732
    kid16_17             1732
    creditcd             1732
    eqpdays                 1
    Customer_ID             0
    dtype: int64


### DROPS

Attributes dropped:
- ethnic: is not ethical to keep this attribute
- numbcars: almost half of the dataset doesn't have this attribute
- Customer_ID: it has not variance because is unique for each customer
- ownrent, lor, dwlltype, HHstatin, dwllsize: They are NaN on 30% of the dataset


```python
datasetClean=dataset.drop(columns=["ethnic", "numbcars", "Customer_ID", "ownrent", "lor", "dwlltype", "HHstatin", "dwllsize"])
print("Dataset size:", datasetClean.shape)
```

    Dataset size: (100000, 92)


In order to code categorical attributes, object attributes need to be transform.


```python
datasetClean.dtypes
```




    rev_Mean        object
    mou_Mean        object
    totmrc_Mean     object
    da_Mean         object
    ovrmou_Mean     object
                    ...   
    kid6_10         object
    kid11_15        object
    kid16_17        object
    creditcd        object
    eqpdays        float64
    Length: 92, dtype: object




```python
datasetClean.select_dtypes(include='object').columns
```




    Index(['rev_Mean', 'mou_Mean', 'totmrc_Mean', 'da_Mean', 'ovrmou_Mean',
           'ovrrev_Mean', 'vceovr_Mean', 'datovr_Mean', 'roam_Mean', 'change_mou',
           'change_rev', 'drop_vce_Mean', 'drop_dat_Mean', 'blck_vce_Mean',
           'blck_dat_Mean', 'unan_vce_Mean', 'unan_dat_Mean', 'plcd_vce_Mean',
           'plcd_dat_Mean', 'recv_vce_Mean', 'recv_sms_Mean', 'comp_vce_Mean',
           'comp_dat_Mean', 'custcare_Mean', 'ccrndmou_Mean', 'cc_mou_Mean',
           'inonemin_Mean', 'threeway_Mean', 'mou_cvce_Mean', 'mou_cdat_Mean',
           'mou_rvce_Mean', 'owylis_vce_Mean', 'mouowylisv_Mean',
           'iwylis_vce_Mean', 'mouiwylisv_Mean', 'peak_vce_Mean', 'peak_dat_Mean',
           'mou_peav_Mean', 'mou_pead_Mean', 'opk_vce_Mean', 'opk_dat_Mean',
           'mou_opkv_Mean', 'mou_opkd_Mean', 'drop_blk_Mean', 'attempt_Mean',
           'complete_Mean', 'callfwdv_Mean', 'callwait_Mean', 'new_cell',
           'crclscod', 'asl_flag', 'totmou', 'totrev', 'adjrev', 'adjmou',
           'avgrev', 'avgmou', 'avgqty', 'prizm_social_one', 'area', 'dualband',
           'refurb_new', 'hnd_price', 'hnd_webcap', 'marital', 'infobase',
           'kid0_2', 'kid3_5', 'kid6_10', 'kid11_15', 'kid16_17', 'creditcd'],
          dtype='object')



Object attributes which are numbers with "," are pass to floats and object with only numbers and more than 2 different numbers are convert to integer.


```python
objectAttributesList = datasetClean.select_dtypes(include='object').columns

for column in objectAttributesList:
   # Looking through all the column to see if there are one string with a number

   if datasetClean[column].str.contains('\d,').any():
      # Casting column from string with "," to float with "."
      datasetClean[column] = datasetClean[column].str.replace(',', '.').astype(float)

   elif datasetClean[column].str.contains('\d+').all() and len(datasetClean[column].unique()) > 2:
      #If the values in the column are numbers and there are not binary, they are not categorical
      datasetClean[column] = datasetClean[column].astype(int)
```

Now, only remains attributes which are categorical, binaries and labels


```python
datasetClean.select_dtypes(include='object').columns
```




    Index(['new_cell', 'crclscod', 'asl_flag', 'prizm_social_one', 'area',
           'dualband', 'refurb_new', 'hnd_webcap', 'marital', 'infobase', 'kid0_2',
           'kid3_5', 'kid6_10', 'kid11_15', 'kid16_17', 'creditcd'],
          dtype='object')



A bit more data exploration. The objective of this part is to detect attributes where the difference between the number of people churning is significantly larger or smaller than those not churning.

Relation between continuous attributes and churning:


```python
datasetClean.groupby('churn').mean()
```

    /var/folders/4c/wcpzkb9d3y79cvx0_vlxmk940000gn/T/ipykernel_54256/1272408106.py:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      datasetClean.groupby('churn').mean()





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
      <th>rev_Mean</th>
      <th>mou_Mean</th>
      <th>totmrc_Mean</th>
      <th>da_Mean</th>
      <th>ovrmou_Mean</th>
      <th>ovrrev_Mean</th>
      <th>vceovr_Mean</th>
      <th>datovr_Mean</th>
      <th>roam_Mean</th>
      <th>change_mou</th>
      <th>...</th>
      <th>avg6rev</th>
      <th>hnd_price</th>
      <th>phones</th>
      <th>models</th>
      <th>truck</th>
      <th>rv</th>
      <th>adults</th>
      <th>income</th>
      <th>forgntvl</th>
      <th>eqpdays</th>
    </tr>
    <tr>
      <th>churn</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.218692</td>
      <td>543.206895</td>
      <td>47.782378</td>
      <td>0.918039</td>
      <td>39.172904</td>
      <td>12.842879</td>
      <td>12.573835</td>
      <td>0.265309</td>
      <td>1.150619</td>
      <td>-5.344265</td>
      <td>...</td>
      <td>59.445354</td>
      <td>108.129344</td>
      <td>1.838511</td>
      <td>1.585959</td>
      <td>0.190411</td>
      <td>0.082447</td>
      <td>2.541654</td>
      <td>5.771879</td>
      <td>0.059130</td>
      <td>363.280925</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58.211074</td>
      <td>483.306417</td>
      <td>44.543091</td>
      <td>0.859019</td>
      <td>43.010449</td>
      <td>14.290904</td>
      <td>14.031045</td>
      <td>0.257244</td>
      <td>1.424969</td>
      <td>-22.759003</td>
      <td>...</td>
      <td>57.916832</td>
      <td>95.539523</td>
      <td>1.734817</td>
      <td>1.504984</td>
      <td>0.187204</td>
      <td>0.082716</td>
      <td>2.518496</td>
      <td>5.794841</td>
      <td>0.056799</td>
      <td>421.089524</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 75 columns</p>
</div>



There isn't any significant difference or gap. Let's explore some potentially interesting categorical labels.


```python
pd.crosstab(datasetClean['creditcd'],datasetClean['churn']).plot(kind='bar')
plt.title('Credit by chrun')
plt.xlabel('creditcd')
plt.ylabel('churn')
```




    Text(0, 0.5, 'churn')




![png](img/output_23_1.png)
    



```python
pd.crosstab(datasetClean['marital'],datasetClean['churn']).plot(kind='bar')
plt.title('marital by churn')
plt.xlabel('marital')
plt.ylabel('churn')
```




    Text(0, 0.5, 'churn')




![png](img/output_24_1.png)
​    



```python
pd.crosstab(datasetClean['area'],datasetClean['churn']).plot(kind='bar')
plt.title('area by churn')
plt.xlabel('area')
plt.ylabel('churn')
```




    Text(0, 0.5, 'churn')




![png](img/output_25_1.png)
​    



```python
pd.crosstab(datasetClean['prizm_social_one'],datasetClean['churn']).plot(kind='bar')
plt.title('prizm_social_one by churn')
plt.xlabel('prizm_social_one')
plt.ylabel('churn')
```




    Text(0, 0.5, 'churn')




​    
![png](img/output_26_1.png)
​    



```python
pd.crosstab(datasetClean['hnd_webcap'],datasetClean['churn']).plot(kind='bar')
plt.title('hnd_webcap by churn')
plt.xlabel('hnd_webcap')
plt.ylabel('churn')
```




    Text(0, 0.5, 'churn')




​    
![png](img/output_27_1.png)
​    


All the labels in the different categorical attributes explored has more or less the same amount of churning and not churning people. Thus, seems to not be any significant relation.


2 test cases exploration:
- Dropping all the samples with NaN values.
- Data insertion to substitute NaN values. Categorical attributes will use mode value and numerical median value.

## DROPPING NAN'S



```python
#dropping all the samples with NaN
dfDropNan=datasetClean.dropna()
print(dfDropNan.shape)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):

   print(dfDropNan.isnull().sum())
```

    (60385, 92)
    rev_Mean            0
    mou_Mean            0
    totmrc_Mean         0
    da_Mean             0
    ovrmou_Mean         0
    ovrrev_Mean         0
    vceovr_Mean         0
    datovr_Mean         0
    roam_Mean           0
    change_mou          0
    change_rev          0
    drop_vce_Mean       0
    drop_dat_Mean       0
    blck_vce_Mean       0
    blck_dat_Mean       0
    unan_vce_Mean       0
    unan_dat_Mean       0
    plcd_vce_Mean       0
    plcd_dat_Mean       0
    recv_vce_Mean       0
    recv_sms_Mean       0
    comp_vce_Mean       0
    comp_dat_Mean       0
    custcare_Mean       0
    ccrndmou_Mean       0
    cc_mou_Mean         0
    inonemin_Mean       0
    threeway_Mean       0
    mou_cvce_Mean       0
    mou_cdat_Mean       0
    mou_rvce_Mean       0
    owylis_vce_Mean     0
    mouowylisv_Mean     0
    iwylis_vce_Mean     0
    mouiwylisv_Mean     0
    peak_vce_Mean       0
    peak_dat_Mean       0
    mou_peav_Mean       0
    mou_pead_Mean       0
    opk_vce_Mean        0
    opk_dat_Mean        0
    mou_opkv_Mean       0
    mou_opkd_Mean       0
    drop_blk_Mean       0
    attempt_Mean        0
    complete_Mean       0
    callfwdv_Mean       0
    callwait_Mean       0
    churn               0
    months              0
    uniqsubs            0
    actvsubs            0
    new_cell            0
    crclscod            0
    asl_flag            0
    totcalls            0
    totmou              0
    totrev              0
    adjrev              0
    adjmou              0
    adjqty              0
    avgrev              0
    avgmou              0
    avgqty              0
    avg3mou             0
    avg3qty             0
    avg3rev             0
    avg6mou             0
    avg6qty             0
    avg6rev             0
    prizm_social_one    0
    area                0
    dualband            0
    refurb_new          0
    hnd_price           0
    phones              0
    models              0
    hnd_webcap          0
    truck               0
    rv                  0
    marital             0
    adults              0
    infobase            0
    income              0
    forgntvl            0
    kid0_2              0
    kid3_5              0
    kid6_10             0
    kid11_15            0
    kid16_17            0
    creditcd            0
    eqpdays             0
    dtype: int64


## INSERTION OF NUMBERS


```python
dfNanSubstituted=datasetClean.copy()

#filling NaN in object categorical attributes
columnCategorical=dfNanSubstituted.select_dtypes(include='object').columns
for column in columnCategorical:
   #Filling with the most common categorical value, the mode
   dfNanSubstituted[column].fillna(dfNanSubstituted[column].mode()[0], inplace=True)

#fillin NaN in not numerical object attributes
notNumericalColumn=dfNanSubstituted.select_dtypes(exclude='object').columns
for column in notNumericalColumn:
   #Filling with the most common categorical value, the mode
   dfNanSubstituted[column].fillna(dfNanSubstituted[column].median(), inplace=True)

#Modifying the maximum number of rows that pandas let to show
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   print(dfNanSubstituted.isnull().sum())
```

    rev_Mean            0
    mou_Mean            0
    totmrc_Mean         0
    da_Mean             0
    ovrmou_Mean         0
    ovrrev_Mean         0
    vceovr_Mean         0
    datovr_Mean         0
    roam_Mean           0
    change_mou          0
    change_rev          0
    drop_vce_Mean       0
    drop_dat_Mean       0
    blck_vce_Mean       0
    blck_dat_Mean       0
    unan_vce_Mean       0
    unan_dat_Mean       0
    plcd_vce_Mean       0
    plcd_dat_Mean       0
    recv_vce_Mean       0
    recv_sms_Mean       0
    comp_vce_Mean       0
    comp_dat_Mean       0
    custcare_Mean       0
    ccrndmou_Mean       0
    cc_mou_Mean         0
    inonemin_Mean       0
    threeway_Mean       0
    mou_cvce_Mean       0
    mou_cdat_Mean       0
    mou_rvce_Mean       0
    owylis_vce_Mean     0
    mouowylisv_Mean     0
    iwylis_vce_Mean     0
    mouiwylisv_Mean     0
    peak_vce_Mean       0
    peak_dat_Mean       0
    mou_peav_Mean       0
    mou_pead_Mean       0
    opk_vce_Mean        0
    opk_dat_Mean        0
    mou_opkv_Mean       0
    mou_opkd_Mean       0
    drop_blk_Mean       0
    attempt_Mean        0
    complete_Mean       0
    callfwdv_Mean       0
    callwait_Mean       0
    churn               0
    months              0
    uniqsubs            0
    actvsubs            0
    new_cell            0
    crclscod            0
    asl_flag            0
    totcalls            0
    totmou              0
    totrev              0
    adjrev              0
    adjmou              0
    adjqty              0
    avgrev              0
    avgmou              0
    avgqty              0
    avg3mou             0
    avg3qty             0
    avg3rev             0
    avg6mou             0
    avg6qty             0
    avg6rev             0
    prizm_social_one    0
    area                0
    dualband            0
    refurb_new          0
    hnd_price           0
    phones              0
    models              0
    hnd_webcap          0
    truck               0
    rv                  0
    marital             0
    adults              0
    infobase            0
    income              0
    forgntvl            0
    kid0_2              0
    kid3_5              0
    kid6_10             0
    kid11_15            0
    kid16_17            0
    creditcd            0
    eqpdays             0
    dtype: int64


## ATTRIBUTE CATEGORICAL CODIFICATION

Attributes object that remains are categorical. It is printed the name of the attribute and the number of unique values in each attribute. Also, categorical attributes are encoded.

Coding dataset with data insertion:


```python
objectAttributesList = datasetClean.select_dtypes(include='object').columns
categoricalAttributes=[]
categoricalAttributes2=[]
for column in objectAttributesList:

   #categorical
   if len(dfNanSubstituted[column].unique()) > 2:
      oneshot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

      #for data insertion test case
      cadena=dfNanSubstituted[column].to_numpy().reshape(-1, 1)
      dataNewColumns=oneshot_encoder.fit_transform(cadena)


      #creation of a list with new names for new coded column
      #The for loop is iterating inside the categories created by the encoder
      categories=oneshot_encoder.categories_[0]
      newColumnNameList=[f'{column}_{cat}' for cat in categories]

      #saving names of categorical columns to calculate correlations after
      categoricalAttributes=categoricalAttributes + newColumnNameList

      newColumns=pd.DataFrame(dataNewColumns, columns=newColumnNameList)

      #adding new columns and dropping categorical column that has been encoded
      dfNanSubstituted=dfNanSubstituted.join(newColumns)
      dfNanSubstituted.drop(columns=column, inplace=True)

      #CODING WITH THE SAME ENCODER DATASET WITH NAN'S DROPPED
      #--------------------------------------------------------------------------------------------

       #for data insertion test case
      cadena2=dfDropNan[column].to_numpy().reshape(-1, 1)
      dataNewColumns2=oneshot_encoder.fit_transform(cadena2)


      #creation of a list with new names for new coded column
      #The for loop is iterating inside the categories created by the encoder
      categories2=oneshot_encoder.categories_[0]
      newColumnNameList2=[f'{column}_{cat}' for cat in categories2]


      #saving names of categorical columns to calculate correlations after
      categoricalAttributes2=categoricalAttributes2 + newColumnNameList2



      newColumns2=pd.DataFrame(dataNewColumns2, columns=newColumnNameList2)

      #if the index are NOT reset, it will generate rows with nan values
      #because index from dataset and new columns doesn't match
      dfDropNan.reset_index(drop=True, inplace=True)
      newColumns2.reset_index(drop=True, inplace=True)

      #adding new columns and dropping categorical column that has been encoded
      dfDropNan=dfDropNan.join(newColumns2)
      dfDropNan.drop(columns=column, inplace=True)


   else:
      #binary categorical encode
      encoder = LabelEncoder()
      dfNanSubstituted[column]=encoder.fit_transform(dfNanSubstituted[column])
      #saving names of categorical columns to calculate correlations after
      categoricalAttributes.append(column)

       #CODING WITH THE SAME ENCODER DATASET WITH NAN'S DROPPED
      #--------------------------------------------------------------------------------------------

      dfDropNan[column]=encoder.fit_transform(dfDropNan[column])

      #saving names of categorical columns to calculate correlations after
      categoricalAttributes2.append(column)


```

Cases that are lost if Nan are dropped:


```python
# Converting attributes lists into sets
set1 = set(categoricalAttributes)
set2 = set(categoricalAttributes2)

# Finding differences
elementos_diferentes = set1.symmetric_difference(set2)

print("Elementos diferentes:", list(elementos_diferentes))

```

    Elementos diferentes: ['crclscod_IF', 'dualband_U', 'hnd_webcap_UNKW', 'crclscod_S', 'crclscod_ZF']


## ATTRIBUTES SELECTION

### CORRELATIONS

Correlations between categorical attributes and categorical attribute churn:


```python
corrCategorical={}
for attribute in categoricalAttributes:
   correlacion, p_valor = pointbiserialr(dfNanSubstituted[attribute], dfNanSubstituted['churn'])
   corrCategorical[attribute]=correlacion
   #print(f"Correlación Punto Biserial para {attribute}: {correlacion:.4f}, p-valor: {p_valor:.4f}")

```


```python
correlationMatrix=pd.DataFrame({"churn":corrCategorical.values()}, index=corrCategorical.keys())

plt.figure(figsize=(10, 20))
sns.heatmap(correlationMatrix, annot=True, fmt="g", cmap='coolwarm')

plt.show()
```


​    
![png](img/output_41_0.png)
​    




Correlation between continuous attributes and churn(categorical):


```python
def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return np.sqrt(stat/(obs*mini))
```


```python
relevantAttributes={}
nanSubs_X=dfNanSubstituted.drop(columns='churn')
nanSubs_Y=dfNanSubstituted['churn']

thresholdCorrelation=0.1

for column in nanSubs_X.columns:

   #we only calculate cramer_V of not categorical attributes
   if column not in categoricalAttributes:
      corr=cramers_V(nanSubs_X[column], nanSubs_Y)

      if abs(corr) > thresholdCorrelation:
         relevantAttributes[column]=corr
         #print(column, corr)
```


```python
correlationMatrix=pd.DataFrame({"churn":relevantAttributes.values()}, index=relevantAttributes.keys())

plt.figure(figsize=(10, 10))
sns.heatmap(correlationMatrix, annot=True, fmt="g", cmap='coolwarm')

plt.show()

```


​    
![png](img/output_46_0.png)
​    


# MODEL SELECTION

Now we try different models for the 2 test cases.

DATASET WITH DATA INSERTIONS:


```python
attr_dataInsert=list(relevantAttributes.keys())
X_dataInsert=dfNanSubstituted[attr_dataInsert]#.drop(columns=['comp_vce_Mean'])
Y_dataInsert=dfNanSubstituted['churn']

X_train_dataInsert, X_test_dataInsert, Y_train_dataInsert, Y_test_dataInsert = train_test_split(X_dataInsert, Y_dataInsert, test_size=0.8, random_state=42)
```

DATASET WITH NAN'S DROPPED:


```python
attr_dropNan=list(relevantAttributes.keys())
dropNan_X=dfDropNan[attr_dropNan]
dropNan_Y=dfDropNan['churn']

X_train_dropNan, X_test_dropNan, Y_train_dropNan, Y_test_dropNan = train_test_split(dropNan_X, dropNan_Y, test_size=0.8, random_state=42)
```

### LOGISTIC REGRESSION

DATASET WITH DATA INSERTIONS:


```python
scaler = StandardScaler()
X_train_scaled_dataInsert = scaler.fit_transform(X_train_dataInsert)
X_test_scaled_dataInsert = scaler.transform(X_test_dataInsert)
model = LogisticRegression()
model.fit(X_train_scaled_dataInsert,Y_train_dataInsert)
y_pred_dataInsert = model.predict(X_test_scaled_dataInsert)
#Calculating recall, true positives
recall_dataInsert = recall_score(Y_test_dataInsert, y_pred_dataInsert)
print(recall_dataInsert)
```

    0.570696876419236


    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


DATASET WITH NAN'S DROPPED:


```python
scaler = StandardScaler()
X_train_scaled_dropNan = scaler.fit_transform(X_train_dropNan)
X_test_scaled = scaler.transform(X_test_dropNan)
model = LogisticRegression()
model.fit(X_train_scaled_dropNan, Y_train_dropNan)
y_pred_dropNan = model.predict(X_test_scaled)
#Calculating recall, true positives
recall_dropNan = recall_score(Y_test_dropNan, y_pred_dropNan)
print(recall_dropNan)
```

    0.47098591549295776


    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(


### DECISION TREE

DATASET WITH DATA INSERTIONS:


```python
tree_model_dataInsert = DecisionTreeClassifier(random_state=42)
tree_model_dataInsert.fit(X_train_dataInsert, Y_train_dataInsert)
y_pred_dataInsert = tree_model_dataInsert.predict(X_test_dataInsert)
recall_dataInsert = recall_score(Y_test_dataInsert, y_pred_dataInsert)
print(recall_dataInsert)
```

    0.5429681586516627


DATASET WITH NAN'S DROPPED:


```python
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_dropNan, Y_train_dropNan)
y_pred_dropNan = tree_model.predict(X_test_dropNan)
recall_dropNan = recall_score(Y_test_dropNan, y_pred_dropNan)
print(recall_dropNan)
```

    0.5176164680390033


### RANDOME FOREST
- **Assembly type:** Bagging.
- **Basis of the algorithm:** Decision trees.
- **Process:** Create multiple decision trees and combine their predictions through voting. The idea is to reduce variance and avoid overfitting.

DATASET WITH DATA INSERTIONS:


```python
# Creation and training
rf_classifier_dataInsert = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_dataInsert.fit(X_train_dataInsert, Y_train_dataInsert)


y_pred_dataInsert = rf_classifier_dataInsert.predict(X_test_dataInsert)

recall_dataInsert = recall_score(Y_test_dataInsert, y_pred_dataInsert)
print(recall_dataInsert)

```

    0.588888328203058


DATASET WITH NAN'S DROPPED:


```python
rf_classifier2_dropNan = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier2_dropNan.fit(X_train_dropNan, Y_train_dropNan)

y_pred_dropNan = rf_classifier2_dropNan.predict(X_test_dropNan)

recall_dropNan = recall_score(Y_test_dropNan, y_pred_dropNan)
print(recall_dropNan)
```

    0.5294474539544962


## ASSEMBLY ALGORITHMS

The current recall result is better with the test case involving data insertion. Additionally, the highest recall score has been achieved using Random Forest. Following these results, various ensemble algorithms will be explored, apart from Random Forest, using the dataset split that includes data insertion.

### ADA BOOST CLASSIFIER
- **Assembly type:** Boosting.
- **Basis of the algorithm:** Weak decision trees (generally shallow trees).
- **Process:** Iteratively trains weak models, assigning more weight to poorly classified instances in each iteration. Thus, it focuses on difficult cases and improves the performance of the model.


```python
ada_classifier_dataInsert = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_classifier_dataInsert.fit(X_train_dataInsert, Y_train_dataInsert)

y_pred_dataInsert = ada_classifier_dataInsert.predict(X_test_dataInsert)

recall_dataInsert = recall_score(Y_test_dataInsert, y_pred_dataInsert)
print(recall_dataInsert)
```

    0.6154059645758692


### GRADIENT BOOSTING CLASSIFIER
- **Assembly type:** Boosting.
- **Basis of the algorithm:** Decision trees (usually shallow trees).
- **Process:** Unlike AdaBoost, Gradient Boosting optimizes the model by directly minimizing the loss function. Each tree fits the residuals (difference between predictions and actual labels) of the existing model.


```python
gb_classifier_dataInsert = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier_dataInsert.fit(X_train_dataInsert, Y_train_dataInsert)
y_pred_dataInsert = gb_classifier_dataInsert.predict(X_test_dataInsert)

recall_dataInsert = recall_score(Y_test_dataInsert, y_pred_dataInsert)
print(recall_dataInsert)
```

    0.6400565171317556


### EXTRA TREES CLASSIFIER
- **Assembly type:** Bagging.
- **Basis of the algorithm:** Decision trees.
- **Process:** Similar to RandomForest, but with additional randomization. Instead of searching for the best threshold to split features, ExtraTrees randomly selects thresholds.


```python
et_classifier_dataInsert = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_classifier_dataInsert.fit(X_train_dataInsert, Y_train_dataInsert)
y_pred_dataInsert = et_classifier_dataInsert.predict(X_test_dataInsert)

recall_dataInsert = recall_score(Y_test_dataInsert, y_pred_dataInsert)
print(recall_dataInsert)
```

    0.565776858253015


# MODEL EVALUATION

Gradient Boosting Classifier is the best option as it has a bigger recall score.

Now, an evaluation of the importance of each feature from the Gradient Boosting Classifier point of view:


```python
#creatin  a list with the importance assigned by Gradient Boosting Classifier
importance=gb_classifier_dataInsert.feature_importances_

#creating a dictionary with the name and the importance of each attribute used to train
dictImportance=dict(zip(attr_dataInsert, importance))

# sorting the dictionary in ascending order
dictImportanceSorted = dict(sorted(dictImportance.items(), key=lambda item: item[1], reverse=True))

#list to save features or attributes with less importance
notImportant=[]
important=[]
thresholdImportance=0.025
print("Important features:")
for key in dictImportanceSorted.keys():

   if dictImportanceSorted[key] < thresholdImportance:
      notImportant.append(key)
   else:
      important.append(key)
      print(key,":",dictImportanceSorted[key])

```

    Important features:
    eqpdays : 0.2123653442618351
    months : 0.1614054803279476
    change_mou : 0.06670277398661263
    mou_Mean : 0.06559430888381787
    totmrc_Mean : 0.06491532780723433
    avgqty : 0.038356727465243275
    hnd_price : 0.03591513920891923
    mou_cvce_Mean : 0.034611498870405266
    avgrev : 0.03160477157990883
    change_rev : 0.030925301696366213
    cc_mou_Mean : 0.02672055575942063


Less important attributes:


```python
print(notImportant)
```

    ['avgmou', 'avg3mou', 'vceovr_Mean', 'ovrmou_Mean', 'mou_rvce_Mean', 'totrev', 'rev_Mean', 'ovrrev_Mean', 'adjrev', 'mouiwylisv_Mean', 'totmou', 'avg6mou', 'mou_opkv_Mean', 'mou_peav_Mean', 'opk_vce_Mean', 'roam_Mean', 'adjqty', 'plcd_vce_Mean', 'inonemin_Mean', 'peak_vce_Mean', 'owylis_vce_Mean', 'avg6qty', 'attempt_Mean', 'totcalls', 'mouowylisv_Mean', 'recv_vce_Mean', 'mou_cdat_Mean', 'unan_vce_Mean', 'adjmou', 'avg3qty', 'mou_opkd_Mean', 'mou_pead_Mean', 'complete_Mean', 'comp_vce_Mean']


Trying new training with a new data split without less important attributes.


```python
attr_dropNan=list(relevantAttributes.keys())
dropNan_X=dfDropNan[attr_dropNan].drop(columns=notImportant)
dropNan_Y=dfDropNan['churn']

X_train_dropNan, X_test_dropNan, Y_train_dropNan, Y_test_dropNan = train_test_split(dropNan_X, dropNan_Y, test_size=0.8, random_state=42)

gb_classifier_dataInsert = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier_dataInsert.fit(X_train_dataInsert, Y_train_dataInsert)
y_pred_dataInsert = gb_classifier_dataInsert.predict(X_test_dataInsert)

recall_dataInsert = recall_score(Y_test_dataInsert, y_pred_dataInsert)
print("Recall: " + str(recall_dataInsert))
print("Most import features:",dropNan_X.columns)
```

    Recall: 0.6400565171317556
    Most import features: Index(['mou_Mean', 'totmrc_Mean', 'change_mou', 'change_rev', 'cc_mou_Mean',
           'mou_cvce_Mean', 'months', 'avgrev', 'avgqty', 'hnd_price', 'eqpdays'],
          dtype='object')



```python
logit_roc_auc = roc_auc_score(Y_test_dataInsert, gb_classifier_dataInsert.predict(X_test_dataInsert))
fpr, tpr, thresholds = roc_curve(Y_test_dataInsert, gb_classifier_dataInsert.predict_proba(X_test_dataInsert)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()
```


​    
![png](img/output_79_0.png)
​    



```python
# adjusting figure size
sns.set(rc={'figure.figsize':(20,12)}) # width=10, height=6

# creating distribution plot
ax=sns.histplot(dataset, x="eqpdays", hue="churn")

#plt.savefig('distribution2.png')
```


​    
![png](img/output_80_0.png)
​    


Starting from the highest peak in the distribution of the variable 'eqpdays,' it is evident that the number of people who churn is consistently larger than those who do not. Getting the range of days of the container in the distribution when starts to be more people churning than not churning.


```python
indexMax=np.argmax(ax.containers[0].datavalues)
rangeValues=[ax.containers[0][indexMax].xy[0], ax.containers[0][indexMax+1].xy[0]]
print(rangeValues)
```

    [308.76119402985074, 322.4029850746269]



```python
sns.set(rc={'figure.figsize':(10,6)}) # width=10, height=6

ax=sns.histplot(dataset, x="months", hue="churn")
```


​    
![png](img/output_83_0.png)
​    



```python
indexMax=np.argmax(ax.containers[0].datavalues)
rangeValues=[ax.containers[0][indexMax].xy[0], ax.containers[0][indexMax+1].xy[0]]
print(rangeValues)
```

    [11.0, 11.555555555555555]


# CONCLUSION

Best results obtained doing data insertion, using Gradient Boosting Classifier. The most important feature to predict churning is the Number of days (age) of current equipment(eqpdays) with a 26% of importance base in Gradient boosting Classifier. The second most important (16%) feature is Total number of months in service(months).

*IMPORTANT*
These results arise from contracts with telecom companies in Spain. Typically, telecom companies offer contracts where clients pay a low price for a set period of time for their services. After this period, prices increase, leading to client churn in telecom services. This period typically spans 11 months, equivalent to 330 days (11 months x 30 days per month). This aligns with previous results, where, starting from the 11th month in the 'months' attribute and 308 days in 'eqpdays,' client churn increased.


Important features:
eqpdays : 21.24% (Number of days (age) of current equipment)
months : 16.14% (Total number of months in service)
change_mou : 6.67% (Percentage change in monthly minutes of use vs the previous three-month average)
mou_Mean : 6.56% (Mean number of monthly minutes of use)
totmrc_Mean : 6.49% (Mean total monthly recurring charge)
avgqty : 3.84% (Average monthly number of calls over the life of the customer)
hnd_price : 3.59% (Current handset price)
mou_cvce_Mean : 3.46% (Mean unrounded minutes of use of completed voice calls)
avgrev : 3.16% (Average monthly revenue over the life of the customer)
change_rev : 3.09% (Percentage change in monthly revenue vs the previous three-month average)
cc_mou_Mean : 2.67% (Mean unrounded minutes of use of customer care calls)


It seems that the most important features for churning are related to the topic of the equipment, minutes of calls and the revenue from the client. As it is possible to see below, there are increasing correlations between this features and churning.


```python
relevantAttributes={}
nanSubs_X=dfNanSubstituted[important]
nanSubs_Y=dfNanSubstituted['churn']

thresholdCorrelation=0.1

for column in nanSubs_X.columns:

   #we only calculate cramer_V of not categorical attributes
   if column not in categoricalAttributes:
      corr=cramers_V(nanSubs_X[column], nanSubs_Y)

      if corr > thresholdCorrelation:
         relevantAttributes[column]=corr

```


```python
correlationMatrix=pd.DataFrame({"churn":relevantAttributes.values()}, index=relevantAttributes.keys())

plt.figure(figsize=(5, 5))
sns.heatmap(correlationMatrix, annot=True, fmt="g", cmap='coolwarm')

plt.show()
```


​    
![png](img/output_87_0.png)
​    



```python

```
