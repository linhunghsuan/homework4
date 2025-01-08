import numpy as np
import pandas as pd
from pycaret.classification import *
from sklearn.model_selection import train_test_split

# 1. 數據讀取與處理
data_train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Titanic-Dataset.csv')
data_test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Titanic-Dataset.csv')

# 數據處理函數
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

# 處理訓練與測試數據
data_train = transform_features(data_train)
data_test = transform_features(data_test)

# 2. 切分訓練數據和測試數據
train_data, test_data = train_test_split(data_train.drop(['PassengerId'], axis=1), random_state=100, train_size=0.8)

# 3. 設置 PyCaret 環境
clf1 = setup(data = train_data, 
             target = 'Survived', 
             categorical_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Lname', 'NamePrefix'])

# 4. 創建不同模型
ridge = create_model('ridge')  # Ridge 回歸
lda = create_model('lda')      # LDA 線性判別分析
gbc = create_model('gbc')      # GBC 梯度提升樹

# 5. 堆疊模型
stacker = stack_models(estimator_list = [ridge, lda, gbc], meta_model = create_model('lr'))  # 使用邏輯回歸作為元模型

# 6. 儲存最佳模型
save_model(stacker, 'stacker_auc')

# 7. 預測測試數據
pred = predict_model(stacker, data = test_data)

# 顯示結果
pred.head()

