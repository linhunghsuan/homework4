from pycaret.classification import *
import pandas as pd

# 讀取數據
data = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/titanic.xlsx', sheet_name='Sheet1')


# 讀取數據
data = pd.read_excel('titanic.xlsx', sheet_name='Sheet1')

# 刪除不需要的欄位
data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Body', 'Home.dest'], axis=1)

# 填補缺失值
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# 選擇必要的特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
data = data[features + ['Survived']]  # 只保留必要的欄位

# 使用 PyCaret setup 進行數據預處理
clf = setup(data, target='Survived', session_id=42,
            categorical_features=['Sex', 'Embarked'], 
            numeric_features=['Age', 'Fare', 'SibSp', 'Parch', 'Pclass'])

# 創建模型
model = create_model('svm')


# 在測試數據集上進行預測
predictions = predict_model(model)

# 評估模型
evaluate_model(model)

# 假設測試集為 test_data
test_data = pd.read_excel('titanic_test.xlsx', sheet_name='Sheet1')

# 預測
test_predictions = predict_model(model, data=test_data)

# 準備提交文件
submission = test_predictions[['PassengerId', 'Label']]  # 'Label' 是預測的目標變量
submission.rename(columns={'Label': 'Survived'}, inplace=True)
submission.to_csv('submission.csv', index=False)
