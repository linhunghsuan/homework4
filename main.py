import pandas as pd
from pycaret.classification import *

# 讀取數據
data = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/titanic.xlsx', sheet_name='Sheet1')

# 初始化 PyCaret 環境
# 'Survived' 是目標變量，其他的則是特徵
try:
    clf = setup(data, target='Survived', session_id=42, 
                categorical_features=['Sex', 'Embarked'], 
                numeric_features=['Age', 'Fare', 'SibSp', 'Parch', 'Pclass'])
except Exception as e:
    print(f"Error occurred: {e}")

# 比較所有可用的模型
best_model = compare_models()

# 創建和訓練最好的模型
model = create_model('cnn') 

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
