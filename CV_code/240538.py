import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

data = np.load('/content/drive/MyDrive/2024-ml-assignment5/train.npz')
test_data = np.load("/content/drive/MyDrive/2024-ml-assignment5/test.npz")

#train data
x_train = data['x']
y_train = data['y']

x_train_2D = x_train.reshape(-1, 50 * 50)

#test data
x_test = test_data['x']
x_test_2D = x_test.reshape(301, 50*50)

#히스토그램 기반 그래디언트 부스팅
hgb = HistGradientBoostingClassifier(random_state = 42)
scores = cross_validate(hgb, x_train_2D, y_train, return_train_score = True)

print("훈련세트: ", np.mean(scores["train_score"]))
print("시험세트: ", np.mean(scores["test_score"]))

#fit
hgb.fit(x_train_2D, y_train)

result = hgb.predict(x_test_2D)

#test 결과를 위한 파일
df = pd.read_csv("/content/drive/MyDrive/2024-ml-assignment5/submission.csv")

df.dropna(axis=1, inplace=True)

y_pred = result

df["Class"] = y_pred
df.to_csv("new_submission.csv", index=False)
