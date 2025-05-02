import pandas as pd
from sklearn.model_selection import train_test_split
import sys

#SPLIT_PERCENT=0.5
#TRAIN_PERCENT=0.8
SPLIT_PERCENT=float(sys.argv[2])
TEST_PERCENT=float(sys.argv[3])

df=pd.read_csv(sys.argv[1], sep=',', header=0)

X=df.drop(columns=['Target'])
y=df['Target']

X_colab0, X_colab1, y_colab0, y_colab1 = train_test_split(X, y, test_size=SPLIT_PERCENT, random_state=42)

X_colab0['Target']=y_colab0
X_colab1['Target']=y_colab1

X_colab0_train, X_colab0_test, y_colab0_train, y_colab0_test = train_test_split(X_colab0, y_colab0, test_size=TEST_PERCENT, random_state=42)
X_colab1_train, X_colab1_test, y_colab1_train, y_colab1_test = train_test_split(X_colab1, y_colab1, test_size=TEST_PERCENT, random_state=42)

X_colab1_train['Target']=y_colab1_train
X_colab0_train['Target']=y_colab0_train
X_colab1_test['Target']=y_colab1_test
X_colab0_test['Target']=y_colab0_test


X_colab1_train.to_csv('colab1_train.csv', sep=',', index=None)
X_colab1_test.to_csv('colab1_test.csv', sep=',', index=None)
X_colab0_train.to_csv('colab0_train.csv', sep=',', index=None)
X_colab0_test.to_csv('colab0_test.csv', sep=',', index=None)
