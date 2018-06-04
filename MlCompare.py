import pandas as pd
from sklearn.model_selection import train_test_split
from lightning.classification import SDCAClassifier
from sklearn import preprocessing

df = pd.read_csv("iris-data.txt", index_col=0,header=None)

le = preprocessing.LabelEncoder()
le.fit(df.values[:,3])

data = df.values[:,:3]
result = le.transform(df.values[:,3])

data_train, data_test, result_train,result_test = train_test_split(data,result,test_size=0.3, random_state=100)

clf = SDCAClassifier()
clf.fit(data_train, result_train)
predicted = le.inverse_transform(clf.predict(data_test))

with open("./result.csv","w") as f :
    for line in predicted :
        f.write(line + "\n" )