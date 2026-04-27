import BLogisticRegression
import csv

def load_data(file_path):
    X = []
    Y = []
    
    # store mappings for categorical columns
    encoders = {}

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            new_row = []

            for i, val in enumerate(row):
                try:
                    # try converting to float
                    val = float(val)
                except:
                    # categorical → encode
                    if i not in encoders:
                        encoders[i] = {}
                    
                    if val not in encoders[i]:
                        encoders[i][val] = len(encoders[i])
                    
                    val = encoders[i][val]
        
                new_row.append(val)

            X.append(new_row[:-2])  # features
            Y.append(new_row[-2])   # label

    return X, Y

o = BLogisticRegression.LogisticRegression()
X,Y_real = load_data('student_exam_performance_dataset.csv')
print(Y_real.count(0),Y_real.count(1))
o.fit(X[0:8000],Y_real[0:8000],max_epoch_lim=10000)

import pickle
pickle.dump(o,open('model.dat','wb'))
matches = 0

for n,d in enumerate(X[8000:]):
    pred = 0
    if o.predict(d) > 0.5:
        pred = 1
    if pred == Y_real[n+8000]:
        matches +=1

print(matches/(len(Y_real) - 8000))