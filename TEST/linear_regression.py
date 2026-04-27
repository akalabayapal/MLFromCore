import csv
import LinearRegression
dataset = open('dataset_regression.csv','r')

csv_r = csv.reader(dataset)

i = True
X = []
# lr = 0.00001
# t0= 0.001
Y_real = []
for row in csv_r:
    if i == True:
        i =False   
        continue
    X.append(row[0:3]+row[4:-1])
    Y_real.append(row[3])

o = LinearRegression.LinearRegression()
o.fit(X[:300],Y_real[:300],max_epoch_lim=10000000)
for n,i in enumerate(o.predictList(X[301:])):
    print(i,Y_real[301+n])



# #get shape
# X_len = len(X[0])
# Y_len = len(X)

# print("Data Shape:",Y_len,'X',X_len)

# W = []


# for j in range(len(X[0])):
#     col = [float(X[i][j]) for i in range(len(X))]
#     mean = sum(col)/len(col)
#     std = (sum((v-mean)**2 for v in col)/len(col))**0.5

#     for i in range(len(X)):
#         X[i][j] = (float(X[i][j]) - mean) / (std + 1e-8)

# Y_real = [float(y) for y in Y_real]
# y_mean = sum(Y_real)/len(Y_real)
# y_std = (sum((y-y_mean)**2 for y in Y_real)/len(Y_real))**0.5

# Y_real = [(y - y_mean)/(y_std + 1e-8) for y in Y_real]

# for wn in range(X_len):
#     W.append(random.uniform(-0.01, 0.01))

# b = 0

# def epoch(X,Y,W,b):
#     for n, x in enumerate(X):
#         y_p = sum(W[i]*float(x[i]) for i in range(len(W))) + b
#         err = y_p - float(Y[n])

#         for i in range(len(W)):
#             W[i] -= lr * err * float(x[i])

#         b -= lr * err
#     return b
#             #print(W[i])
# def loss(X, Y, W, b):
#     total = 0
#     for n, x in enumerate(X):
#         y_p = sum(W[i]*float(x[i]) for i in range(len(W))) + b
#         total += (y_p - float(Y[n]))**2
#     return total / len(X)
# loss_last = 0

# print(W)
# for x in range(10000):
#     b2 = epoch(X[0:200],Y_real[0:200],W,b)
#     b = b2

#     l = loss(X[0:200],Y_real[0:200],W,b)
#     delt_l = loss_last-float(l)
#     loss_last = l
#     if t0 > math.sqrt(delt_l**2):
#         break
# print(W)