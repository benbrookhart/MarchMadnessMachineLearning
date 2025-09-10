import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("TkAgg") # fixes a Qt platform error
import matplotlib.pyplot as plt

def tts(x, t, train_pct):
	indexes = np.arange(len(x))
	np.random.shuffle(indexes)
	split_i = int(train_pct * len(x))
	x_train, x_test = x[indexes[:split_i]], x[indexes[split_i:]]
	t_train, t_test = t[indexes[:split_i]], t[indexes[split_i:]]
	return x_train, x_test, t_train, t_test

def sigmoid(x):
	return(1/(1 + np.exp(-x)))

def f_forward(x, w1, w2):
	z1 = x.dot(w1)
	a1 = sigmoid(z1)
	z2 = a1.dot(w2)
	a2 = sigmoid(z2)
	return(a2)

def generate_wt(x, y):
	li =[]
	for i in range(x * y):
		li.append(np.random.randn())
	return(np.array(li).reshape(x, y))

def loss(out, y):
	rmse_loss = np.mean((out - y) ** 2)
	mae_loss = np.mean(np.abs(out - y))
	num = np.sum((out - y) ** 2)
	denom = np.sum(y - (np.mean(y) ** 2))
	r2_loss = num / denom
	return rmse_loss + mae_loss + r2_loss

def back_prop(x, y, w1, w2, alpha):
	z1 = x.dot(w1)
	a1 = sigmoid(z1) 
	z2 = a1.dot(w2)
	a2 = sigmoid(z2)
	d2 =(a2-y)
	d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), (np.multiply(a1, 1-a1)))
	w1_adj = x.transpose().dot(d1)
	w2_adj = a1.transpose().dot(d2)
	w1 = w1-(alpha*(w1_adj))
	w2 = w2-(alpha*(w2_adj))
	return(w1, w2)

def is_right(pred, actual):
	p = pred[0]
	max_pred = 0
	pred_index = -1
	for i in range(len(pred[0])):
		if p[i] > max_pred:
			max_pred = p[i]
			pred_index = i
	actual_index = -1
	for i in range(len(actual)):
		if actual[i] == 1:
			actual_index = i
	# print(f"pred {pred_index} vs actual {actual_index}")
	weighted_score = 0
	diff = np.abs(actual_index - pred_index)
	if diff == 0:
		weighted_score = 1
	elif diff == 1:
		weighted_score = 0.5
	elif diff == 2:
		weighted_score = 0.25
	else:
		weighted_score = 0
	return pred_index, actual_index, pred_index == actual_index, weighted_score

def train(x, y, w1, w2, alpha = 0.01, epoch = 10):
	acc =[]
	losss =[]
	for j in range(epoch):
		l =[]
		correct = 0
		weighted = 0
		for i in range(len(x)):
			out = f_forward(x[i], w1, w2)
			l.append((loss(out, y[i])))
			w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
			_, _, right, weight = is_right(out, y[i])
			correct += right
			weighted += weight
		accuracy = correct / len(x)
		weighted_acc = weighted / len(x)
		print("epochs:", j + 1, " Accuracy:", accuracy, "Weighted Accuracy:", weighted_acc, "Loss: ", (sum(l)/len(x))) 
		acc.append(accuracy)
		losss.append(sum(l)/len(x))
	return(acc, losss, w1, w2)

df = pd.read_csv('PREPROCESSED_DATA.csv', index_col = False)
data = df.to_numpy()
x = data[:,2:]
x = x.astype(float)
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
t = data[:,0]
t = np.array([int(i) for i in t])

layer_sizes = [10, 30, 100, 300]
learning_rates = [0.1, 0.01, 0.001]
iterations = [250, 500]

x_train, x_test, t_train, t_test = tts(x, t, train_pct = 0.8)
smote=SMOTE(sampling_strategy='not majority') 
x_train_smoted, t_train_smoted = smote.fit_resample(x_train, t_train)

t_test_final = []
for i in t_test:
	temp = [0 for i in range(7)]
	temp[i] = 1
	t_test_final.append(temp)
	
t_train_final = []
for i in t_train_smoted:
	temp = [0 for i in range(7)]
	temp[i] = 1
	t_train_final.append(temp)
	
samples = np.shape(x_train_smoted)[0]
x_train_final = x_train_smoted.reshape(samples, 1, 128)

samples = np.shape(x_test)[0]
x_test_final = x_test.reshape(samples, 1, 128)

# for ls in layer_sizes:
# 	for lr in learning_rates:
# 		for iter in iterations:
# 			w1 = generate_wt(128, ls)
# 			w2 = generate_wt(ls, 7)

# 			acc, losss, w1, w2 = train(x_train_final, t_train_final, w1, w2, lr, iter)

# 			l = []
# 			correct = 0
# 			weighted = 0
# 			test_preds = []
# 			test_actuals = []
# 			for i in range(len(x_test_final)):
# 				out = f_forward(x_test_final[i], w1, w2)
# 				l.append((loss(out, t_test_final[i])))
# 				pred, act, right, weight = is_right(out, t_test_final[i])
# 				correct += right
# 				weighted += weight
# 				test_preds.append(pred)
# 				test_actuals.append(act)
# 			weighted_acc = weighted / len(x_test_final)
# 			acc = correct / len(x_test_final)
# 			losss = sum(l)/len(x_test_final)
# 			# print(f"Test Set Raw Accuracy: {acc}")
# 			print(f"With layer size = {ls}, learning rate = {lr}, and iterations = {iter}, we got an accuracy of {round(acc, 2)}, with weighted accuracy {round(weighted_acc, 2)}, with loss {round(losss, 2)}")
# 			# print(f"Test Set Loss: {losss}")

# 			cm = confusion_matrix(test_preds, test_actuals, labels=[0, 1, 2, 3, 4, 5, 6])
# 			# print(cm)

w1 = generate_wt(128, 300)
w2 = generate_wt(300, 7)

acc, losss, w1, w2 = train(x_train_final, t_train_final, w1, w2, 0.001, 100)

iters = np.linspace(1, 100, 100)
# print(acc)
# print(losss)
plt.plot(iters, acc, color='blue', linestyle='-', marker='o')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title('Accuracies of Training')
plt.show()
plt.plot(iters, losss, color='red', linestyle='-', marker='o')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title('Losses of Training')
plt.show()

l = []
correct = 0
weighted = 0
test_preds = []
test_actuals = []
for i in range(len(x_test_final)):
	out = f_forward(x_test_final[i], w1, w2)
	l.append((loss(out, t_test_final[i])))
	pred, act, right, weight = is_right(out, t_test_final[i])
	correct += right
	weighted += weight
	test_preds.append(pred)
	test_actuals.append(act)
acc = correct / len(x_test_final)
weighted_acc = weighted / len(x_test_final)
losss = sum(l)/len(x_test_final)
print(f"Test Set Raw Accuracy: {acc}")
print(f"Test Set Weighted Accuracy: {weighted_acc}")
print(f"Test Set Loss: {losss}")

cm = confusion_matrix(test_preds, test_actuals, labels=[0, 1, 2, 3, 4, 5, 6])
#print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5, 6])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()