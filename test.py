import numpy as np
from kan import *

def get_funcs(test_split=0.2,N=100,squared=True,device="cpu"):
  sin_cos = lambda x1, x2: np.sin(x1) + np.cos(x2)
  sin_cos_2 = lambda x1, x2: sin_cos(x1,x2)**2
  x1 = np.linspace(-1,1,N).reshape((N,1))
  x2 = np.linspace(-1,1,N).reshape((N,1))
  xs = np.hstack([x1,x2])
  target = sin_cos(x1,x2)
  if squared:
    target = sin_cos_2(x1,x2)
  X_train, X_test, y_train, y_test = train_test_split(xs,target, test_size=test_split, random_state=0)
  dataset ={'train_input':X_train ,'train_label':y_train ,'test_input':X_test ,'test_label':y_test}
  for k,v in dataset.items():
    dataset[k] = torch.tensor(v,dtype=torch.float32).to(device)
  return dataset

dataset = get_funcs(test_split=0.2,N=100,squared=True,device="cpu")

model = KAN(width=[2,4,1], grid=3, k=3, seed=0,device="cpu",)
_ = model.train(dataset, opt="LBFGS", steps=50, lamb=0.01, lamb_entropy=10, lr=0.01,device="cpu")
model.plot()

model = model.prune()
_ = model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10, lr=0.01,device="cpu")
model.plot()


import matplotlib.pyplot as plt
train_pred = model(dataset['train_input']).detach().numpy()
test_pred = model(dataset['test_input']).detach().numpy()

# Plot scatter plots
fig, axes = plt.subplots(2, 1, figsize=(6, 10))

# Scatter plot for training data
plt.subplot(2, 1, 1)
plt.scatter(dataset['train_label'], train_pred, label="train")
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Scatter Plot - Training Data')
plt.legend()

# Scatter plot for test data
plt.subplot(2, 1, 2)
plt.scatter(dataset['test_label'], test_pred, label="test")
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Scatter Plot - Test Data')
plt.legend()

plt.tight_layout()
plt.show()

mode = "auto"
if mode == "auto":
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)
model.symbolic_formula()[0][0]


def math_model(x1,x2):
  T1 = 1.34 * (-0.01*np.sin(5.92*x1 + 9.79) + np.sin(1.99*x2 - 0.19) + 0.98 )**0.5
  T2 = -0.77*np.log(0.03*np.sin(6.21*x2 + 8.02) + 0.1) - 1.88
  return T1 + T2
math_pred = math_model(dataset['test_input'][:,0],dataset['test_input'][:,1])
plt.scatter(dataset['test_label'],test_pred,label="pred test")
plt.scatter(dataset['test_label'],math_pred,label="math pred test")
plt.legend()
plt.show()



import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes,  load_breast_cancer

from kan import *

def get_data(test_split=0.2,random_state=42,device="cpu",is_regression=False):
  #_dataset = load_diabetes()
  _dataset = load_breast_cancer()
  print(_dataset.data.shape,_dataset.target.shape)
  y_shape = _dataset.target.shape
  target = _dataset.target
  if not is_regression:
    if len(y_shape)<2:
      N = len(_dataset.target)
      unique_values = np.unique(_dataset.target)
      unique_dic = {_:i for i,_ in enumerate(unique_values)}
      encoded = np.zeros((N, len(unique_values)), dtype=int)
    for i, value in enumerate(_dataset.target):
      encoded[i, unique_dic[value]] = 1
    target = encoded

  X_train, X_test, y_train, y_test = train_test_split(_dataset.data,target, test_size=test_split, random_state=random_state)
  mean = X_train.mean(0)
  std =  X_train.std(0)
  X_train = (X_train - mean)/std
  X_test = (X_test - mean)/std
  #
  #mean, std = y_train.mean(), y_train.std()
  #y_train = (y_train - mean)/std
  #y_test = (y_test - mean)/std
  #
  dataset ={'train_input':X_train ,'train_label':y_train ,'test_input':X_test ,'test_label':y_test}
  for k,v in dataset.items():
    dataset[k] = torch.tensor(v,dtype=torch.float32).to(device)
  return dataset

dataset = get_data(test_split=0.2,random_state=42,device="cpu")


dataset['train_label'].shape
model = KAN(width=[30,4,2], grid=3, k=3, seed=0,device="cpu",)
_ = model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10, lr=0.01,device="cpu",
                loss_fn=torch.nn.CrossEntropyLoss())

model.plot()

modelp = model.prune()


_ = modelp.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10, lr=0.01,device="cpu",
                loss_fn=torch.nn.CrossEntropyLoss())


modelp.plot()


train_true = dataset['train_label'].argmax(1)
train_pred = modelp(dataset['train_input']).cpu().detach().numpy().round().argmax(1)
cm_train = confusion_matrix(train_true, train_pred)

test_true = dataset['test_label'].argmax(1)
test_pred = modelp(dataset['test_input']).cpu().detach().numpy().round().argmax(1)
cm_test = confusion_matrix(test_true, test_pred)

# Plot confusion matrices
import seaborn as sns

# Plot confusion matrices
fig, axes = plt.subplots(2, 1, figsize=(6, 10))

# Plot confusion matrix for training data
plt.subplot(2, 1, 1)
sns.set(font_scale=1.2)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Training Data')

# Plot confusion matrix for testing data
plt.subplot(2, 1, 2)
sns.set(font_scale=1.2)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Test Data')

plt.tight_layout()
plt.show()

mode = "auto" # "manual"
if mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    modelp.auto_symbolic(lib=lib)

modelp.symbolic_formula()[0][0]


