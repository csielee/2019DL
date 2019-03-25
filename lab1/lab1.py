import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy(n=11):
    inputs = []
    labels = []
    step = 1/(n-1)
    for i in range(n):
        inputs.append([step*i, step*i])
        labels.append(0)
        
        if i == int((n-1)/2):
            continue
        
        inputs.append([step*i, 1 - step*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(n*2 - 1,1)

def show_result(x, y, pred_y):
    pred_y = np.round(pred_y)
    cm = LinearSegmentedColormap.from_list(
        'mymap', [(1, 0, 0), (0, 0, 1)], N=2)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    plt.scatter(x[:,0], x[:,1], c=y[:,0], cmap=cm)
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    plt.scatter(x[:,0], x[:,1], c=pred_y[:,0], cmap=cm)
    
    plt.show()
    
def show_data(xs, ys, ts):
    cm = LinearSegmentedColormap.from_list(
        'mymap', [(1, 0, 0), (0, 0, 1)], N=2)
    n = len(xs)
    plt.figure(figsize=(5*n, 5))
    for i, x, y, t in zip(range(n), xs, ys, ts):
        y = np.round(y)
        plt.subplot(1,n, i+1)
        plt.title(t, fontsize=18)
        plt.scatter(x[:,0], x[:,1], c=y[:,0], cmap=cm)
    
    plt.show()
    
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def loss(y, y_hat):
    return np.mean((y - y_hat)**2)
    
def derivative_loss(y, y_hat):
    return (y - y_hat)*(2/y.shape[0])

class layer():
    def __init__(self, input_size, output_size):
        self.w = np.random.normal(0, 1, (input_size+1, output_size))
        
    def forward(self, x):
        x = np.append(x, np.ones((x.shape[0],1)), axis=1)
        self.forward_gradient = x
        self.y = sigmoid(np.matmul(x, self.w))
        return self.y
    
    def backward(self, derivative_C):
        self.backward_gradient = np.multiply(derivative_sigmoid(self.y), derivative_C)
        return np.matmul(self.backward_gradient, self.w[:-1].T) 

    def update(self, learning_rate):
        self.gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)
        self.w -= learning_rate*self.gradient
        return self.gradient
        
class NN():
    def __init__(self, sizes, learning_rate = 0.1):
        self.learning_rate = learning_rate
        sizes2 = sizes[1:] + [0]
        self.l = []
        for a,b in zip(sizes, sizes2):
            if (a+1)*b == 0:
                continue
            self.l += [layer(a,b)]
            
    def forward(self, x):
        _x = x
        for l in self.l:
            _x = l.forward(_x)
        return _x
    
    def backward(self, dC):
        _dC = dC
        for l in self.l[::-1]:
            _dC = l.backward(_dC)
            
    def update(self):
        gradients = []
        for l in self.l:
            gradients += [l.update(self.learning_rate)]
        return gradients
    
if __name__ == "__main__":
    nn_linear = NN([2,4,4,1], 1)
    nn_XOR = NN([2,4,4,1], 1)
    epoch_count = 10000
    loss_threshold = 0.005
    linear_stop = False
    XOR_stop = False
    x_linear, y_linear = generate_linear()
    x_XOR, y_XOR = generate_XOR_easy()
    for i in range(epoch_count):
        if not linear_stop:
            y = nn_linear.forward(x_linear)
            loss_linear = loss(y, y_linear)
            nn_linear.backward(derivative_loss(y, y_linear))
            nn_linear.update()
            
            if loss_linear < loss_threshold:
                print('linear is covergence')
                linear_stop = True
        
        if not XOR_stop:
            y = nn_XOR.forward(x_XOR)
            loss_XOR = loss(y, y_XOR)
            nn_XOR.backward(derivative_loss(y, y_XOR))
            nn_XOR.update()
            
            if loss_XOR < loss_threshold:
                print('XOR is covergence')
                XOR_stop = True
    
        if i%200 == 0 or (linear_stop and XOR_stop):
            print(
                '[{:4d}] linear loss : {:.4f} \t XOR loss : {:.4f}'.format(
                    i, loss_linear, loss_XOR))
            
        if linear_stop and XOR_stop:
            break
            
    y1 = nn_linear.forward(x_linear)
    show_result(x_linear, y_linear, y1)
    print('linear test loss : ', loss(y1, y_linear))
    y2 = nn_XOR.forward(x_XOR)
    show_result(x_XOR, y_XOR, y2)
    print('XOR test loss : ', loss(y2, y_XOR))
    print('\n linear test result : \n',y1)
    print('\n XOR test result : \n',y2)