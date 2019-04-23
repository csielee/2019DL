import numpy as np
import matplotlib.pyplot as plt

def toBinaray(x, digits, complement=False):
    if complement and x < 0:
        x += (1<<(digits))
    x = abs(x)
    return [ float(int(i)) for i in list(("{:0" + str(digits) + "b}").format(x))[::-1]][:digits]

def toNumber(b, complement=False):
    if complement:
        last = b[-1]
        b = b[:-1]
    d = sum([int(x)<<i for i, x in enumerate(b)])
    if complement:
        d -= int(last)*(1<<(len(b)))
    return d

def BinaryDataset(digits=8):
    thr = (1<<(digits-1))
    while True:
        a, b = np.random.randint(0, thr, 2)
        c = a + b
        x = np.array([toBinaray(a, digits), toBinaray(b, digits)])
        y = np.array([toBinaray(c, digits)])
        yield [Variable(x.T[i:i+1, :]) for i in range(digits)], [Variable(y.T[i:i+1, :]) for i in range(digits)]

class Variable():
    def __init__(self, data, T=None, grad=None, copy=True):
        if data is None or type(data) != np.ndarray:
            raise AttributeError('Wrong data type')
        
        if copy:
            self.data = data.copy()
        else:
            self.data = data
        if grad is None:
            grad = np.zeros_like(self.data)
        self.grad = grad
        if T is None:
            T = Variable(self.data.T, self, self.grad.T, copy=False)
        self.T = T
        self.fn = None
        self.child = []
        self.ready = False
    
    def zero_grad(self):
        self.grad[:,:] = 0.0
        self.child = []
        self.ready = False
    
    def __repr__(self):
        return 'Variable(\n{}\n)\n'.format(self.data.__str__())
    
    def __str__(self):
        return self.data.__str__()
    
    def __add__(self, b):
        if type(b) is not Variable:
            b = Variable(np.ones_like(self.data)*b)
            
        c = Variable(self.data + b.data)
        c.fn = [Variable.__grad_add__, self, b]
        
        self.child.append(c)
        b.child.append(c)
        return c
    
    def __grad_add__(self, a, b):
        a.grad += np.ones_like(a.grad) * self.grad
        b.grad += np.ones_like(b.grad) * self.grad
    
    def __sub__(self, b):
        if type(b) is not Variable:
            b = Variable(np.ones_like(self.data)*b)
        c = Variable(self.data - b.data)
        c.fn = [Variable.__grad_sub__, self, b]
        
        self.child.append(c)
        b.child.append(c)
        return c
    
    def __grad_sub__(self, a, b):
        a.grad += np.ones_like(a.grad) * self.grad
        b.grad -= np.ones_like(b.grad) * self.grad
    
    def __mul__(self, b):
        if type(b) is not Variable:
            b = Variable(np.ones_like(self.data)*b)
        
        c = Variable(self.data * b.data)
        c.fn = [Variable.__grad_mul__, self, b]
        
        self.child.append(c)
        b.child.append(c)
        return c
    
    def __grad_mul__(self, a, b):
        a.grad += b.data * self.grad
        b.grad += a.data * self.grad
    
    def __matmul__(self, b):
        c = Variable(np.matmul(self.data, b.data))
        c.fn = [Variable.__grad_matmul__, self, b]
           
        self.child.append(c)
        b.child.append(c)
        return c
    
    def __grad_matmul__(self, a, b):
        a.grad += np.matmul(self.grad, b.data.T)
        b.grad += np.matmul(a.data.T, self.grad)
    
    
    def tanh(self):
        c = Variable(np.tanh(self.data))
        c.fn = [Variable.__grad_tanh__, self]
        
        self.child.append(c)
        return c
        
    def __grad_tanh__(self, a):
        a.grad += self.grad * (1 - (self.data**2))
    
    def crossentropy(self, target):
        s = self.softmax(1)
        if type(target) is Variable:
            target = target.data
            
        target = target.astype(np.int)
        
        if target.shape[0] > 1:
            slis = tuple(zip(range(target.shape[0]), target))
        else:
            slis = (0, target[0])
        
        c = Variable(np.array(np.sum(-np.log(s[slis]))))
        c.fn = [Variable.__grad_corssentropy, self, target]
        
        self.child.append(c)
        return c
    
    def __grad_corssentropy(self, a, target):
        y = np.zeros_like(a.grad)
        if target.shape[0] > 1:
            slis = tuple(zip(range(target.shape[0]), target))
        else:
            slis = (0, target[0])
            
        y[slis] = 1.0
        a.grad += (a.softmax(1) - y)
    
    def softmax(self, dim):
        # move dim idxs
        exp_data = np.exp(self.data)
        return exp_data / np.sum(exp_data, axis=dim).reshape([-1]+[1 for _ in range(dim)])
    
    def argsoftmax(self):
        s = self.softmax(1)
        return Variable(np.argmax(s, axis=1).reshape(-1,1))
    
    def backward(self, backward_grad):
        if type(backward_grad) is Variable:
            backward_grad = backward_grad.data
        
        if backward_grad.shape != self.data.shape:
            raise AttributeError('Wrong backward grad shape {} != {}'.format(backward_grad.shape, self.data.shape))
        
        self.grad = backward_grad
        self.__backward()
    
    def __backward(self):
        if self.fn is None:
            return;
        
        # check self grad is ready, trace child variables
        self.ready = True
        for child in self.child:
            self.ready &= child.ready
        
        if not self.ready:
            return;
        
        backward_op = self.fn[0]
        
        backward_op(self, *self.fn[1:])
        
        for v in self.fn[1:]:
            if type(v) is Variable:
                v.__backward()
                
class RNN():
    def __init__(self, in_channels, out_channels, hidden_channels):
        self.U = Variable(np.random.uniform(-1,1, (in_channels, hidden_channels)))
        self.W = Variable(np.random.uniform(-1,1, (hidden_channels, hidden_channels)))
        self.V = Variable(np.random.uniform(-1,1, (hidden_channels, out_channels)))
        self.b = Variable(np.random.uniform(-1,1, (1, hidden_channels)))
        self.c = Variable(np.random.uniform(-1,1, (1, out_channels)))
    
    def forward(self, x):
        t = len(x)
        self.h = None
        y = []
        
        for i in range(t):
            a = self.b + (x[i] @ self.U)
            if self.h is not None:
                a += (self.h @ self.W)
        
            self.h = a.tanh()
            
            o = self.c + (self.h @ self.V)
            y.append(o)
        
        return y
    
    def zero_grad(self):
        self.U.zero_grad()
        self.W.zero_grad()
        self.V.zero_grad()
        self.b.zero_grad()
        self.c.zero_grad()
    
    def step(self, lr=1e-1):
        if lr is None:
            lr = 1e-2
        self.U.data -= lr * self.U.grad
        self.W.data -= lr * self.W.grad
        self.V.data -= lr * self.V.grad
        self.b.data -= lr * self.b.grad
        self.c.data -= lr * self.c.grad

def train(model, digits, epoch_size, lr=None, show_size=1000):
    dataset = BinaryDataset(digits)
    
    all_error = []
    all_loss = []
    all_accuracy = []
    
    for epoch in range(epoch_size):
        x, y = next(dataset)
        
        model.zero_grad()
        output = model.forward(x)
        loss = [output[i].crossentropy(y[i]) for i in range(len(y))]
        
        for l in loss[::-1]:
            l.backward(np.array(1))
            
        model.step(lr)
        
        e = np.count_nonzero([np.all(output[i].argsoftmax().data != y[i].data) for i in range(len(y))])
        all_error.append(e)
        all_loss.append(sum([l.data for l in loss]))
        
        if e == 0:
            all_accuracy.append(1)
        else:
            all_accuracy.append(0)
        
        if (epoch+1) % show_size == 0:
            print('[{:5d}] error : {}, loss : {}'.format(epoch+1, sum(all_error[-show_size:])/show_size, sum(all_loss[-show_size:])/show_size ))
        
    return all_error, all_loss, all_accuracy

def evaluation(model, digits, epoch_size, show_size):
    accuracy = 0
    
    all_error = []
    all_accuracy = []

    eval_dataset = BinaryDataset(digits)

    for epoch in range(epoch_size):
        x, y = next(eval_dataset)
    
        output = model.forward(x)
    
        e = np.count_nonzero([np.all(output[i].argsoftmax().data != y[i].data) for i in range(len(y))])
        all_error.append(e)
        
        if e == 0:
            all_accuracy.append(1)
        else:
            all_accuracy.append(0)
    
        output = [float(o.argsoftmax().data) for o in output]\
    
        x = np.concatenate([v.data for v in x]).T
        y = np.concatenate([v.data for v in y]).T
        
        if (epoch+1) % show_size == 0:
            print('[{:5d}] error : {}\t{:d} + {:d} = {:d}, model:{:d}'.format(
                epoch+1, sum(all_error[-show_size:])/show_size,
                toNumber(x[0,:]), toNumber(x[1, :]), toNumber(y[0,:]), toNumber(output)
            ))
    
    accuracy =  sum(all_accuracy) / epoch_size
    if show_size < epoch_size:
        print('Accuracy : {:.3f}%'.format(accuracy))
    
    return accuracy, all_error, all_accuracy


def show(all_accuracy, mean_size=1000):
    x = [((i+0.5)*mean_size) for i in range(len(all_accuracy)//mean_size)]
    y = [sum(all_accuracy[i*mean_size:(i+1)*mean_size])/ mean_size for i in range(len(all_accuracy)//mean_size)]
    
    plt.figure(figsize=(10,6))
    plt.title('Episode Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.plot(x, y, '-o', label='accuracy')
    
    plt.xlim((0, len(all_accuracy)))
    plt.xticks(x)
    plt.legend()
    plt.show()


model = RNN(2, 2, 16)



print('Training ... ')
_, _, a = train(model, 8, 20000)

show(a[:10000])

print('Evaluation ...')
evaluation(model, 8, 1000, 100)