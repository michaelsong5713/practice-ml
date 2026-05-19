import torch
import numpy as np
import matplotlib.pyplot as plt


'''
Basic idea: Utilize pytorch to implement a linear regression from scratch. 
Try to teach the model to get as close as possible to the line y = 2x + 1.
'''
values = []
iterations = 20
for runs in range(iterations):
    generations = 100
    lr = 0.01
    datax = []
    datay = []
    ycorrect = []
    for x in range(-10,10):
        y = 2*x+1
        noise_x = np.random.normal(loc=0, scale=2)*0.005
        noise_y = np.random.normal(loc=0, scale=2)*0.005
        datax.append(x+noise_x)
        datay.append(y+noise_y)
        ycorrect.append(y)

    tensor_x = torch.tensor(datax)
    tensor_y = torch.tensor(datay)
    proper_y = torch.tensor(ycorrect)

    weight = torch.randn(1,requires_grad=True)
    bias = torch.randn(1,requires_grad=True)

    for i in range(generations):
        y = weight*tensor_x+bias
        MSE = 0
        for i in range(y.numel()):
            MSE += (y[i]-tensor_y[i])**2
        MSE = MSE/y.numel()
        MSE.backward()
        with torch.no_grad():
            weight -= weight.grad*lr
            bias -= bias.grad*lr
        weight.grad = None
        bias.grad = None

    with torch.no_grad():
        y = weight*tensor_x+bias
        percentage_error = (y-proper_y).abs() / proper_y.abs() * 100
        percentage = percentage_error.mean().item()
        print(f"Percentage error: {percentage}%")
        print(f"Final weight: {weight.item()}")
        print(f"Final bias: {bias.item()}")
        values.append([weight,bias])

#The following was vibe-coded to represent my data
weights = [v[0].item() for v in values]
biases = [v[1].item() for v in values]

plt.scatter(biases, weights, color='blue', label='Runs')
plt.axhline(y=2, color='red', linestyle='--', label='Target weight (2)')
plt.axvline(x=1, color='green', linestyle='--', label='Target bias (1)')
plt.xlabel('Bias')
plt.ylabel('Weight')
plt.title('Weight vs Bias across runs')
plt.legend()
plt.savefig('plot.png')
plt.show()