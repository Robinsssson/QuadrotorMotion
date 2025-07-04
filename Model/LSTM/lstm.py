import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Lstm(nn.Module):
    def __init__(self, hidden_size, number_layer=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layer = number_layer
        self.lstm = nn.LSTM(1, hidden_size, number_layer)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, input, hidden=None):
        out, self.hidden = self.lstm(input.view(len(input) ,1, -1), hidden)
        out = out.view(-1, self.hidden_size)
        out = self.fc(out)
        return out

def per_train():  
    test = np.array([i for j in range(100) for i in range(j, 5+j)], dtype=np.float64)
    # test 500
    test_set = test[:400]
    tmean = np.mean(test_set)
    tmax = test_set.max()
    tmin = test_set.min()
    test_set = (test_set - tmean)/(tmax - tmin)
    test = torch.FloatTensor(test)
    per = test_set[:400].tolist()
    return test, test_set, per, (tmean, tmax, tmin)

def create_test(test_set, len):
    lst = []
    for i in range(400-len-1): #380 
        lst.append((torch.FloatTensor(test_set[i:i+len]), torch.FloatTensor(test_set[i+1:i+len+1])))
    return lst

def train(test_set, times, length):    
    model = Lstm(16, 4)
    loss_function = nn.MSELoss()
    loss_function.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    test_set = torch.FloatTensor(test_set)
    model.train()
    model = model.to(device)
    single_loss = None
    min_loss = 100.
    for i in range(times):
        for test_s, label_s in create_test(test_set, length):
            test_s = test_s.to(device)
            label_s = label_s.to(device)
            optimizer.zero_grad()
            out = model(test_s)
            out = out.to(device)
            single_loss = loss_function(out.view(-1), label_s.view(-1))
            single_loss.backward()
            optimizer.step()
            if min_loss > single_loss.item():
                min_loss = single_loss.item()
                torch.save(model.state_dict(), 'model.pth')
        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.10f} minloss = {min_loss:10.10f}')
            
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    return model, length

## perdict
def perdict(model,test, per, tmean, tmax, tmin, length):
    model.eval()
    model = model.to(device)
    test = test.to(device)
    loss_function = nn.MSELoss()
    loss_function = loss_function.to(device)
    for i in range(100):
        test_s = torch.FloatTensor(per[-length:])
        test_s = test_s.to(device)
        with torch.no_grad():
            out = model(test_s)
            out = out.to(device)
            per.append(out[-1].item())
            out = out * (tmax - tmin) + tmean
            single_loss = loss_function(out.view(-1), test[401-length+i:401+i].view(-1))
            print(f'perdict: {i:3} loss: {single_loss.item():10.10f}')
    per_plt = np.array(per[-100:])
    per_plt = per_plt * (tmax - tmin) + tmean
    return per_plt
    
def mt_plot(test, per_plt):
    fig, ax = plt.subplots()
    ax.plot([i for i in range(400, 500)], test[400:], color="red")
    ax.plot([i for i in range(400, 500)], per_plt, color="black")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    plt.savefig("test"+current_time+".png")
    plt.show()
    
if __name__ == '__main__':
    LENGTH = 80
    test, test_set,per, (tmean, tmax, tmin) = per_train()
    model, length = train(test_set, 300, LENGTH)
    model.load_state_dict(torch.load('model.pth'))
    perplt = perdict(model,test, per, tmean, tmax, tmin, length)
    mt_plot(test, perplt)
