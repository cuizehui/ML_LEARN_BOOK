import torch


x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    # 构造函数
    def __init__(self):
        super(LinearModel, self).__init__()
        # Linear （in_features: int, out_features: int, bias: bool = True, device=None, dtype=None)
        # 实例化对象
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # linear是个对象，对象后面加括号，表示创建了一个可调用的对象（callable）
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
# criterion = torch.nn.MSELoss(size_average=False)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    # loss是个对象
    loss = criterion(y_pred, y_data)
    # 打印loss时会自动调用loss.__str()__
    print(epoch, loss.item())
    # 梯度置0
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 自动更新，根据梯度，所设置的学习率自动更新
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())


#保存
torch.save(model.state_dict(),'model.pkl')  #保存模型的参数  w  b
#加载
# model.load_state_dict(torch.load('model.pkl'))   #加载