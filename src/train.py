from models import *
from utils import *

logger = open("regression.log", "a")
for res in ['5000', '10000', '50000', '100000']:
    cool = cooler.Cooler('data/chm13.draft_v1.0.mcool::/resolutions/' + res + '/')
    print(cool.chroms()[:].shape)
    print(cool.bins()[:].shape)
    print(cool.pixels()[:].shape)
    print(cool.chroms()[:])
    chrs = cool.chroms()[:].loc[:, 'name'].values[1:22]
    
    cl = 50
    model = LeNet_regression()
    if torch.cuda.is_available():
        print("cuda")
        model.cuda()
    else:
        print("cpu")

    opt = optim.Adam(model.parameters())
    loss = nn.MSELoss()
    data = DataIteratorFiltered_TopRight_Regression(cool, sz=32, bs=100, n_classes=cl, chrs=chrs[:-1], delta=0)
    test_data = DataIteratorFiltered_TopRight_Regression(cool, sz=32, bs=100, n_classes=cl, chrs=[chrs[-1]], delta=0)

    epochs=10
    path = "HiC_" + model.name + "_" + res + "_" + str(datetime.date.today())
    accuracies, losses = train_net(model, opt, loss, epochs, data, test_data, path, regression=True)
    plot_accuracy(accuracies, losses, epochs, path)
    print(path + "\naccuracy =", accuracies[-1], "loss =", losses[-1], file=logger)
