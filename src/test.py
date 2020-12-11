from models import *
from utils import *
from matplotlib.colors import LogNorm

def predict(x):
    model.train(False)
    x = x[:, np.newaxis]
      
    X = torch.from_numpy(x).float()

    if torch.cuda.is_available():
        X = X.cuda()
    return model(X).cpu().detach().numpy()[0]


if __name__ == "__main__":
    cool = cooler.Cooler("../test/mat18_50k.cool::/resolutions/50000")

    model = LeNet_regression()
    if torch.cuda.is_available():
        print("cuda")
        model.cuda()
    else:
        print("cpu")
    checkpoint = torch.load("../models/HiC_LeNet_regression_50k.tar", map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])
    #print("acc", checkpoint['accuracies'])
    #print("loss", checkpoint['losses'])

    data = cool.matrix()[:]
    #print(data, data.shape, type(data))
    data = np.nan_to_num(data)
    #print(data, data.shape, type(data))
    #print(np.min(data), np.max(data))

    chr = cool.chroms()[:]
    chr = chr[chr['length']>50000*32]
    #print(chr)
    mybins = cool.bins()[:]
    ends = mybins[mybins['end'].isin(list(chr["length"]))]
    starts = mybins[(mybins['chrom'].isin(list(chr["name"]))) & (mybins['start']==0)]
    #print(ends)
    #print(starts)

    j = 2977
    for i, row in ends.iterrows():
        a = data[i-32 : i, i-32 : i]
        b = data[i-32 : i, j+1 : j+33]
        c = data[j+1 : j+33, i-32 : i]
        d = data[j+1 : j+33, j+1 : j+33]

        tmp = np.empty((64, 64))

        tmp[:32, :32] = a
        tmp[:32, 32:] = b
        tmp[32:, :32] = c
        tmp[32:, 32:] = d

        q = int(predict((b.copy())[np.newaxis, :]) * 2500000)
        print(row['chrom'] + ": " + str(q))
        
        plt.imshow(tmp, cmap='gray_r', norm=LogNorm())
        plt.axis('off')
        plt.title(row['chrom'] + ": " + str(q))
        plt.savefig("test_50k_" + row['chrom'] + ".png", bbox_inches="tight")
        plt.close()
