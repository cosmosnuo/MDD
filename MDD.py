from matplotlib.pyplot import MultipleLocator
import ParseXX
import cv2 as cv
import os
import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from model import GAE, GraphEmbedding


def getActivateVector(modeList, activityList):
    col_num = len(activityList)
    activity_matrix = [[0] * col_num for _ in range(len(modeList))]
    for i, l in enumerate(modeList):
        for j, e in enumerate(activityList):
            activity_matrix[i][j] = l.count(e)
    activity_arr = np.array(activity_matrix)
    return activity_arr


def getTransVector(modeList, transitionList):
    col_num = len(transitionList)
    transition_matrix = [[0] * col_num for _ in range(len(modeList))]
    for i, l in enumerate(modeList):
        for j, e in enumerate(transitionList):
            transition_matrix[i][j] = l.count(e)
    transition_arr = np.array(transition_matrix)
    return transition_arr


def countActivity(logList):
    activityList = []
    activityDic = {}
    for trace in logList:
        for activity in trace:
            if activity in activityList:
                activityDic[activity] += 1
            else:
                activityList.append(activity)
                activityDic[activity] = 1
    return activityList


def getTrans(logList):
    logtransitionList = []
    for trace in range(len(logList)):
        trace1 = []
        for j in range(len(logList[trace]) - 1):
            trace1.append(logList[trace][j] + '-' + logList[trace][j + 1])
        logtransitionList.append(trace1)
    return logtransitionList


def countTrans(logtransList):
    transitionList = []
    transitionDic = {}
    for trace in logtransList:
        for transition in trace:
            if transition in transitionList:
                transitionDic[transition] += 1
            else:
                transitionList.append(transition)
                transitionDic[transition] = 1
    return transitionList


class Dataset(object):
    def __init__(self, k, feature):
        self.k = k
        self.feature = feature

        for i in range(self.feature.shape[0]):
            self.feature[i] = self.feature[i].T

        self.idx = np.arange(self.feature[0].shape[0])
        self.graph_dict = {}

        for i in range(self.feature.shape[0]):
            g = pair(self.k, self.feature[i])
            self.graph_dict[i] = g

        for i in range(self.feature.shape[0]):
            self._load(self.feature[i], self.idx, self.graph_dict[i], i)

    def _load(self, feature, idx, graph, i):
        features = sp.csr_matrix(feature, dtype=np.float32)

        idx = np.asarray(idx, dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = graph
        edges = np.asarray(list(map(idx_map.get, edges_unordered.flatten())),
                           dtype=np.int32).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(feature.shape[0], feature.shape[0]),
                            dtype=np.float32)


        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph_dict[i] = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        self.feature[i] = np.asarray(features.todense())


def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    rowsum[rowsum != 0] = 1.0 / rowsum[rowsum != 0]
    r_inv = rowsum.flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def pair(knn='', data='', metrix='euclidean'):
    x_train = data
    n_train = len(x_train)
    x_train_flat = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))[:n_train]
    train_neighbors = NearestNeighbors(n_neighbors=knn + 1, metric=metrix).fit(x_train_flat)
    _, idx = train_neighbors.kneighbors(x_train_flat)

    new_idx = np.empty((idx.shape[0], idx.shape[1] - 1))
    assert (idx >= 0).all()
    for i in range(idx.shape[0]):
        try:
            new_idx[i] = idx[i, idx[i] != i][:idx.shape[1] - 1]
        except Exception as e:
            print(idx[i, ...], new_idx.shape, idx.shape)
            raise e

    idx = new_idx.astype(int)
    graph = np.empty(shape=[0, 2], dtype=int)
    for i, m in enumerate(idx):
        for mm in m:
            graph = np.append(graph, [[i, mm]], axis=0)
    return graph


def trace_loss(adj, k):
    adj = torch.clamp(adj, 0, 1)
    adj = torch.round(adj)
    rowsum = adj.sum(axis=1).detach().cpu().numpy()
    d = torch.zeros(adj.shape).numpy()
    row, col = np.diag_indices_from(d)
    d[row, col] = rowsum
    l = d - adj.detach().cpu().numpy()
    e_vals, e_vecs = np.linalg.eig(l)
    sorted_indices = np.argsort(e_vals)
    q = torch.tensor(e_vecs[:, sorted_indices[0:k:]], dtype=torch.float32).cuda()
    m = torch.mm(torch.t(q), adj)
    m = torch.mm(m, q)
    return torch.trace(m)


def pertrain(trainfile, window_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_logList = ParseXX.Parse(trainfile)
    train_activityList = countActivity(train_logList)
    train_logtransList = getTrans(train_logList)
    train_transList = countTrans(train_logtransList)

    times = int(len(train_logList) / 500)
    for i in range(times):
        start = 500 * i
        end = start + window_size
        print("Training window, start point:{:02d},end point:{:02d}".format(start, end))
        Frame = train_logList[start:end]
        FrameTran = train_logtransList[start:end]

        Act = getActivateVector(Frame, train_activityList)
        Tran = getTransVector(FrameTran, train_transList)
        Feature = np.array([Act.T, Tran.T], dtype='object')
        data = Dataset(k=3, feature=Feature)
        feature0 = torch.FloatTensor(data.feature[0]).to(device)
        Grah1 = dgl.from_networkx(data.graph_dict[0]).to(device)
        Grah2 = dgl.from_networkx(data.graph_dict[1]).to(device)
        adj1 = Grah1.adjacency_matrix().to_dense()
        adj2 = Grah2.adjacency_matrix().to_dense()

        model_g = GraphEmbedding(feature0.shape[0], int(feature0.shape[0] / 2)).cuda()
        optim_ge_p = torch.optim.Adam(model_g.parameters(), lr=0.0001)
        optim_ge_t = torch.optim.Adam(model_g.parameters(), lr=0.01)
        criterion_m = torch.nn.MSELoss()

        model_g.train()
        for epoch in range(600):
            adjin = adj1
            adjin = torch.add(adjin, adj2)

            optim_ge_p.zero_grad()
            adj_r = model_g.forward(adj1, adj2, adjin)
            adj_r = adj_r.to('cuda:0')
            adj1 = adj1.to('cuda:0')
            adj2 = adj2.to('cuda:0')
            loss_gre = (criterion_m(adj_r, adj1) + criterion_m(adj_r, adj2)) / 2
            loss_gtr = trace_loss(adj_r, 3) ** 2
            loss_ge = loss_gre + 0.001 * loss_gtr
            loss_ge.backward()
            optim_ge_p.step()

            torch.save(model_g.state_dict(), 'model_g.pth')


def Embedding(frameLen,stepLen,fileName):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 0:
        print('Lets use', torch.cuda.device_count(), 'GPUs!')

    logList = ParseXX.Parse(fileName)
    activityList = countActivity(logList)
    logtransList = getTrans(logList)
    transList = countTrans(logtransList)
    startIndex = 0
    splitPoint = startIndex + frameLen
    endIndex = splitPoint + frameLen
    trainSet = {"firstFus": [], "secondFus": [], "train_y": []}
    trainnum = 0
    while endIndex <= len(logList):
        firstFrame = logList[startIndex:splitPoint]
        firstFrameTran = logtransList[startIndex:splitPoint]
        secondFrame = logList[splitPoint:endIndex]
        secondFrameTran = logtransList[splitPoint:endIndex]

        startIndex = startIndex + stepLen
        splitPoint = startIndex + frameLen
        endIndex = splitPoint + frameLen

        firstAct = getActivateVector(firstFrame, activityList)
        firstTran = getTransVector(firstFrameTran, transList)
        secondAct = getActivateVector(secondFrame, activityList)
        secondTran = getTransVector(secondFrameTran, transList)

        firstFeature = np.array([firstAct.T, firstTran.T], dtype='object')
        secondFeature = np.array([secondAct.T, secondTran.T], dtype='object')

        firstdata = Dataset(k=3, feature=firstFeature)
        seconddata = Dataset(k=3, feature=secondFeature)

        firstfeature0 = torch.FloatTensor(firstdata.feature[0]).to(device)
        firstfeature1 = torch.FloatTensor(firstdata.feature[1]).to(device)
        secondfeature0 = torch.FloatTensor(seconddata.feature[0]).to(device)
        secondfeature1 = torch.FloatTensor(seconddata.feature[1]).to(device)

        firstIn_feats = [firstfeature0.shape[1], firstfeature1.shape[1]]
        secondIn_feats = [secondfeature0.shape[1], secondfeature1.shape[1]]

        firstGrah1 = dgl.from_networkx(firstdata.graph_dict[0]).to(device)
        firstGrah2 = dgl.from_networkx(firstdata.graph_dict[1]).to(device)
        secondGrah1 = dgl.from_networkx(seconddata.graph_dict[0]).to(device)
        secondGrah2 = dgl.from_networkx(seconddata.graph_dict[1]).to(device)

        firstadj1 = firstGrah1.adjacency_matrix().to_dense()
        firstadj2 = firstGrah2.adjacency_matrix().to_dense()
        secondadj1 = secondGrah1.adjacency_matrix().to_dense()
        secondadj2 = secondGrah2.adjacency_matrix().to_dense()

        firstmodel_g = GraphEmbedding(firstfeature0.shape[0], int(firstfeature0.shape[0] / 2)).cuda()
        firstmodel_g.load_state_dict(torch.load('model_g.pth'))
        firstmodel_g.eval()
        firstadjin = firstadj1
        firstadjin = torch.add(firstadjin, firstadj2)
        firstadj_r = firstmodel_g.forward(firstadj1, firstadj2, firstadjin)

        adj_p = torch.clamp(firstadj_r, 0, 1)
        adj_p = torch.round(adj_p + 0.1)
        adj_pn = adj_p.detach().cpu().numpy()
        adj_pn += adj_pn.T
        firstgraph = nx.from_numpy_array(adj_pn, create_using=nx.DiGraph())
        firstgraph = dgl.from_networkx(firstgraph)
        firstgraph = firstgraph.to(device)

        secondmodel_g = GraphEmbedding(secondfeature0.shape[0], int(secondfeature0.shape[0] / 2)).cuda()
        secondmodel_g.load_state_dict(torch.load('model_g.pth'))
        secondmodel_g.eval()
        secondadjin = secondadj1
        secondadjin = torch.add(secondadjin, secondadj2)
        secondadj_r = secondmodel_g.forward(secondadj1, secondadj2, secondadjin)

        adj_p = torch.clamp(secondadj_r, 0, 1)
        adj_p = torch.round(adj_p + 0.1)
        adj_pn = adj_p.detach().cpu().numpy()
        adj_pn += adj_pn.T
        secondgraph = nx.from_numpy_array(adj_pn, create_using=nx.DiGraph())
        secondgraph = dgl.from_networkx(secondgraph)
        secondgraph = secondgraph.to(device)


        model = GAE(firstIn_feats, [32, 32], [32, 32], 2).cuda()
        model = model.to(device)
        firstfus, firsth0, firsth1 = model.forward(firstGrah1, firstGrah2, firstfeature0, firstfeature1, firstgraph)
        secondfus, secondh0, secondh1 = model.forward(secondGrah1, secondGrah2, secondfeature0, secondfeature1, secondgraph)

        train_y = 0
        trainSet["firstFus"].append(firstfus)
        trainSet["secondFus"].append(secondfus)
        trainSet["train_y"].append(train_y)
    return trainSet


def Wasserstein_distance(x, y):
    x = x.cpu()
    x = x.detach()
    x = x.numpy()
    y = y.cpu()
    y = y.detach()
    y = y.numpy()
    x_num_rows = x.shape[0]
    x_insert = 1 / x_num_rows
    x_new = np.insert(x, 0, x_insert, axis=1)
    y_num_rows = y.shape[0]
    y_insert = 1 / y_num_rows
    y_new = np.insert(y, 0, y_insert, axis=1)
    dis = cv.EMD(x_new, y_new, cv.DIST_L2)
    return dis[0]


def createTest(trainSet,frameLen,stepLen):
    disList = []
    SPList = []
    SP = frameLen
    stepLen = stepLen
    for i in range(len(trainSet["train_y"])):
        SPList.append(SP)
        SP = SP + stepLen
        firstFus = trainSet["firstFus"][i]
        secondFus = trainSet["secondFus"][i]
        distance = Wasserstein_distance(firstFus, secondFus)
        disList.append(distance)
    return disList, SPList


def kmeans(sumDisList,SPList,jump_dis,change_num):
    xArray = np.array(sumDisList,dtype = float)
    a = xArray.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(a)
    y_kmeans = kmeans.predict(a)
    num = 0
    y_kmeans = y_kmeans.tolist()
    pos0 = y_kmeans.index(0)
    pos1 = y_kmeans.index(1)
    pos2 = y_kmeans.index(2)

    if sumDisList[pos0] > sumDisList[pos1] and sumDisList[pos0] > sumDisList[pos2]:
        num = 0
    elif sumDisList[pos1] > sumDisList[pos0] and sumDisList[pos1] > sumDisList[pos2]:
        num = 1
    else:
        num = 2
    sum_change=findChange(num,y_kmeans,jump_dis,SPList)
    avg_list=filterChange(sum_change,change_num)
    return avg_list


def findChange(num, y_kmeans, jump, SPList):
    sum_change = []
    change_list10 = []
    sign = 'un_num'
    now_jump = 0
    for i in range(len(y_kmeans)):
        if y_kmeans[i] != num and sign == 'un_num':
            continue
        elif y_kmeans[i] != num and sign == 'num':
            now_jump = now_jump + 1
            if now_jump >= jump:
                sum_change.append(change_list10)
                change_list10 = []
                sign = 'un_num'
                now_jump = 0
        elif y_kmeans[i] == num and sign == 'un_num':
            change_list10.append(SPList[i])
            sign = 'num'
        elif y_kmeans[i] == num and sign == 'num':
            now_jump = 0
            change_list10.append(SPList[i])
        if i == len(y_kmeans) - 1 and sign == 'num':
            sum_change.append(change_list10)
    print(sum_change)
    return sum_change


def filterChange(sum_change, change_num):
    avg_list = []
    for i in sum_change:
        if len(i) >= change_num:
            avg_pos = (i[0] + i[len(i) - 1]) / 2
            avg_list.append(int(avg_pos))
    return avg_list


def get_data(filename):
    filename = os.path.basename(filename)
    number_first = list(filter(str.isdigit, filename))[0]
    index1 = filename.find(number_first)
    index2 = filename.find('.')
    num = filename[index1:index2]
    name = filename[:index2]
    return num, name


def demo_plot(x, y, x_maxsize, title, x_major_locator,num,filename):
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(dpi=1080)
    plt.plot(x, y, linewidth=2.0, color="deepskyblue")
    plt.title(title)
    plt.xlabel("trace", fontsize=15)
    plt.ylabel("distance", fontsize=15)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    maxsize = x_maxsize
    m = 0.2
    N = len(x)
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    if num == "2.5k":
        sign = 250
    elif num == "5k":
        sign = 500
    elif num == "7.5k":
        sign = 750
    elif num == "10k":
        sign = 1000
    for i in range(9):
        plt.axvline((i + 1) * sign, color='red', linestyle='--')
    plt.savefig("%s%s.jpg" % (os.path.dirname(filename), title), bbox_inches='tight', dpi=1080)
    # plt.show()
    plt.close()

def main():
    usage = """\
    usage:
        driftDetection.py [-w value] [-r value] [-p value] log_file_path
    options:
        -w complete window size, integer, default value is 100
        -j detection window size, integer, default value is 3
        -n stable period, integer, default value is 3
        """
    import getopt, sys

    try:

        opts, args = getopt.getopt(sys.argv[1:], "w:j:n:f:")
        if len(opts) == 0:
            print(usage)
            return

        window_size = 200
        jump_dis = 3
        change_num = 3
        stepLen = 10
        trainfile = ["cp5k.mxml"]
        log = "cp5k.mxml"

        for opt, value in opts:
            if opt == '-w':
                window_size = int(value)
            elif opt == '-j':
                jump_dis = int(value)
            elif opt == '-n':
                change_num = int(value)
            elif opt == '-f':
                log = value


        print("--------------------------------------------------------------")
        print(" Log: ", log)
        print(" Train_Log: ", trainfile)
        print(" window_size: ", window_size)
        print(" jump_dis: ", jump_dis)
        print(" change_num: ", change_num)
        print(" stepLen: ", stepLen)
        print("--------------------------------------------------------------")

        pertrain(trainfile[0], window_size)
        trainSet = Embedding(window_size, stepLen, log)
        print("views: 2")
        print("------Calculated distance list------")
        DisgraList, SPList = createTest(trainSet, window_size, stepLen)
        print("------Find drift points------")
        avg_list = kmeans(DisgraList, SPList, jump_dis, change_num)
        print("All change points detected: ", avg_list)

        num, name = get_data(log)
        demo_plot(SPList, DisgraList, 30, name, 250, num, log)
        # print(SPList)
        # print(sumDisList)
    except getopt.GetoptError:
        print(usage)
    except SyntaxError as error:
        print(error)
        print(usage)
    return 0


if __name__ == '__main__':
    main()
