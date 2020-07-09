import math
from pylab import *
import random as rand
import matplotlib.pyplot as plt


def find_distance(point, centroid, vectorSize):
    sum = 0
    for i in range(vectorSize):
        sum += (point[i] - centroid[i]) ** 2
    return math.sqrt(sum)


def fcm(x, num_of_clusters, m=2, max_iteration=100):
    start_time1 = time.time()
    rows = len(x)
    cols = num_of_clusters
    size_of_inside_x = 2
    # 1st step: initialize the membership matrix using this equation
    U = np.empty([rows, cols], dtype=float)
#    U = np.zeros([rows, cols], dtype=float)
#    U[:,0]=np.random.randn(rows, 1)
#    U[:,1]=np.abs (np.sum(U,axis=1,initial=-1.0))

    # This ia loop :)
    for i in xrange(rows):
        sumLeftTo1 = 1
        for j in xrange(cols - 1):
            randNum = rand.random()
            sumLeftTo1 -= randNum
            U[i,j]=randNum
        U[i, j+1]=sumLeftTo1
    #U=np.array([[0.8,0.2],[0.9,0.1],[0.7,0.3],[0.3,0.7],[0.5,0.5],[0.2,0.8]])

    Cj = np.empty([cols, size_of_inside_x], dtype=float)
    for h in range(max_iteration):
        # 2nd step: calculate the centroid using equation

        for k in range(size_of_inside_x):
            for t in range(num_of_clusters):
                resultup = x[:, k] * U[:, t]
                resultup = np.dot(resultup, U[:, t].T)
                resultdown = np.dot(U[:, t], U[:, t].T)
                Cj[t][k] = resultup / resultdown

        # step 3- calculate dissimilarity between the data points and centroids using the Euclidean distance

        dist = np.empty((0, rows), float)
        for k in range(num_of_clusters):
            Cjmatrix = np.tile(Cj[k, :], [rows, 1])
            distres = x - Cjmatrix
            distres = distres * distres
            ones = np.ones([2, 1])
            dist = np.append(dist, ((np.dot(distres, ones).T) ** 0.5), axis=0)

        dist=dist.T
        # step 4- update the membership value
        p=(m-1)**-1
        dist=dist**p
        result=np.sum(dist,axis=1)
        result = result.reshape([rows,1])
        result = np.tile(result, [1, 2])
        U=dist/result

    elapsed_time = time.time() - start_time1
    print ("time is:     ",elapsed_time)
    return U, Cj


data=[]
count = 0
with open('g2-2-30.txt', 'r') as f:
    line = f.readline()
    while line:
        split_line = line.strip().split(",")  # ["1", "0" ... ]
        count += 1
        nums_ls = [int(x) for x in split_line]  # get rid of the quotation marks and convert to int
        data.append(nums_ls)
        line = f.readline()
data=np.array(data)
#data=np.array([[1,6],[2,5],[3,8],[4,4],[5,7],[6,9]])
membershipMatrix, Cj = fcm(data, 2)

for i in range(len(membershipMatrix)):
    if membershipMatrix[i,0] > 0.5:
        plt.plot(data[i,0], data[i,1], 'yo', markersize=2)
    else:
        plt.plot(data[i,0], data[i,1], 'bo', markersize=2)
print(Cj[0,0], Cj[0,0])
print(Cj[1,0], Cj[1,1])
plt.plot(Cj[0,0], Cj[0,1], 'ro', markersize=10)
plt.plot(Cj[1,0], Cj[1,1], 'ro', markersize=10)
plt.show()
