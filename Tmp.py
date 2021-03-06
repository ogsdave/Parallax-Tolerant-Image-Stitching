import numpy as np
import cv2
import math
import itertools
from matplotlib import pyplot as plt
import random
from scipy.optimize import leastsq
from scipy.optimize import minimize
from quadprog import solve_qp
import scipy.misc
import maxflow
from ContentPreservingWarping import *
from numpy import concatenate, ones, zeros

def ravel_index(x, dims):
    i = 0
    for dim, j in zip(dims, x):
        i *= dim
        i += j
    return i

def func1(param, coordinates):
    addition = param[0]*coordinates[0]-param[1]*coordinates[1]+param[2]
    return addition

def func2(param, coordinates):
    addition = param[1]*coordinates[0]+param[0]*coordinates[1]+param[3]    
    return addition

def func(param,coordinates,coordinates1):
    addition = np.sqrt((func1(param,coordinates)-coordinates1[0])**2+(func2(param,coordinates)-coordinates1[1])**2)
    return addition

def Align_func(X,feature_points,Dst_Points,Coefs,sz):

    Ep,T=Local_alignment(X,feature_points,Dst_Points,Coefs,sz)
    Eg=Global_alignment(X,T,sz)
    Es=Smooth_alignment(V,1,feature_points,sz)
    E = (Ep + Eg*0.01 + Es*0.001)
    return E

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def graph_cut(I,src_pts):
    g = maxflow.Graph[float]()
    i_inf = np.inf
    nodeids = g.add_grid_nodes(I.shape)
    h,w = np.shape(I)
    mu, sigma = 0, 1.5 # mean and standard deviation
    weights = np.zeros((h,w))
    l=0
    for k in src_pts:
        weights_x = np.zeros(h)
        weights_y = np.zeros(w)
        weights_tmp = np.zeros((h,w))
        for i in range(h):
            weights_x[i]+=(k[0][0]-i)**2
        weights_x = np.tile(weights_x,(w,1))
        weights_x = np.transpose(weights_x)
        for i in range(w):    
            weights_y[i]+=(k[0][1]-i)**2
        weights_y = np.tile(weights_y,(h,1))
        weights_tmp = weights_x+weights_y
        weights_tmp = gaussian(np.sqrt(weights_tmp),np.mean(weights_tmp),np.std(weights_tmp))
        weights+=weights_tmp
    weights = 1.0/(weights+0.01)
    for i in range(h):
        for j in range(w):
            weights[i][j]=weights[i][j]*I[i][j]
    structure = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    g.add_grid_edges(nodeids, weights, structure=structure,symmetric=True)
    left_most = concatenate((np.arange(I.shape[0]).reshape(1, I.shape[0]), zeros((1, I.shape[0])))).astype(np.uint64)
    left_most = np.ravel_multi_index(left_most, I.shape)
    g.add_grid_tedges(left_most, i_inf, 0)

    
    right_most = concatenate((np.arange(I.shape[0]).reshape(1, I.shape[0]), ones((1, I.shape[0])) * (np.size(I, 1) - 1))).astype(np.uint64)
    right_most = np.ravel_multi_index(right_most, I.shape)
    g.add_grid_tedges(right_most, 0, i_inf)
    x = g.maxflow()
    return x

def Local_alignment(V,feature_points,Dst_Points,Coefs,sz):
    T=np.zeros([sz[0],sz[1]])
    Ep=0

    for i in xrange(len(feature_points)):
        a,b=int(feature_points[i][0]),int(feature_points[i][1])

        T[a][b]=1
        T[a+1][b]=1
        T[a][b+1]=1
        T[a+1][b+1]=1

        tmpx=0
        tmpy=0
        for j in xrange(len(Dst_Points[i])):
            x = Dst_Points[i][j][0]
            y = Dst_Points[i][j][1]

            Idx0 = ravel_index((x, y, 0), sz)
            Idx1 = ravel_index((x, y, 1), sz)

            # print Idx0,Idx1

            tmpx+=V[Idx0]*Coefs[i][j]
            tmpy+=V[Idx1]*Coefs[i][j]

        tmpx-=feature_points[i][0]
        tmpy-=feature_points[i][1]

        Ep += tmpy*tmpy + tmpx*tmpx

    return Ep,T

def Global_alignment(V,T,sz):
	Eg=0
    for i in xrange(sz[0]):
        for j in xrange(sz[1]):
            Idx0 = ravel_index((i, j, 0), sz)
            Idx1 = ravel_index((i, j, 1), sz)
            # print i,j,Idx0,Idx1
            Eg += T[i][j]*( (V[Idx0]-i)*(V[Idx0]-i) + (V[Idx1]-j)*(V[Idx1]-j) )
    return Eg

def Smooth_alignment(V,ws,feature_points,sz):
	Es=0
    for i in range(0,len(feature_points)):
    	for j in range(i,len(feature_points)):
        	for k in range(j,len(feature_points)):
        		a1,b1=int(feature_points[i][0]),int(feature_points[i][1])
        		a2,b2=int(feature_points[j][0]),int(feature_points[j][1])
        		a3,b3=int(feature_points[k][0]),int(feature_points[k][1])

	            Idx0 = ravel_index((a1, b1, 0), sz)
	            Idx1 = ravel_index((a2, b2, 0), sz)
	            Idx2 = ravel_index((a3, b3, 0), sz)

	            Idy0 = ravel_index((a1, b1, 1), sz)
	            Idy1 = ravel_index((a2, b2, 1), sz)
	            Idy2 = ravel_index((a3, b3, 1), sz)

	            u=float((a1-a2)*(a3-a2)+(b1-b2)*(b3-b2))/float((a3-a2)*(a3-a2)+(b3-b2)*(b3-b2))
	            v=float((a1-a2)-u*(a3-a2))/float(b3-b2)

	            tmpx = V(Idx0) - ( V(Idx1) + u*(V(Idx2)-V(Idx1)) + v*(V(Idy2)-V(Idy1)) )
	            tmpy = V(Idy0) - ( V(Idy1) + u*(V(Idy2)-V(Idy1)) - v*(V(Idx2)-V(Idx1)) )

	            Es += ws*(tmpx*tmpx + tmpy*tmpy)
    return Es

i=1
MIN_MATCH_COUNT = 10

fnm1='./Folder'+str(i)+'/'+str(1)+'.jpg'
fnm2='./Folder'+str(i)+'/'+str(2)+'.jpg'

img1 = cv2.imread(fnm1,0)          # queryImage
img2 = cv2.imread(fnm2,0) 		   # trainImage
# Initiate SIFT detector

Size=np.asarray((np.asarray(img1)).shape)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    size = len(src_pts)
    size_dst = len(dst_pts)
    penalty_score = np.full(size, 0)
    has_been_selected = np.full(size,0)
    align_quality = np.inf
    best_alignment_so_far = np.inf
    best_H = np.zeros((3,3))
    while align_quality>=100:
        quality = 0.0
        feature_points_src = []
        feature_points_dst = []
        fl = 0
        src_min_overlap_point_x = np.inf
        src_max_overlap_point_x = -np.inf
        dst_min_overlap_point_x = np.inf
        dst_max_overlap_point_x = -np.inf
        while fl == 0 :
    		seed_index = random.randint(0,size-1)
    		if has_been_selected[seed_index] == 0 and np.mean(penalty_score) >= penalty_score[seed_index]:    			
    			has_been_selected[seed_index] = 1
    			fl=1
    			penalty_score = penalty_score + 1

        dist_list = []
        for i in range(size):
            if src_min_overlap_point_x > src_pts[i][0][0]:
                src_min_overlap_point_x = src_pts[i][0][0]
            if src_max_overlap_point_x < src_pts[i][0][0]:
                src_max_overlap_point_x = src_pts[i][0][0]
            diff_x = src_pts[i][0][0]-src_pts[seed_index][0][0]
            diff_y = src_pts[i][0][1]-src_pts[seed_index][0][1]
            dist = diff_x*diff_x + diff_y*diff_y
            dist_list.append([i,dist])

        for i in range(size_dst):
            if dst_min_overlap_point_x > dst_pts[i][0][0]:
                dst_min_overlap_point_x = dst_pts[i][0][0]
            if dst_max_overlap_point_x < dst_pts[i][0][0]:
                dst_max_overlap_point_x = dst_pts[i][0][0]
        dist_list.sort(key=lambda x: x[1])

        resize_img1 = img1[int(src_min_overlap_point_x):int(src_max_overlap_point_x),:]
        resize_img2 = img2[int(dst_min_overlap_point_x):int(dst_max_overlap_point_x),:]
        resize_img1 = cv2.resize(resize_img1, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
        resize_h,resize_w = np.shape(resize_img1)
        resize_img2 = cv2.resize(resize_img2, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
        resize_img2 = cv2.resize(resize_img2, (resize_w, resize_h) ,interpolation = cv2.INTER_CUBIC)

        for i in range(5):
        	feature_points_src.append(src_pts[dist_list[i][0]])
        	feature_points_dst.append(dst_pts[dist_list[i][0]])
        i+=1
        while quality<=0.01:
            feature_points_src.append(src_pts[dist_list[i][0]])
            feature_points_dst.append(dst_pts[dist_list[i][0]])
            feature_points_src1 = np.float32(feature_points_src)
            feature_points_dst1 = np.float32(feature_points_dst)

            feature_points_src1[:, :, 0], feature_points_src1[:, :, 1] = feature_points_src1[:, :, 1], feature_points_src1[:, :, 0].copy()
            feature_points_dst1[:, :, 0], feature_points_dst1[:, :, 1] = feature_points_dst1[:, :, 1], feature_points_dst1[:, :, 0].copy()

            H, mask = cv2.findHomography(feature_points_src1, feature_points_dst1, cv2.RANSAC,5.0)
            i+=1
            h,w = np.shape(img1)
            pts_src = np.array([[0,0,h-1,h-1], [0, w-1,0,w-1], [1,1,1,1]])
            pts_prime = MATMUL(H,pts_src)

            pts_prime1 = [[],[]]
            pts_prime1[0] = pts_prime[0]/pts_prime[2]
            pts_prime1[1] = pts_prime[1]/pts_prime[2]
            pts_prime1 = np.array(pts_prime1)
            
            x = np.array([[0,0,h-1,h-1],[0,w-1,0,w-1]])
            y = np.array(pts_prime1.reshape(2,4))
            param_init = [1,2,3,4]

            params, success = leastsq(func,param_init,args=(x,y))
            
            H_ = np.array([[params[0],-params[1],params[2]], [params[1],params[0],params[3]], [0,0,1]])
            pts_ = MATMUL(H_,pts_src)
            Hy=pts_[:2,:]
            quality = (np.sum(np.sqrt(np.sum(np.square(y-Hy),axis=0))))/(h*w)
            # print i,quality
        edges1 = cv2.Canny(resize_img1,100,200)
        warped_image = cv2.warpPerspective(resize_img2,H,(np.size(resize_img2,1),np.size(resize_img2,0)))
        edges2 = cv2.Canny(warped_image,100,200)
        edge_map = np.abs(edges1-edges2)
        # print edge_map
        align_quality = graph_cut(edge_map,src_pts)
        # print align_quality, np.mean(penalty_score)
        if best_alignment_so_far > align_quality:
            best_alignment_so_far = align_quality
            best_H = H
        if align_quality < 100:
            best_H = H
            break
        if np.mean(penalty_score) >= 2:
            best_H = H
            break
    # print feature_points_dst1
    (x,y,z)=np.shape(feature_points_dst1)
    Coefs,Dst_Points=Find_Coef(best_H,(feature_points_dst1).reshape(x,z),Size)
    print Coefs

    sz=np.array([h,w,2])
    param_init=(np.zeros(h*w*2)).tolist()
    for i in xrange(h*w*2):
        param_init[i]=i+1

    (x,y,z)=np.shape(feature_points_src1)
    (feature_points_src1)=(feature_points_src1).reshape(x,z)
    feature_points=[]
    for i in xrange(x):
        feature_points+=[feature_points_src1[i].tolist()]

    Bh=(0.0,float(h))
    Bw=(0.0,float(w))
    Bnds=[]
    for i in xrange(h*w):
    		Bnds+=[Bh]
    		Bnds+=[Bw]

    Bnds=tuple(Bnds)
    print Bnds
    Sol = minimize(Align_func,param_init,method='SLSQP',bounds=Bnds)
    print Sol
        
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None