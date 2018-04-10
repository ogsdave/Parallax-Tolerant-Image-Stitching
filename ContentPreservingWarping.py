import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import pdb
# from cvxopt import matrix, solvers
# from cvxopt.modeling import variable , op, dot
from scipy.optimize import leastsq

def MATMUL(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        print "Matrices are not compatible to Multiply. Check condition C1==R2"
        return

    # Create the result matrix
    # Dimensions would be rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]
    #print C

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    C = np.matrix(C).reshape(len(A),len(B[0]))

    return C

def Find_Coef(H,feature_pts,grid):
	Dst_Point=[]
	sz=grid
	grid=[1280,720]

	grid[0]=grid[0]/100
	grid[1]=grid[1]/100

	grid[0]=1
	grid[1]=1

	print grid
	# feature_pts=[[34,67,1],[100,303,1],[54,101,1]]

	# H=[[10,0.5,20],[0.8,2,50],[1.5,2,1]]
	coef=[]

	for pts in feature_pts:
		pts=[pts[0],pts[1],1]
		Pts_Grid=np.asarray([[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1]])
		# print pts

		Pts_Grid[0][0]=(pts[0]/grid[0])*grid[0]
		Pts_Grid[0][1]=(pts[1]/grid[1])*grid[1]

		Pts_Grid[1][0]=min((pts[0]/grid[0] + 1)*grid[0],sz[0])
		Pts_Grid[1][1]=(pts[1]/grid[1])*grid[1]

		Pts_Grid[2][0]=(pts[0]/grid[0])*grid[0]
		Pts_Grid[2][1]=min((pts[1]/grid[1] + 1)*grid[1],sz[1])

		Pts_Grid[3][0]=min((pts[0]/grid[0] + 1)*grid[0],sz[0])
		Pts_Grid[3][1]=min((pts[1]/grid[1] + 1)*grid[1],sz[1])

		Dst_Point+=[[[Pts_Grid[0][0],Pts_Grid[0][1]],[Pts_Grid[1][0],Pts_Grid[1][1]],[Pts_Grid[2][0],Pts_Grid[2][1]],[Pts_Grid[3][0],Pts_Grid[3][1]]]]

		Pts_Grid=MATMUL(H,np.transpose(Pts_Grid))

		Pts_Grid[0]=Pts_Grid[0]/Pts_Grid[2]
		Pts_Grid[1]=Pts_Grid[1]/Pts_Grid[2]
		Pts_Grid[2]=Pts_Grid[2]/Pts_Grid[2]

		pts_=MATMUL(H,np.transpose([pts]))
		x=pts_[0]/pts_[2]
		y=pts_[1]/pts_[2]

		Tmp=np.asarray([[1,-1,-1,1],[-1,1,0,0],[-1,0,1,0],[1,0,0,0]])
		Pts_Grid=np.asarray(Pts_Grid[:-1][:])

		# print x,y
		# print Pts_Grid
		
		Alphabets=np.array(MATMUL(Tmp,np.transpose(Pts_Grid)))

		a=Alphabets[0][0]
		b=Alphabets[1][0]*grid[1]
		c=Alphabets[2][0]*grid[0]
		d=Alphabets[3][0]*grid[0]*grid[1]
		e=Alphabets[0][1]
		f=Alphabets[1][1]*grid[1]
		g=Alphabets[2][1]*grid[0]
		h=Alphabets[3][1]*grid[0]*grid[1]

		# print a,b,c,d,e,f,g,h

		A = a*f - b*e
		B = e*x - a*y + a*h - d*e + c*f - b*g
		C = g*x - c*y + c*h - d*g

		D = a*g - c*e
		E = e*x - a*y + a*h - d*e - c*f + b*g
		F = f*x - b*y + b*h - d*f

		# print A,B,C,D,E,F

		u=np.array([-1.0,-1.0])
		v=np.array([-1.0,-1.0])

		if(A):
			# pdb.set_trace()
			u[0]=(-B+np.sqrt(B*B-4*A*C))/(2*A)
			u[1]=(-B-np.sqrt(B*B-4*A*C))/(2*A)

			v[0]= (x - b*u[0] - d)/(a*u[0] + c)
			v[1]= (x - b*u[1] - d)/(a*u[1] + c)
		else:
			u[0]=-C/B

			v[0]= -F/E
		# print u[1],v[1],(u[1]>=0 and v[1]>=0 and u[1]<=1 and v[1]<=1)
		if(u[0]>=0 and v[0]>=0 and u[0]<=1 and v[0]<=1):
			coef+=[[(grid[0]-u[0])*(grid[1]-v[0]),(grid[0]-u[0])*(v[0]),(u[0])*(grid[1]-v[0]),(u[0])*(v[0])]]
		elif(u[1]>=0 and v[1]>=0 and u[1]<=1 and v[1]<=1):
			coef+=[[(grid[1]-u[1])*(grid[1]-v[1]),(grid[0]-u[1])*(v[1]),(u[1])*(grid[1]-v[1]),(u[1])*(v[1])]]
		else: coef+=[[-1,-1,-1,-1]]
	return coef,Dst_Point