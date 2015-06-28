"""	Optimization and differential system of equations problem.

	The main model with the 36 equations.	
	
	Implementation idea based on http://wiki.scipy.org/Cookbook/Zombie_Apocalypse_ODEINT

	Author: MiLia , milia@protonmail.com
"""
#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import csv

# PART 1: Solving the system.

# Defining the main subroutine of the program, depends on R,s,a,T and returns the square(abs(det)).

def tmp(x,R,n,m,T):
	s, a = x[0], x[1]

        h = m*((1.-n**2)/(4.*n**2))**(0.5)
        c = (-n**2)/(1-n**2)

	# Defining some functions
	w = lambda t: t*(1.-n)+n

	g = lambda t: ((1.-w(t)**2)*n**2)/((1.-n**2)*w(t)**2)

	k = lambda t: (1.-n)/w(t)

	n1 = lambda t: 4.*k(t)**2*h**2*c-a**2

	n2 = lambda t: -(s+h*np.sqrt(T)*g(t)+a*R*np.log(w(t))/np.log(n))

	q = lambda t: 4.*c*k(t)**2*h*np.sqrt(T)
	
	b = lambda t: 2.*c*k(t)*h*np.sqrt(T)

	# Constructing the time grid [a1,b1]:
	alpha = 0.
	beta  = 1.
	Nsteps = 1001.
	t= np.linspace(alpha,beta,Nsteps) 	
	# Calculating the first derivatives of the functions y at time t = 0.
	# At the same time we setup our system, which is of the form dy/dx = f(y(t),g(t),k(t),n1(t),n2(t),q(t),b(t),t)
	#@profile	
	def fun(y,t):
		# Assigning the values of the vector y to the values of y at time zero: y{i}(0) = y{i} = y[i-1]
		#
		(y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,y28,y29,
		y30,y31,y32,y33,y34,y35,y36) = (y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9],y[10],y[11],y[12],y[13],y[14],y[15],
		y[16],y[17],y[18],y[19],y[20],y[21],y[22],y[23],y[24],y[25],y[26],y[27],y[28],y[29],y[30],y[31],y[32],y[33],y[34],y[35]) 	
		# The equations of the model. All the f{i} are the derivatives of y{i}(t) at t = 0.
		#
		f1 = -k(t)*y1 + b(t)*y10 + a*y16 
		f2 = -k(t)*y2 + b(t)*y11 + a*y17
		f3 = -k(t)*y3 + b(t)*y12 + a*y18
		f4 = -k(t)*y4 - b(t)*y7  - a*y13   
		f5 = -k(t)*y5 - b(t)*y8  - a*y14 
		f6 = -k(t)*y6 - b(t)*y9  - a*y15

		f7 =  y25 - k(t)*y7
		f8 =  y26 - k(t)*y8
		f9 =  y27 - k(t)*y9
		f10 = y28 - k(t)*y10
		f11 = y29 - k(t)*y11
		f12 = y30 - k(t)*y12

		f13 = y31
		f14 = y32
		f15 = y33
		f16 = y34
		f17 = y35
		f18 = y36

		f19 = -n1(t)*y1 + n2(t)*y4 + T*g(t)*y7 - q(t)*y10 
		f20 = -n1(t)*y2 + n2(t)*y5 + T*g(t)*y8 - q(t)*y11
		f21 = -n1(t)*y3 + n2(t)*y6 + T*g(t)*y9 - q(t)*y12
		f22 = -n1(t)*y4 - n2(t)*y1 + T*g(t)*y10 + q(t)*y7
		f23 = -n1(t)*y5 - n2(t)*y2 + T*g(t)*y11 + q(t)*y8
		f24 = -n1(t)*y6 - n2(t)*y3 + T*g(t)*y12 + q(t)*y9

		f25 = (-n1(t)*y7 + n2(t)*y10 + y1 - ((2*k(t)*h*b(t))/np.sqrt(T))*y7 - ((2*k(t)*h*a)/np.sqrt(T))*y13 -((2*k(t)*h)/np.sqrt(T))*y22
			-((4*k(t)**2*h)/np.sqrt(T))*y4)
	
		f26 = (-n1(t)*y8 + n2(t)*y11 + y2 - ((2*k(t)*h*b(t))/np.sqrt(T))*y8 - ((2*k(t)*h*a)/np.sqrt(T))*y14 -((2*k(t)*h)/np.sqrt(T))*y23
			-((4*k(t)**2*h)/np.sqrt(T))*y5)
	
		f27 = (-n1(t)*y9 + n2(t)*y12 + y3 - ((2*k(t)*h*b(t))/np.sqrt(T))*y9 - ((2*k(t)*h*a)/np.sqrt(T))*y15 -((2*k(t)*h)/np.sqrt(T))*y24
			-((4*k(t)**2*h)/np.sqrt(T))*y6)
	
		f28 = (-n1(t)*y10 - n2(t)*y7 + y4 - ((2*k(t)*h*b(t))/np.sqrt(T))*y10 - ((2*k(t)*h*a)/np.sqrt(T))*y16 +((2*k(t)*h)/np.sqrt(T))*y19
			+((4*k(t)**2*h)/np.sqrt(T))*y1)

		f29 = (-n1(t)*y11 - n2(t)*y8 + y5 - ((2*k(t)*h*b(t))/np.sqrt(T))*y11 - ((2*k(t)*h*a)/np.sqrt(T))*y17 +((2*k(t)*h)/np.sqrt(T))*y20
			+((4*k(t)**2*h)/np.sqrt(T))*y2)	

		f30 = (-n1(t)*y12 - n2(t)*y9 + y6 - ((2*k(t)*h*b(t))/np.sqrt(T))*y12 - ((2*k(t)*h*a)/np.sqrt(T))*y18 +((2*k(t)*h)/np.sqrt(T))*y21
			+((4.*k(t)**2*h)/np.sqrt(T))*y3)		

		f31 = -k(t)*y31 - n1(t)*y13 + n2(t)*y16 + a*b(t)*y7  + a**2*y13 + a*y22 + ((k(t)*R)/np.log(n))*y1
		f32 = -k(t)*y32 - n1(t)*y14 + n2(t)*y17 + a*b(t)*y8  + a**2*y14 + a*y23 + ((k(t)*R)/np.log(n))*y2
		f33 = -k(t)*y33 - n1(t)*y15 + n2(t)*y18 + a*b(t)*y9  + a**2*y15 + a*y24 + ((k(t)*R)/np.log(n))*y3
		f34 = -k(t)*y34 - n1(t)*y16 - n2(t)*y13 + a*b(t)*y10 + a**2*y16 - a*y19 + ((k(t)*R)/np.log(n))*y4
		f35 = -k(t)*y35 - n1(t)*y17 - n2(t)*y14 + a*b(t)*y11 + a**2*y17 - a*y20 + ((k(t)*R)/np.log(n))*y5
		f36 = -k(t)*y36 - n1(t)*y18 - n2(t)*y15 + a*b(t)*y12 + a**2*y18 - a*y21 + ((k(t)*R)/np.log(n))*y6
				
		return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,
			f27,f28,f29,f30,f31,f32,f33,f34,f35,f36])
  
	# Initial Conditions
	y19o,y22o,y26o,y29o,y33o,y36o = np.ones(6)

	(y1o,y2o,y3o,y4o,y5o,y6o,y7o,y8o,y9o,y10o,y11o,y12o,y13o,y14o,y15o,y16o,y17o,y18o,y20o,y21o,y23o,y24o,y25o,y27o,y28o,
	y30o,y31o,y32o,y34o,y35o) = np.zeros(30)

	y0 = np.array([y1o,y2o,y3o,y4o,y5o,y6o,y7o,y8o,y9o,y10o,y11o,y12o,y13o,y14o,y15o,y16o,y17o,y18o,y19o,y20o,y21o,y22o,y23o,y24o,y25o,y26o,
 	y27o,y28o,y29o,y30o,y31o,y32o,y33o,y34o,y35o,y36o]) # initial condition vector

	# Solve the ODEs
	# Information for odeint(): http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html.

	soln = odeint(fun,y0,t)

	y1final = soln[:,0]
	y2final = soln[:,1]
	y3final = soln[:,2]
	y4final = soln[:,3]
	y5final = soln[:,4]
	y6final = soln[:,5]
	y7final = soln[:,6]
	y8final = soln[:,7]
	y9final = soln[:,8]
	y10final = soln[:,9]
	y11final = soln[:,10]
	y12final = soln[:,11]
	y13final = soln[:,12]
	y14final = soln[:,13]
	y15final = soln[:,14]
	y16final = soln[:,15]
	y17final = soln[:,16]
	y18final = soln[:,17]
	y19final = soln[:,18]
	y20final = soln[:,19]
	y21final = soln[:,20]
	y22final = soln[:,21]
	y23final = soln[:,22]
	y24final = soln[:,23]
	y25final = soln[:,24]
	y26final = soln[:,25]
	y27final = soln[:,26]
	y28final = soln[:,27]
	y29final = soln[:,28]
	y30final = soln[:,29]
	y31final = soln[:,30]
	y32final = soln[:,31]
	y33final = soln[:,32]
	y34final = soln[:,33]
	y35final = soln[:,34]
	y36final = soln[:,35]
	
	#
	# PART 2: Calculating the determinant
	#	  Remember that all matrices in python start from 0. Thus the last element of a 1000 member array will be array[999]
	#
	M = np.array([
	       [y1final[Nsteps-1] +  1j*y4final[Nsteps-1],  y2final[Nsteps-1] +  1j*y5final[Nsteps-1],   y3final[Nsteps-1] + 1j*y6final[Nsteps-1]],
	       [y7final[Nsteps-1] + 1j*y10final[Nsteps-1],  y8final[Nsteps-1] + 1j*y11final[Nsteps-1],  y9final[Nsteps-1] + 1j*y12final[Nsteps-1]],
	      [y13final[Nsteps-1] + 1j*y16final[Nsteps-1], y14final[Nsteps-1] + 1j*y17final[Nsteps-1], y15final[Nsteps-1] + 1j*y18final[Nsteps-1]]
	])

	# Straightforward method
	det = np.linalg.det(M)

	# Calculating the rest:
	b1 = np.abs(det)
	d1 = b1**2

	# Exiting the function temp()
	return d1

#--------------------------------------------------------------------------------------------------------------------------------------------
# PART 3: Exploring the minimum value of tmp() while looking for the minimum value of T
#	  at the same time.
#--------------------------------------------------------------------------------------------------------------------------------------------

# Creating a function minT which tries to find the minimum T through a bisection method.
# This function prints the results in csv files as well.
def minTcsv(n,m,R,Tmin,Tmax,myfile):
	csvout = csv.writer(open(myfile, "wb"))
	csvout.writerow(("m=",m,"n=",n,"R=",R))
	csvout.writerow(("  "))
	csvout.writerow(("Tmin", "Tmax","s","a","tmp"))
	a,b= Tmin, Tmax
	while (abs(a-b)>1):
		c=int(((a+b)/2.)//1)#getting the integer part of the number (a+b)/2: Let number a. Then integer part of a is int(a//1).
		T=c
		sol = minimize(tmp,[0,3],args=(R,n,m,T),bounds=((-150,0),(1.5,6)),tol=1.e-9)
		if sol.fun>1.e-9:
			a=c
		else:
			b=c
		print a," ",b," ", sol.x, sol.fun
		csvout.writerow((a,b,sol.x[[0]],sol.x[[1]],sol.fun))
	csvout.writerow(("  "))
	csvout.writerow(("Tmin= ",c,"s=",sol.x[[0]],"a=",sol.x[[1]],"tmp=",sol.fun))	
	return c,sol

# The same function as above, only without printing the results in csv.
def minTsimple(n,m,R,Tmin,Tmax):
        a,b= Tmin, Tmax
        while (abs(a-b)>1):
                c=int(((a+b)/2.)//1)#getting the integer part of the number (a+b)/2: Let number a. Then integer part of a is int(a//1).
                T=c
                sol = minimize(tmp,[0,3],args=(R,n,m,T),bounds=((-150,0),(1.5,6)),tol=1.e-9)
                if sol.fun>1.e-9:
                        a=c
                else:
                        b=c
                print a," ",b," ", sol.x, sol.fun
        return c,sol

# A function which will help us create the csv file names iteratevly
def printname(x,m,y,n,z,R):
	name = x + str(m) + y + str(n) +  z + str(R)  + ".csv"
	return name

# A function to merge files 
def mergefiles():
	fout=open("final.csv","a")
	for i in [0.0,1.0,2.0,3.0]: # m
        	for k in [0.9, 0.8, 0.75, 0.6, 0.5, 0.4, 0.3,0.2, 0.1]:# n
                	for j in [0.,5.,10.,30.,70.,80.,100.]: # R      
                        	f = open(printname("minTm",i,"n",k,"R",j))
                        	fout.write("n="+str(k)+"\n")
                        	for line in f:
                                	fout.write(line)
                        	fout.write('\n')
                        	f.close() # not really needed
	fout.close()
	return

# The main function
def main():
	# A for loop which will print us all the files.
	for i in [0,1,2,3]: # m
		for k in [0.9, 0.8, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:# n
			for j in [0,5,10,30,70,80,100]: # R
				print "Writing for: m=",i,",n=",k," and R=",j
				print minTcsv(k,i,j,0,10000,printname("minTm",i,"n",k,"R",j)) #Tmax set to 10k.
	mergefiles()


# Calling the main function.
if __name__ == "__main__":
	main()
