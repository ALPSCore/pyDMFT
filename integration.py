from numpy import *
import sys

# Return \int_a^b dx f(x)
#  f_i (i=0,...n+1) must contain f(x_i): x_i = ((b-a)/n)*i+a
#  n : the number of subintervals (must be even) 
def CompositeSimpsonRules(n,a,b,f):
				if (n%2==1):
								print "CompositeSimpsonRules: the number of the subintervals must be even"
								sys.exit(-1)
				if (a>b):
								print "CompositeSimpsonRules: a must be smaller than b"
								sys.exit(-1)
				h=(b-a)/n
				r1=0.
				for i in range(2,n-1,2):
								r1+=f[i]
				r2=0.
				for i in range(1,n,2):
								r2+=f[i]
				for i in range(1,n+1):
								print i, f[i]
				return (h/3)*(f[0]+2*r1+4*r2+f[n])

# Return \int_a^b dx f(x)
#  f_i (i=0,...n+1) must contain f(x_i): x_i = ((b-a)/n)*i+a
#  n : the number of subintervals
def TrapezoidalRule(n,a,b,f):
				if (a>b):
								print "TrapezoidalRule: a must be smaller than b"
								sys.exit(-1)
				h=(b-a)/n
				r1=0.
				for i in range(1,n):
								r1+=f[i]
				return h*(0.5*(f[0]+f[n])+r1)
 
# Return \int_a^b dx f(x)
#  x: x_i in non-descreasing order
#  f: f[i] (i=0,...n) stores f(x_i).
#  n : the number of the data points
def TrapezoidalRule2(n,x,f):
				if (n<2):
								print "TrapezoidalRule2: n<2"
								sys.exit(-1)

				r1=0.
				for i in range(n-1):
								r1+=0.5*(f[i+1]+f[i])*(x[i+1]-x[i])
								if(x[i+1]<x[i]):
												print "TrapezoidalRule2: x must be given in non-decreasing order"
                                                                                                print "x[i]>x[i+1] i = ", i
                                                                                                for j in range(n):
                                                                                                    print j, x[j]
												sys.exit(-1)
				return r1
 
##################################################################
# Recursive generation of the Legendre polynomial of order n
def Legendre(n,x):
	x=array(x)
	if (n==0):
		return x*0+1.0
	elif (n==1):
		return x
	else:
		return ((2.0*n-1.0)*x*Legendre(n-1,x)-(n-1)*Legendre(n-2,x))/n
 
##################################################################
# Derivative of the Legendre polynomials
def DLegendre(n,x):
	x=array(x)
	if (n==0):
		return x*0
	elif (n==1):
		return x*0+1.0
	else:
		return (n/(x**2-1.0))*(x*Legendre(n,x)-Legendre(n-1,x))
##################################################################
# Roots of the polynomial obtained using Newton-Raphson method
def LegendreRoots(polyorder,tolerance=1e-20):
	if polyorder<2:
		err=1 # bad polyorder no roots can be found
	else:
		roots=[]
		# The polynomials are alternately even and odd functions. So we evaluate only half the number of roots. 
		for i in range(1,int(polyorder)/2 +1):
			x=cos(pi*(i-0.25)/(polyorder+0.5))
			error=10*tolerance
		        iters=0
		        while (error>tolerance) and (iters<1000):
		                dx=-Legendre(polyorder,x)/DLegendre(polyorder,x)
		                x=x+dx
		                iters=iters+1
		                error=abs(dx)
			roots.append(x)
		# Use symmetry to get the other roots
		roots=array(roots)
		if polyorder%2==0:
			roots=concatenate( (-1.0*roots, roots[::-1]) )
		else:
			roots=concatenate( (-1.0*roots, [0.0], roots[::-1]) )
		err=0 # successfully determined roots
	return [roots, err]
##################################################################
# Weight coefficients
def GaussLegendreWeights(polyorder):
	W=[]
	[xis,err]=LegendreRoots(polyorder)
	if err==0:
		W=2.0/( (1.0-xis**2)*(DLegendre(polyorder,xis)**2) )
		err=0
	else:
		err=1 # could not determine roots - so no weights
	return [W, xis, err]
