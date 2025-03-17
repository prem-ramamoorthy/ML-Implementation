import matplotlib.pyplot as plt

def linearModel(x ,y , xpred):
    xsquare = [i*i for i in x]
    xandy = [x[i]*y[i] for i in range(len(x))]
    xmean = sum(x) / len(x)
    ymean = sum(y)/len(y ) 
    xandymean = sum(xandy) / len(xandy)
    xmeansquare = xmean * xmean 
    xsquaremean = sum(xsquare) / len(xsquare)
    ymean = sum(y)/len(y)
    a1 = (xandymean - xmean*ymean )/(xsquaremean - xmeansquare)
    a0 = ymean - a1*xmean
    plt.scatter(x, y, color="blue", label="Data points")
    plt.plot(x, [a0 + a1 * xi for xi in x], color="red", label="Regression Line")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.title("Linear Regression Model")
    plt.legend()
    plt.show()
    return a0 + a1* xpred 

x = [1,2,3,4,5]
y = [1.2 , 1.8 ,2.6 ,3.2 ,3.8]
xpred = 12
print(linearModel(x,y,xpred))

    

    