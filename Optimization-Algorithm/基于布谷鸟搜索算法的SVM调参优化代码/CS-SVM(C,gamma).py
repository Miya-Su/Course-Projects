import numpy as np
import  math
import pandas as pd
import matplotlib.pyplot as plt

#布谷鸟搜索算法对SVM参数调优(SVM在sklearn库中主要三个参数有kernel（核函数linear、RBF），C是惩罚系数，即对误差的宽容度，c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合，C过大或过小，泛化能力变差），gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。）
#现在我们选择布谷鸟算法对SVM中的C惩罚系数和gamma值进行调优。)

#随机走动函数，寻找鸟窝点nest(更新鸟窝公式）
def get_cuckoos(nest,best,Lb,Ub):
    n=nest.shape[0]    #鸟巢个数，矩阵行数
    beta=3/2
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
   #求gamma函数值
    for j in range(n):
        s=nest[j,:]    #提取当前鸟巢的参数
        w=s.shape
        u=np.random.randn(w[0])*sigma   #生成服从N(0,sigma^2)的随机数u，u为长度为参数个数的向量
        v=np.random.randn(w[0])          #生成服从N(0,1)的随机数v向量
        step=u/abs(v)**(1/beta)        #计算步长
        stepsize=0.01*step*(s-best)    #巢穴位置变化量，如果当前巢穴为最优解，则变化量为0
        s=s+stepsize*np.random.randn(w[0])        #步长调整，更新另外一组鸟巢
        nest[j,0]=simplebounds(s,Lb,Ub)[0]        #落在边界以外的点使用simplebounds使落在定义域内
        nest[j,0]=simplebounds(s,Lb,Ub)[1]        #落在边界以外的点使用simplebounds使落在定义域内

    return nest

#找到当前最优鸟巢
def get_best_nest(nest,new_nest,fitness,X_train,Y_train,X_test,Y_test):
    w=nest.shape[0]
    for j in range(w):
        fnew=fobj(new_nest[j,:],X_train,Y_train,X_test,Y_test)      #对每个新巢穴，计算目标函数值
        if fnew <=fitness[j]:       #如果新巢穴的目标函数值优于对应旧目标的函数值
            fitness[j]=fnew       #更新当前巢穴目标函数值
            nest[j,:]=new_nest[j,:]   #更新对应的目标函数

    fmin=np.min(fitness)        #找到当前鸟巢最优函数值
    K=np.where(fitness==np.min(fitness))[0]     #找到最优函数值位置（k代表第几行，当前k个鸟巢为最优鸟巢）
    best=new_nest[K,:]    #找到最优鸟巢位置，k代表第K行的鸟巢，即前k个最优

    return fmin,best,nest,fitness,fnew,K

#构建新鸟巢来代替
def empty_nest(nest,Lb,Ub,pa):
    n=nest.shape[0]    #鸟巢个数
    K=(np.random.rand(nest.shape[0],nest.shape[1])>pa)+0     #判断鸟巢是否会被发现
    nest1= nest[np.random.permutation(n),:]   # 重新随机改变鸟巢位置，得到新的鸟巢位置
    nest2 = nest[np.random.permutation(n), :]
    stepsize=np.random.rand()*(nest1-nest2)       #计算调整步长
    new_nest=nest+stepsize*K       #新巢穴
    p=new_nest.shape[0]
    for j in range(p):       #遍历每个巢穴
        s=new_nest[j,:]      #提取当前巢穴的参数
        new_nest[j,0] = simplebounds(s, Lb, Ub)[0]
        new_nest[j,1] = simplebounds(s, Lb, Ub)[1]

    return new_nest

#落在边界以外的点使用simplebounds使落在定义域内
def simplebounds(s,Lb,Ub):
    ns_tem=s   #复制临时变量(参数）
    ns_tem=ns_tem.reshape(-1,1)
    Lb=Lb.reshape(-1,1)
    if ns_tem[0]<Lb[0]:   #判断参数是否小于下临界值
            ns_tem[0]=Lb[0]
    if ns_tem[1]<Lb[1]:   #判断参数是否小于下临界值
            ns_tem[1]=Lb[1]
    if ns_tem[0]>Ub[0]:   #判断参数是否大于下临界值
            ns_tem[0]=Ub[0]
    if ns_tem[1] > Ub[1]:  # 判断参数是否大于下临界值
            ns_tem[1] = Ub[1]
    s=ns_tem       #更新参数
    return s

#目标函数求最优函数
from sklearn import svm
def fobj(bestnest,X_train,Y_train,X_test,Y_test):
    model=svm.SVC(C=bestnest[0],gamma=bestnest[1])
    model.fit(X_train,Y_train)
    r=model.score(X_test,Y_test)
    fitness=1-r    #以预测错误率作为函数优化目标
    return fitness


# 读取数据
data=pd.DataFrame(pd.read_csv('/Users/miya/PycharmProjects/miya/Dataset/diabetes.csv'))
#处理异常值,重新赋值data
da = data.values
cont_clou = len(da)   #获取行数
#遍历数据，打印出符合条件的数据
for i in range(0,len(da)):
    if(data.values[i][4]>318):
        da[i][4] = '58'         #异常值处理，将commments大于318的数据comments设置为58
    if (data.values[i][3] > 61):
        da[i][3] = '23'
    if (data.values[i][5] > 50):
        da[i][5] = '32'
# 重新赋值，根据筛选出的值进行赋值
data['Insulin'] = data['Insulin'].replace([846,744,680],58)   #将846，744，680赋值为58（58是均值）
data['SkinThickness'] = data['SkinThickness'].replace(67.1,23) #将67.1赋值为23（均值）
data['BMI'] = data['BMI'].replace(99,32) #将99赋值为32（均值）
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']]=data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']].replace(0,np.NaN)# 将0替代为空值
data.fillna(data.mean(),inplace=True)# 将空值填充为均值
# 将数据表分解为特征值和目标值，并分割为训练集和测试集数据
X=np.array(data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']])
Y=np.array(data['Outcome'])
#随机划分,导入交叉验证库
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# SVM要求所有的特征要在相似的度量范围内变化。我们需要重新调整各特征值尺度使其基本上在同一量表上。
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


#CS-SVC
def CSSVC(time):
    #time:迭代次数
    n=20  #n为巢穴数量
    dim=2  #需要寻优的参数个数
    pa=0.25
    Lb=np.array([0.00001,0.00001]) #参数下界
    Ub=np.array([10000,10000])  #参数上界
    #随机初始化巢穴
    nest=np.zeros((n,dim))
    for i in range(n):     #遍历每个巢穴
         nest[i,:]=Lb+(Ub-Lb)*np.random.rand(1,len(Lb))        #对每个巢穴初始化参数
    #目标函数值初始化
    fitness=np.ones([n,1])

    fmin1,bestnest1,nest1,fitness1,fnew1,K1=get_best_nest(nest,nest,fitness,X_train_scaled,Y_train,X_test_scaled,Y_test)
    a=[]
    #开始迭代
    for t in range(time):
        new_nest1 = get_cuckoos(nest1,bestnest1,Lb, Ub)  # 保留当前最优解，寻找新巢穴
        fmin2, bestnest2, nest2, fitness2, fnew2, K2=get_best_nest(nest1, new_nest1, fitness1, X_train_scaled, Y_train, X_test_scaled, Y_test)
        new_nest2=empty_nest(nest2, Lb, Ub, pa)  #发现并更新劣质巢穴

        fmin3, bestnest3, nest3, fitness3, fnew3, K3 =get_best_nest(nest2, new_nest2, fitness2, X_train_scaled, Y_train, X_test_scaled, Y_test)

        if fnew3<fmin1:
            fmin1=fnew3
            bestnest1=bestnest3

        print("K",K3)
        print("fmin",fmin1)
        a.append(fmin1)
        print("c",bestnest1)
        print("time=",t)
    #画图显示迭代
    plt.plot(range(time), a, label="training_accuracy")
    plt.show()

CSSVC(time=1000)