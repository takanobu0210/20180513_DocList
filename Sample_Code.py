
# coding: utf-8

# In[1]:


print("Hello, Scikit-learn !")#画面に出力


# In[2]:


for i in range(5):
    print(i)#0~5まで出力


# In[3]:


x =5 #xの値を書き換えてみましょう
if x > 5:#左の条件式を満たす場合に、１行下の文が実行
    print("xは5より大きいです。")
elif x <5:#上記の条件を満たさなかった場合に、1行下の文が実行
    print("xは5より小さいです。")
else:#上記2つの条件を共に満たさなかった場合に、1行下の文が実行
    print("xと5は等しいです。")


# In[4]:


print("a")


# In[5]:


def mul(x, y):#掛け算する関数
    return x*y

print(mul(3, 5))


# In[6]:


import numpy as np #numpyというライブラリを導入し, npという名前をつける.
import pandas as pd #表計算、ファイル読み込み、出力するためのライブラリ
import matplotlib.pyplot as plt #可視化するためのライブラリ


print(np)#numpyが読み込めている.
print(dir(np))#numpyのモジュール一覧確認.
print(help(np.array))#np.arrayという関数の仕様を確認.


# In[7]:


import pandas as pd

csv_data = pd.read_csv("house.csv")#住宅データを読み込む

print(csv_data)#csv_dataすべてを出力
print(csv_data["price"])#価格だけを出力
print(csv_data["price"][0:100])#100行目までの価格を出力
print(csv_data[["price", "sqft_living"]])#価格と家の面積を出力

name_age = csv_data[["price", "sqft_living"]]#価格と家の面積を、name_ageに格納

name_age.to_csv("price_sqft_living.csv")#ファイルの出力


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd

csv_data = pd.read_csv("house.csv")#住宅データを読み込む

#散布図を生成する関数
plt.scatter(csv_data["sqft_living"],csv_data["price"], s=1)
plt.xlabel("sqft_living" ,size = 20)#x軸のタイトル
plt.ylabel("price", size=20)#y軸のタイトル
plt.show()#画面に出力


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csv_data = pd.read_csv("house.csv")#住宅データを読み込む

#正規化する関数
def min_max(x):
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

#散布図を生成する関数
plt.scatter( min_max(csv_data["sqft_living"]), min_max(csv_data["price"]), s=1)
plt.xlabel("sqft_living" ,size = 20)#x軸のタイトル
plt.ylabel("price", size=20)#y軸のタイトル
plt.show()#画面に出力


# In[11]:


#lasso回帰の例(単回帰)
from sklearn.linear_model import Lasso#lasso回帰を用いる.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

array_X = np.zeros((1500,1), float)#1500行目までの説明変数を格納する.#Lassoに2次元配列として与える必要があるため
array_Y = np.zeros((1500,1), float)#1500行目までの目的変数を格納する.

array_X[:,0] =  csv_data["sqft_living"][0:1500]#説明変数.
array_Y[:,0]=  csv_data["price"][0:1500]#目的変数.

# alphaの値を0に近づければ近づけるほど、二乗誤差のみの計算となる
clf = Lasso(alpha=0.2,max_iter=100)#ハイパーパラメータ設定

clf.fit(array_X, array_Y)#家の大きさ, 売却価格の関係を学習.

array_test = np.zeros((10000,1),float)
array_test[:,0] = range(10000)#予測する家の大きさ.

p = clf.predict(array_test)#学習済みのlassoモデルを用いて予測.

plt.scatter(array_X,  array_Y,  s=1, c ="r")

plt.xlabel("sqft_living", size=20 )
plt.ylabel("price", size=20)
plt.plot(p)
plt.show()
print(p[4000])#家の大きさが4000平方フィートの時の推定価格を表示.


# In[12]:


#lasso回帰例(重回帰)
from sklearn.linear_model import Lasso
import numpy as np
from mpl_toolkits.mplot3d import Axes3D#3次元のグラフを描画するために読み込み

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

array_X = np.zeros((1500,2), float)#1500行目までの説明変数を格納する.重回帰分析なので１５００行2列になる
array_Y = np.zeros((1500,1), float)#1500行目までの目的変数を格納する.

array_X[:,:] =  csv_data[["sqft_living", "grade"]][0:1500]#説明変数.家の大きさとグレード
array_Y[:,0] =  csv_data["price"][0:1500]#目的変数.

clf = Lasso(alpha=0.2,max_iter=100)

clf.fit(array_X, array_Y)#家の大きさとgrade, 売却価格の関係を学習.

array_test = np.zeros((12,2),float)
array_test[:,0] = range(0, 12000,1000)#最高気温の軸を格納
array_test[:,1] = range(12)#最低気温の軸を格納

p = clf.predict(array_test)#学習済みのlassoモデルを用いて予測

a1, a2 = clf.coef_ #係数を取得
b = clf.intercept_#切片を取得

fig = plt.figure()#3次元のグラフを描画するメソッド
ax = Axes3D(fig)
ax.scatter3D(np.ravel(array_X[:,0]), np.ravel(array_X[:,1]), np.ravel(array_Y), c = 'red')

X, Y = np.meshgrid(np.arange(0, 10000, 1), np.arange(0, 20, 1))#平面の範囲を規定
Z = a1 * X + a2 * Y + b #平面を決める式
ax.plot_surface(X, Y, Z, alpha = 0.7) #alphaで透明度を指定
ax.set_xlabel("sqft_living", size = 20)#x軸名をつける
ax.set_ylabel("grade", size = 20)
ax.set_zlabel("price", size =20)
plt.show()


# In[13]:


#SVM
#グレードを予測する
from sklearn import svm#サポートベクターマシーン読み込み
import pandas as pd
import numpy as np

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.
#print(titanic_csv.columns)

train_X = np.zeros((1500,2), float)#教師データ(1500行)の説明変数を格納
train_Y= np.zeros((1500,1), float)#教師データ(1500行)の目的変数を格納

# 学習用データ
train_X[:,0] = csv_data["price"][0:1500]
train_X[:,1] = csv_data["sqft_living"][0:1500]
train_Y[:,0]  = csv_data["grade"][0:1500]

# 検証用データ
test_X = np.zeros((500,2),float)#500行のテストデータを格納
test_X[:,0] =  csv_data["price"][1500:]
test_X[:,1] =  csv_data["sqft_living"][1500:]

answer =  np.zeros((500,1),float)#答えを格納
answer[:,0] =  csv_data["grade"][1500:]

# モデルの作成
clf = svm.SVC()#分類器の生成

clf.fit(train_X, train_Y)#svmを用いて学習
test_pred = clf.predict(test_X)#学習済み分類器を用いて予測

#正答率を調べる.
count = 0
num = 500
for i in range(num):
    if answer[i] == test_pred[i]:
        count = count+1
print("正答率は"+str(count * 100 /num)+"%です。")


# In[14]:


#SVM(問2(1))
from sklearn import svm#サポートベクターマシーン読み込み
import pandas as pd
import numpy as np

#正規化する関数
def min_max(x):
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

csv_data ["price"] =  min_max(csv_data ["price"])#データの正規化
csv_data ["sqft_living"] =  min_max(csv_data ["sqft_living"])

train_X = np.zeros((1500,2), float)#教師データ(1500行)の説明変数を格納
train_Y= np.zeros((1500,1), float)#教師データ(1500行)の目的変数を格納

train_X[:,0] = csv_data["price"][0:1500]
train_X[:,1] = csv_data["sqft_living"][0:1500]
train_Y[:,0]  = csv_data["grade"][0:1500]

test_X = np.zeros((500,2),float)#500行のテストデータを格納
test_X[:,0] =  csv_data["price"][1500:]
test_X[:,1] =  csv_data["sqft_living"][1500:]

answer =  np.zeros((500,1),float)#答えを格納
answer[:,0] =  csv_data["grade"][1500:]

clf = svm.SVC()#分類器の生成

clf.fit(train_X, train_Y)#svmを用いて学習
test_pred = clf.predict(test_X)#学習済み分類器を用いて予測

#正答率を調べる.
count = 0
num = 500
for i in range(num):
    if answer[i] == test_pred[i]:
        count = count+1
print("正答率は"+str(count * 100 /num)+"%です。")


# In[15]:


#SVM(問2(2))
from sklearn import svm#サポートベクターマシーン読み込み
import pandas as pd
import numpy as np

#正規化する関数
def min_max(x):
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

csv_data ["grade"] =  min_max(csv_data ["grade"])#データの正規化
csv_data ["sqft_living"] =  min_max(csv_data ["sqft_living"])

train_X = np.zeros((1500,2), float)#教師データ(1500行)の説明変数を格納
train_Y= np.zeros((1500,1), float)#教師データ(1500行)の目的変数を格納

train_X[:,0] = csv_data["grade"][0:1500]
train_X[:,1] = csv_data["sqft_living"][0:1500]
train_Y[:,0]  = csv_data["condition"][0:1500]

test_X = np.zeros((500,2),float)#500行のテストデータを格納
test_X[:,0] =  csv_data["grade"][1500:]
test_X[:,1] =  csv_data["sqft_living"][1500:]

answer =  np.zeros((500,1),float)#答えを格納
answer[:,0] =  csv_data["condition"][1500:]

clf = svm.SVC()#分類器の生成

clf.fit(train_X, train_Y)#svmを用いて学習
test_pred = clf.predict(test_X)#学習済み分類器を用いて予測

#正答率を調べる.
count = 0
num = 500
for i in range(num):
    if answer[i] == test_pred[i]:
        count = count+1
print("正答率は"+str(count * 100 /num)+"%です。")


# In[16]:


from sklearn.cluster import KMeans #K平均法.

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

train_X = np.zeros((1500,2), float)#教師データ(1500行)の説明変数を格納.
train_X[:,:] = csv_data[["sqft_living","price"]][0:1500]


#K平均法を用いて4クラスに分類.
pred = KMeans(n_clusters=4).fit_predict(train_X)

class_1 = train_X [pred == 0]#クラス1の要素を取得.
class_2 =  train_X [pred == 1]#クラス2の要素取得.
class_3 =  train_X [pred == 2]#クラス3の要素取得.
class_4 = train_X [pred == 3]#クラス4の要素取得.


plt.scatter(class_1[:,0], class_1[:,1], c = "r", s =1, label = "class1")#クラス1に赤点.
plt.scatter(class_2[:,0], class_2[:,1], c = "g", s = 1,label = "class2")#クラス2に緑点.
plt.scatter(class_3[:,0], class_3[:,1], c = "b", s = 1,label = "class3")#クラス3に青点.
plt.scatter(class_4[:,0], class_4[:,1], c = "y", s= 1,label = "class4")#クラス4に黄点.
plt.xlabel("sqft_living", size=20)
plt.ylabel("price", size=20)
plt.legend() # 凡例を表示
plt.show()


# In[17]:


#正規化したversion
from sklearn.cluster import KMeans #K平均法.

#正規化する関数
def min_max(x):
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

train_X = np.zeros((1500,2), float)#教師データ(1500行)の説明変数を格納.
train_X[:,0] = min_max(csv_data["sqft_living"][0:1500])
train_X[:,1] = min_max(csv_data["price"][0:1500])


#K平均法を用いて4クラスに分類.
pred = KMeans(n_clusters=4).fit_predict(train_X)

class_1 = train_X [pred == 0]#クラス1の要素を取得.
class_2 =  train_X [pred == 1]#クラス2の要素取得.
class_3 =  train_X [pred == 2]#クラス3の要素取得.
class_4 = train_X [pred == 3]#クラス4の要素取得.


plt.scatter(class_1[:,0], class_1[:,1], c = "r", s =1, label = "class1")#クラス1に赤点.
plt.scatter(class_2[:,0], class_2[:,1], c = "g", s = 1,label = "class2")#クラス2に緑点.
plt.scatter(class_3[:,0], class_3[:,1], c = "b", s = 1,label = "class3")#クラス3に青点.
plt.scatter(class_4[:,0], class_4[:,1], c = "y", s= 1,label = "class4")#クラス4に黄点.
plt.xlabel("sqft_living", size=20)
plt.ylabel("price", size=20)
plt.legend() # 凡例を表示
plt.show()


# In[18]:



from sklearn.cluster import KMeans #K平均法(3次元用　
import numpy as np
from mpl_toolkits.mplot3d import Axes3D#3次元のグラフを描画するために読み込み

#正規化する関数
def min_max(x):
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

train_X = np.zeros((1500,3), float)#教師データ(1500行)の説明変数を格納.

train_X[:,0] = min_max(csv_data["sqft_living"][0:1500])
train_X[:,1] = min_max(csv_data["price"][0:1500])
train_X[:,2] = min_max(csv_data["grade"][0:1500])

#K平均法を用いて4クラスに分類
pred = KMeans(n_clusters=4).fit_predict(train_X)

class_1 = train_X[pred == 0]#クラス1の要素を取得
class_2 =  train_X[pred == 1]#クラス2の要素を取得
class_3 =  train_X[pred == 2]#クラス3の要素を取得
class_4 =  train_X[pred == 3]#クラス3の要素を取得

fig = plt.figure()#3次元のグラフを描画するメソッド
ax = Axes3D(fig)
ax.scatter3D(class_1[:,0], class_1[:,1], class_1[:,2], c = 'r', s=5, label= "class1")
ax.scatter3D(class_2[:,0], class_2[:,1], class_2[:,2], c = 'b', s=5, label = "class2")
ax.scatter3D(class_3[:,0], class_3[:,1], class_3[:,2], c = 'g', s=5, label = "class3")
ax.scatter3D(class_4[:,0], class_4[:,1], class_4[:,2], c = 'y', s=5, label = "class4")
ax.set_xlabel("sqft_living", size = 20)#x軸名をつける
ax.set_ylabel("grade", size = 20)
ax.set_zlabel("price", size =20)
ax.legend()
plt.show()


# In[19]:


#PCA
#多次元のデータを2次元に落とし込む
from sklearn.decomposition import PCA

#2次元のデータフレームを正規化する関数
def min_max_df(x):
    result_array = np.zeros((x.shape), float)
    for i in range(x.shape[1]):
        min = np.min(x.iloc[:, i])
        max = np.max(x.iloc[:, i])
        result_array[:,i] = (x.iloc[:, i]-min)/(max-min)
    return result_array

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

csv_array = min_max_df(csv_data[["sqft_living","sqft_lot","price",
                                 "condition","bedrooms","bathrooms","floors"]] )


pca = PCA(n_components=2)#何次元に次元削減するか設定

pca.fit(csv_array) #csv_arrayに対して主成分分析

PCA_array = pca.transform(csv_array)#各軸を主成分とした空間に射影

components = pca.components_#主成分を格納

Contribution_rate = pca.explained_variance_ratio_

plt.scatter(PCA_array[:,0], PCA_array[:,1], s=1)
plt.xlabel("first components",size = 20)
plt.ylabel("second components", size = 20)
plt.show()

print("固有ベクトルは"+str(components)+"です。")
print("各次元の寄与率は"+str(Contribution_rate)+"です。")


# In[20]:


from sklearn.decomposition import PCA

#正規化する関数
def min_max_df(x):
    result_array = np.zeros((x.shape), float)
    for i in range(x.shape[1]):
        min = np.min(x.iloc[:, i])
        max = np.max(x.iloc[:, i])
        result_array[:,i] = (x.iloc[:, i]-min)/(max-min)
    return result_array

#正規化する関数
def min_max(x):
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

csv_data = pd.read_csv("house.csv")#住宅データを読み込む.

csv_array = min_max_df(csv_data[["sqft_living","sqft_lot","grade","condition","bedrooms","bathrooms","floors"]] )

price_array = min_max(csv_data["price"])

pca = PCA(n_components=2)#何次元に次元削減するか設定

ans = pca.fit(csv_array) #csv_arrayを適用

components = ans.components_#主成分を格納

PCA_array = ans.transform(csv_array)#pca.transformで、入力データを各軸を主成分とした空間に射影

class_1 =PCA_array[price_array < 0.1] #クラス1の要素を取得.
class_2 = PCA_array[(price_array >=  0.1)*(price_array <=0.2)]#クラス2の要素取得.
class_3=PCA_array[price_array > 0.2] #クラス3の要素を取得.

plt.scatter(class_1[:,0], class_1[:,1], c = "r", s =1, label = "Low price")#クラス1に赤点.
plt.scatter(class_2[:,0], class_2[:,1], c = "g", s = 1,label = "Normal price")#クラス2に緑点.
plt.scatter(class_3[:,0], class_3[:,1], c = "b", s = 1,label = "High price")#クラス2に緑点.
plt.legend()
plt.xlabel("first components", size = 20)
plt.ylabel("second components", size = 20)
plt.show()
