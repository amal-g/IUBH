import unittest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import csv
from collections import defaultdict
import numpy as np 
import sqlite3 
from sqlite3 import Error
import glob
import itertools
from pandas.plotting import scatter_matrix 
from numpy import sin
from numpy import sqrt
from numpy import arange
from scipy.optimize import curve_fit


class ArithmeticFunctions():

	def func(self, x, y):
		return x + y


	def f_model(self, x, y):
		return x * y

#def f_model(x, a, c):
#       return pd.np.log((a + x)**2 / (x - c)**2)

#def func(x, a, b, c, d):
#       return a + b*x - c*np.exp(-d*x)	

arithmetic_object = ArithmeticFunctions()


# Test class
class TestClass(unittest.TestCase):
	"""TestClass used to Test arithmetic operations"""

	def test_func(self):
		try:
		 self.assertTrue(arithmetic_object.func (2, 3) == 5)
		except:
		 raise Exception("error genereted")



def main():	
 
  conn = None
  try:	

   print("*******************TRAIN DATASET************************")

   path =r'C:\Users\amel\Desktop\IUBH\Codes\PythonAssignement'
   filenames = glob.glob(path + "/*.csv")
   dfs = []
   fname_train="train.csv"
   fname_ideal="ideal.csv"
   fname_test="test.csv"

   for filename in filenames:
       #print(filename)
       if fname_train in filename:
          df_train = pd.read_csv(filename, index_col=None, header=0)
          print(df_train)
          print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
          print('Mean:', np.mean(df_train))
          print('Standard Deviation:', np.std(df_train))

       if fname_ideal in filename:
          df_ideal = pd.read_csv(filename, index_col=None, header=0)
          print(df_ideal )
       if fname_test in filename:
          df_test = pd.read_csv(filename, index_col=None, header=0)
          print(df_test)

   print("creating dataframe for train dataset")
   df_train = pd.DataFrame(df_train,columns= ['x','y1','y2','y3','y4'])
   
   for lines in df_train.itertuples():
    linesX=lines.x
    linesY1=lines.y1
    linesY2=lines.y2
    linesY3=lines.y3
    linesY4=lines.y4
    
   print("creating a new database")
   conn = sqlite3.connect("AssignDB.db")
   print("Connection is open")
   print("reading database created")
   conn = sqlite3.connect(r"AssignDB.db",uri=True)

   df_train = pd.read_csv(r'train.csv')
   df_train.to_sql('train_table', conn, if_exists='append', index=False)
   c = conn.cursor()
   c_ideal = conn.cursor()
   c_test = conn.cursor()
   c_result=conn.cursor()

   
   print("creating first table train")
   #c.execute('''CREATE TABLE train_table(x, y1,y2,y3,y4)''')
   print("insert train dataset into table")
   #c.execute("INSERT INTO train_table (x, y1,y2,y3,y4) VALUES(?,?,?,?,?)", (linesX, linesY1, linesY2, linesY3, linesY4)) 
   #for row in c.execute('SELECT * FROM train_table'):
   # print("result view after insert")
   # print(row)   
   # print()
    
   print("*******************TEST DATASET************************")

   print("reading CSV test dataset")
   data_test = pd.read_csv (r'test.csv')
   print(data_test)
   print("creating dataframe TEST")
   df_test = pd.DataFrame(data_test,columns= ['x','y'])
   for lines_test in df_test.itertuples():
     linesX=lines_test.x
     linesY=lines_test.y
    
     #c_test.execute('''CREATE TABLE test_table(x, y)''')
     df_test = pd.read_csv(r'test.csv')
     df_test.to_sql('test_table', conn, if_exists='append', index=False)
     
     #c_test.execute("INSERT INTO test_table (x, y) VALUES(?,?)", (linesX, linesY)) 
     #for row_test in c_test.execute('SELECT * FROM test_table'):
     #   print("result view after insert")
     #   print(row_test)      
     #   print()  

   print("*******************IDEAL DATASET************************")
  
   print("reading CSV ideal dataset")
   data_ideal = pd.read_csv (r'ideal.csv')
   print(data_ideal)
   print("creating dataframe ideal")
   df_ideal = pd.DataFrame(data_ideal,columns= ['x','y1','y2','y3','y4','y5','y6','y7','y8','y9','y10','y11','y12','y13','y14','y15','y16',
                                                 'y17','y18','y19','y20','y21','y22','y23','y24','y25','y26','y27','y28','y29','y30','y31','y32',
                                                 'y33','y34','y35','y36','y37','y38','y39','y40','y41','y42','y43','y44','y45','y46','y47','y48',
                                                 'y49','y50'])
  
   for lines_ideal in df_ideal.itertuples():
     #print(linesX)
     linesX=lines_ideal.x
     linesY1=lines_ideal.y1
     linesY2=lines_ideal.y2
     linesY3=lines_ideal.y3
     linesY4=lines_ideal.y4
     linesY5=lines_ideal.y5
     linesY6=lines_ideal.y6
     linesY7=lines_ideal.y7
     linesY8=lines_ideal.y8
     linesY9=lines_ideal.y9
     linesY10=lines_ideal.y10
     linesY11=lines_ideal.y11
     linesY12=lines_ideal.y12
     linesY13=lines_ideal.y13
     linesY14=lines_ideal.y14
     linesY15=lines_ideal.y15
     linesY16=lines_ideal.y16
     linesY17=lines_ideal.y17
     linesY18=lines_ideal.y18
     linesY19=lines_ideal.y19
     linesY20=lines_ideal.y20
     linesY21=lines_ideal.y21
     linesY22=lines_ideal.y22
     linesY23=lines_ideal.y23
     linesY24=lines_ideal.y24
     linesY25=lines_ideal.y25
     linesY26=lines_ideal.y26
     linesY27=lines_ideal.y27
     linesY28=lines_ideal.y28
     linesY29=lines_ideal.y29
     linesY30=lines_ideal.y30
     linesY31=lines_ideal.y31
     linesY32=lines_ideal.y32
     linesY33=lines_ideal.y33
     linesY34=lines_ideal.y34
     linesY35=lines_ideal.y35
     linesY36=lines_ideal.y36
     linesY37=lines_ideal.y37
     linesY38=lines_ideal.y38
     linesY39=lines_ideal.y39
     linesY40=lines_ideal.y40
     linesY41=lines_ideal.y41
     linesY42=lines_ideal.y42
     linesY43=lines_ideal.y43
     linesY44=lines_ideal.y44
     linesY45=lines_ideal.y45
     linesY46=lines_ideal.y46
     linesY47=lines_ideal.y47
     linesY48=lines_ideal.y48
     linesY49=lines_ideal.y49
     linesY50=lines_ideal.y50
     
    
     #c_ideal.execute('''CREATE TABLE ideal_table(x, y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,y28,y29,y30,y31,y32,
     #                                            y33,y34,y35,y36,y37,y38,y39,y40,y41,y42,y43,y44,y45,y46,y47,y48,
     #                                            y49,y50)''')
     df_ideal = pd.read_csv(r'ideal.csv')
     df_ideal.to_sql('ideal_table', conn, if_exists='append', index=False)
     
     #c_ideal.execute("INSERT INTO ideal_table (x, y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,y28,y29,y30,y31,y32,y33,y34,y35,y36,y37,y38,y39,y40,y41,y42,y43,y44,y45,y46,y47,y48,y49,y50) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
     #                                                                               (linesX, linesY1, linesY2, linesY3, linesY4,
     #                                                                               linesY5, linesY6, linesY7, linesY8, linesY9,
     #                                                                               linesY10, linesY11, linesY12, linesY13, linesY14,
     #                                                                               linesY15, linesY16, linesY17, linesY18, linesY19,
     #                                                                               linesY20, linesY21, linesY22, linesY23, linesY24,
     #                                                                               linesY25, linesY26, linesY27, linesY28, linesY29,
     #                                                                               linesY30, linesY31, linesY32, linesY33, linesY34,
     #                                                                               linesY35, linesY36, linesY37, linesY38, linesY39,
     #                                                                               linesY40, linesY41, linesY42, linesY43, linesY44,
     #                                                                               linesY45, linesY46, linesY47, linesY48, linesY49,
     #                                                                               linesY50)) 
     #for row_ideal in c_ideal.execute('SELECT * FROM ideal_table'):
     #    print("result IDEAL view after insert")
     #    print(row_ideal)      
     #    print()   
    
   conn.commit()
   c.close()
   c_ideal.close()
   c_test.close()

   print("----------------DEVIATION AND CONFORMITY-----------------------")
   df_train.head()
   print(df_train.shape)
   print("bbbbbbbbbbbbbbbbbbbbbbbb")
   print(df_train ['x'].unique())
   print("ccccccccccccccccccccc")

   feature_names = ['y1','y2','y3','y4']
   XXX = df_ideal[feature_names]
   yyy = df_train['x']
   cmap = cm.get_cmap('gnuplot')
   scatter = pd.plotting.scatter_matrix(XXX, c = yyy, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)

   plt.suptitle('Scatter-matrix for each input variable !!')
   plt.savefig('AG_scatter_matrix')
   plt.show()
   
   
   print("---------------GRAPHEEEEEE------------------------")
   my_model = np.poly1d(np.polyfit(df_test['x'], df_test['y'], 4))
   print("my model")
   print(my_model)
   print("ccccooollllummmmns")
   resultX=my_model[1]
   resultY=my_model[2]
   resultIdealFct=my_model[3]
   resultDeviation=my_model[4]
   print("creating LASSSSSt table reasult")
   #c_result.execute('''CREATE TABLE result_table(x, y,ideal_fct,deviation)''')
   print("insert result  into table")
   c_result.execute("INSERT INTO result_table (x, y,ideal_fct,deviation) VALUES(?,?,?,?)", (resultX, resultY, resultIdealFct, resultDeviation)) 
   for row_result in c_result.execute('SELECT * FROM result_table'):
    print("result view ")
    print(row_result)   
    print()
   conn.commit()
   plt.plot(df_test['x'], my_model(df_test['y']), 'g-')
   plt.close()

   print("---------------------------------------")

  except  Error as e:
      print(e)
  finally:
      if conn:
       c.close()
       c_ideal.close()
       c_test.close()
       conn.close()

if __name__ == '__main__':
    main()	