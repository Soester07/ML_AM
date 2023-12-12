import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r'C:\Users\Kartthik\Desktop\Personal\Lab01\Lab1.xlsx')
irctc = pd.read_excel(r'C:\Users\Kartthik\Desktop\Personal\Lab01\Lab1.xlsx', sheet_name= "IRCTC Stock Price")
df = pd.DataFrame(data)
df.drop(df.iloc[:, 5:22], axis=1, inplace=True)
print(df)

purchase = df.iloc[0:9,1:4]
print(purchase)
A=np.array(purchase)
print(A)

total = df.iloc[0:9,4]
B = np.array(total)
B = B.reshape(9,1)
print(B)

print("Dimensionality of given data is", df.shape)
#Vectors are spanned by rows hence number of rows
print("Number of rows are", df.shape[0])

print("The rank of matrix is ", np.linalg.matrix_rank(purchase))

#Inverse usinfg np.linalg.pinv

X = np.linalg.pinv(A) 
print(X)

Z = np.dot(X,B)
print("Therefore solution is", Z)

print("Cost of a candy is", Z[0])
print("COst of a mangoe is", Z[1])
print("Cost of a milk packet is", Z[2])

#Catogarizing based on Payment

new_df = df
status_pay = []
for row in df['Payment (Rs)']:
  if row < 200 :
    status_pay.append('Poor')
  elif row>=200 :
    status_pay.append('Rich')
new_df['Pay']= status_pay
print(new_df)

#irctc

irctc_df = pd.DataFrame(irctc)
print(irctc_df)

#mean of the price

mean_irctc = irctc_df["Price"].mean()
print("Price mean = ", mean_irctc)

#variance of the price

var_irctc = irctc_df["Price"].var()
print("Price Variance = ", var_irctc)

#Comparing avg of wed with overall avg (lowest sales)

wed_mean = irctc_df.loc[irctc_df['Day'] == 'Wed', 'Price'].mean()
print("Sales at IRCTC on a wednesday are : ",wed_mean)
print(wed_mean," < ",mean_irctc)
print("As we can see the sales at IRCTC are less during Wednesdays compared to average")


#Comparing avg of april with overall avg (highest sales)

apr_mean = irctc_df.loc[irctc_df['Month'] == 'Apr', 'Price'].mean()
print("Sales at IRCTC during April moonth are : ",apr_mean)
print(mean_irctc," < ",apr_mean)
print("As we can see the sales at IRCTC are higher during the April Month compared to the average")

# Calculating possibilities of making loss

neg = 0

for index,row in irctc_df.iterrows():
    if row['Chg%'] < 0:
        neg+=1
        
print("Probability of making loss is = ",neg/irctc_df.shape[0])


# Calculating possibilities of making profit on wednesday

wed=0
wed_pos=0



for index,row in irctc_df.iterrows():
    if row['Day']=='Wed':
        if row['Chg%']>0:
            wed_pos+=1
        wed+=1
        
print("Probability of making profit on Wednesday ",wed_pos/wed)

#ploting using matplotlib

days= []
chg= []
for index,row in irctc_df.iterrows():
    days.append(row['Day'])
    chg.append(row['Chg%'])


plt.scatter(days, chg)
plt.show()




'''A = np.array(df)
#print(A) To check if it has converted correctly
C_total = df.iloc[0:9,4]
C = np.array(C_total)
C = C.reshape(9,1)
#print(C) To check the matrix in 1X3
print("Dimensionality of given data is", df.shape)
#Vector Space is spanned by rows hence the number of rows should give you the number of vectors
print("Number of Vectors is", df.shape[0])
print("The Rank of a Matrix: ", np.linalg.matrix_rank(C_total))
#Psudo Inverse using np.linalg.svd()
Y = np.linalg.svd(C)
print(Y)
X =np.dot(Y,C)
print(X)'''
