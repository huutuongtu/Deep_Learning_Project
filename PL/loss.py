f = open("./loss.txt", "r")
x = f.read()
# print(x)
x = x.split("tensor(")
# print(x[0])
for i in range(len(x)):
  x[i] = x[i].split(",")[0]
  # print(x[i])

a = [[0] for i in range(len(x)-1)]
for i in range(len(x)-1):
  a[i] = x[i+1]
for i in range(len(a)):
    a[i] = float(a[i])
sum1 = 0
# sum2 = 0
# sum3 = 0
sum4 = 0
sum5 = 0
sum6 = 0
sum7 = 0
for i in range(2000):
  sum7 = sum7 + a[i+140000]
  sum1 = sum1 + a[i+160000]
  sum4 = sum4 + a[i+250000]
  sum5 = sum5 + a[i+480000]
  sum6 = sum6 + a[i+580000]
  
print(sum7) #0.44
print(sum1)
# print(sum2)
# print(sum3)
print(sum4)
print(sum5)
print(sum6)
