f = open("./cer.txt", "r")
x = f.read()
# print(x)
s = 0
x = x.split("\n")
for i in x:
  try:
    # print(float(i))
    s = s + float(i)
  except:
    i = 0
# print(s)
print("cer: " + str(s/len(x)))
f = open("./wer.txt", "r")
x = f.read()
# print(x)
s = 0
x = x.split("\n")
for i in x:
  try:
    # print(float(i))
    s = s + float(i)
  except:
    i = 0
# print(s)
print("wer: " + str(s/len(x)))