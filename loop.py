n=int(input("Enter number of terms:"))
n1=0
n2=1
print(n1)
print(n2)
for i in range(0,n-2):
    t=n1
    n1=n2
    n2=n1+t
    print(n2)
