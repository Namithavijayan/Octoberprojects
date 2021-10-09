inputs=[]
n=int(input("Enter number of terms"))
for i in range(0,n):
    inputs.append(int(input("Enter the term: ")))
print("The list after removing negative numbers")    
for i in inputs:
    if i>0:
       print(i)   