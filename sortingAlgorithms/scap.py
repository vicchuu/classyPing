class A:
    var1 = 1
    var2 = 2

    def m1(self):
        print("M1 in A")
    def m2(self):
        print("M2 in A")

class B:
    def __init__(self,v1):
        self._v1 = v1
        self._v2 = 0

    def m1(self):
        print("M1 in B")
    @property
    def v1(self):
        return self.__v1
    @v1.setter
    def v1(self,value):
        self.__v1 = value

    def m2(self):
        print("M2 in B")


a = {"a":1,"b":2,"c":3}
b = B(299)


books = ["vishnu" , " vicky",23,"zooooom"]

for book in books:
    if type(book)==int:
        print(f" Books name is :{book}")

print(a.keys())
print(a.items())

n= [1,2,3,4,5,6,7,8,9]

v= iter(["a","a","a"])
while (next(v)!=NULL):
    print(next(v))


print(list(map(lambda x:x<4,n)))

print(list(filter(lambda x:x*x,n)))
from functools import reduce
print(reduce(lambda x,y:x*y,n))


