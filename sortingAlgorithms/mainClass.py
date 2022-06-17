import time

from sortingAlgorithms.bubbleSorting  import bubbleSorting


import copy
from collections import Counter

class car:
    def __init__(self,carname , year,enginetype,*moreinfo):
        self.carname=carname
        self.year=year
        self.enginetype=enginetype
        self.info=moreinfo

    def printcarDetails(self,func):
        print(f"car name is {self.carname}")
        print(f"car manufactured year is {self.year}")
        print(f"car engine type is {self.enginetype}")
        for a in self.info:
            print(f"other car details are {a}")
        func()

    def changeEngineType(self,enginetype):
        if (self.enginetype==enginetype):
            print("already same engine type , no need to change.")
        else:
            print("sucessfully changed your engine type")


class electricCar(car):
    def __init__(self,carname , year,enginetype,*moreinfo):
        super().__init__(carname , year,enginetype,moreinfo)

    def changeEngineType(self,enginetype):

        print("This is electric vehicle you cant change engine")



    # def printcarDetails(self,func):
    #     func()
    #     super().printcarDetails(func)
    #     func()
    #
    # def printcarDeails(self,n):
    #
    #     print("##########")



car1 = car("Ford Endevaour" , 2022, "petrol","BS6","AI updated")
car2 = car("Mini Cooper" , 2021, "petrol","BS4","AI updated"," Top Window")
car3 = electricCar("Tesla S",2023,"Electric")


def calc(x,y):
   
    return x*y


def twodigit(func):
    def wert(*args, **kwargs):
        val = func(*args,**kwargs)
        print("val :",val)
        return val
    return wert

@twodigit
def subtr(x,y):
    return x-y

val=(subtr(9,5))

print(val)

from multiprocessing import Process

from threading import Thread ,Lock

import time

database=0
def increase(lock):
    global database

    lock.acquire()
    local=database

    local+=1
    time.sleep(1.0)
    database=local
    lock.release()
    a=b


if __name__=="__main__":
    lock = Lock()
    t1= Thread(target=increase,args=(lock,))
    t2=Thread(target=increase,args=(lock,))
    print(f" Before global var :{database}")
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print(f" After Glbal :{database}")



# a={}
#
# #b= dict(a:10)
#
# a["zz"]=99
# #a.append(90)
# print((a))
# #print(b)
#
# s = bubbleSorting()
# #print(s.sort(a))
