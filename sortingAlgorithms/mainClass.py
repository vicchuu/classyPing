
from sortingAlgorithms.bubbleSorting  import bubbleSorting


import copy
from collections import Counter


def user_info(age , **infor):
    print("age :",age)
    for a,v in infor.items():
        print(a,v)



user_info(23,name="physics",name1="maths",name2="Science",name3="Social")


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
