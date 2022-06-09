
class mergeSort:
    def __init__(self):
        pass
    def sort(self,arr):
        if (len(arr)<2):
            return arr
        self.mergeSorting(arr)



    def mergeSorting(self,arr):
        middle = len(arr)//2
        left=arr[:middle]
        right=arr[middle:]
        self.sorting(arr,left,right)


    def sorting(self,arr, left,right):

        if left<








