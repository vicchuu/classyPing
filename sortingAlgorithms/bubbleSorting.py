class bubbleSorting:
    def __init__(self):
        print("call cons bubble sorting....")
        pass
    def sort(self,arr):
        if len(arr)<2:
            return arr
        for a in range (len(arr)):
            for b in range (len(arr)-a-1):
                if arr[b]>arr[b+1]:
                    arr[b],arr[b+1]=arr[b+1],arr[b]


        return arr


