import sys
import copy


class SubArray:
    def __init__(self, start, end, summ):
        self.start_number = start
        self.end_number = end
        self.summ = summ

def findMaxSubArray(A):
    if len(A) == 0:
        raise Exception("list cannot be zero length")
    max_sub_array = SubArray(0, 0, 0)
    curr_sub_array = SubArray(0, 0, 0)
    for i, value in enumerate(A):
        if curr_sub_array.summ >= 0:
            curr_sub_array.end_number = i
            curr_sub_array.summ += value   
        else:
            curr_sub_array = SubArray(i, i, value)
        if curr_sub_array.summ > max_sub_array.summ:
            max_sub_array = copy.copy(curr_sub_array)
    return A[max_sub_array.start_number : max_sub_array.end_number + 1]

def print_help():
    print("Usage: {program_name} {array of integer without parentheses}",
          "for example: python3 find_sub_array.py -2,1,-3,4,-1,2,1,-5,4", sep="\n")

def main(argv):
    arr = []
    if len(argv) > 1:
        try:
            arr = [int(i) for i in argv[1].split(',')]
        except Exception as e:
            print(f"Failed to parse integer list: {e}")
            print_help()
            return
    else:
        print_help()
        return
    
    print(findMaxSubArray(arr))

if __name__== "__main__":
    main(sys.argv)