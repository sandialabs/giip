import itertools

def partition(nTotal, nPartitions, indexInitial=0):
    temp = int(nTotal / nPartitions)
    ends = [temp for _ in range(nPartitions)]
    for i in range(nTotal - nPartitions*temp): ends[i] += 1
    ends = list(itertools.accumulate(ends))
    starts = [0] + ends
    ends = [i+indexInitial for i in ends]
    starts = [i+indexInitial for i in starts]
    return (starts,ends)

if __name__=='__main__':
    starts, ends = partition(10,3,100)
    starts, ends = partition(11,3,100)
    starts, ends = partition(12,3,100)

    starts, ends = partition(0,3,100)
    starts, ends = partition(1,3,100)
    starts, ends = partition(2,3,100)
    starts, ends = partition(3,3,100)
    print('done')
