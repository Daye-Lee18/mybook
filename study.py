def heappush(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1)

def _siftdown(heap, root, pos):
    newitem = heap[pos] # newitem 값 저장 
    while pos > root:
        parent_pos = (pos-1) >> 1 
        if heap[parent_pos] > newitem: # 항상 newtiem과 부모를 비교해야함. 
            heap[pos] = heap[parent_pos] # 부모를 한 칸 내려보냄 
            pos = parent_pos
            continue 
        break 
    # SWAP 
    heap[pos] = newitem 

def heappop(heap):
    # 맨 "마지막 원소"를 제거: heap안의 원소가 하나라면 맨 처음 원소이고, 아니라면 root position에 갈 원소가 됨. 
    removed_item = heap.pop() # heap이 비어있으면 에러를 일으킴 
    if heap: # pop 이후에도 heap안에 item이 있는 경우 
        returnitem = heap[0]
        heap[0] = removed_item 
        _siftup(heap, 0)
        return returnitem 
    return removed_item 

def is_leaf(heap, pos):
    # time complexity for len(list) = O(1), 내부에 길이를 따로 저장하고 있음.
    # "왼쪽 자식 인덱스가 배열 길이보다 크거나 같은 경우"로 판정하면 충분 
    return (pos * 2) + 1 >= len(heap)
  

def _siftup(heap, pos):
    end = len(heap)
    newitem = heap[pos]
    child = 2* pos + 1 # 왼쪽 자식 

    while child < end: # is_leaf()와 동일한 형식   
        right= child + 1 
        # 더 작은 자식을 child로 선택 
        if right < end and heap[right] < heap[child]:
            child = right 
        
        # child가 newitem보다 작으면 child를 끌어올리고, pos를 child로 이동 
        if heap[child] < newitem:
            heap[pos] = heap[child] # child를 siftup 
            pos = child 
            child = 2 * pos + 1 
        else:
            break # newitem이 들어갈 자리이면 멈춤 

    
    heap[pos] = newitem 
    _siftdown(heap, 0, pos)
    

heap = [1, 3, 5, 4, 6, 9, 10, 8]
heappop(heap)
print(heap)

