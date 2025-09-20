
parent = [0] * 100 
gCnt = [0] * 100 

def get_parent(ch):
    if parent[ch] == 0:
        return ch 
    
    ret = get_parent(parent[ch])
    parent[ch] = ret # make short-cut 
    return ret 


def insert(ch1, ch2):
    a = get_parent(ch1)
    b = get_parent(ch2)

    if a != b:
        parent[b] = a 

    gCnt[a] += gCnt[b]
    gCnt[b] = 0

def get_count(ch):
    ret = get_parent(ch)
    return gCnt[ret]
