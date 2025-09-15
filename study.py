import string 

def is_valid(cur_s):
    # 암호는 최소 1개의 모음 (a, e, i, o, u)와 최소 두 개의 자음으로 구성되어 있다고 알려져 있다. 
    all_possible_consonants = list(string.ascii_lowercase)
    all_possible_consonants.remove('a')
    all_possible_consonants.remove('e')
    all_possible_consonants.remove('i')
    all_possible_consonants.remove('o')
    all_possible_consonants.remove('u')
    # print(all_possible_consonants)
    if 'a' in cur_s or 'e' in cur_s or 'i' in cur_s or 'o' in cur_s or 'u' in cur_s:
        # 두 개의 자음으로 구성  
        cnt =0
        for char in cur_s:
            if char in all_possible_consonants:
                cnt += 1
                if cnt >= 2:
                    return True 
                
    return False 

def combinations(lev, start, path):
    if lev == L:
        # 암호는 최소 1개의 모음 (a, e, i, o, u)와 최소 두 개의 자음으로 구성되어 있다고 알려져 있다. 
        if is_valid(''.join(path)):
            # result.append(''.join(path))
            print(''.join(path))
            return
    
    for idx in range(start, C):
        path.append(chars[idx]) # choose 
        combinations(lev+1, idx+1, path) # explore 
        path.pop()  # unchoose 


def solve():
    global L, C, chars, result
    # f = open('/Users/dayelee/Documents/GitHub/mybook/Input.txt', 'r')
    # length = L, characters num = C 
    L, C = map(int, input().split())
    result = []
    # 가능성 있는 암호: ascending order 
    chars = list(input().split())
    chars.sort() # ascending order 

    # ascending order만 필요하므로 (1, 3, 2) 이런 것은 안된다. 따라서, 이미 ascending order로 되어 있는 것을 combination 조합으로 해결한다. 
    combinations(0, 0, [])

    

if __name__ == '__main__':
    solve()