from collections import defaultdict

global words_dict, total, chars
total = 0
chars = "AEIOU"
words_dict = defaultdict(int)

# words= "AEIOU", k 1~5개 글자, 중복 허용, 순서 무관 'A'앞에 'E'가 와도 됨-> k가 없는 combination (subsequences)
def create_words_dict(start,  lev, path):
    global total 
    if lev >= 5:
        return 
    
    for idx in range(5):
        path.append(chars[idx])
        total += 1 
        words_dict[''.join(path)] = total
        # print(''.join(path), total)
        create_words_dict(idx, lev+1, path)# 중복 허용, idx+1가 아닌 idx넣기
        path.pop()

def solution(word):
    
    answer = 0
    create_words_dict(0, 0, [])
    answer = words_dict[word]

    return answer


if __name__ == "__main__":
    # print(solution('EIO'))
    # print(solution('AAAE'))
    print(solution('I'))