'''
BPE - Byte Pair Encoding: 通过迭代地合并文本中最频繁出现的相邻字符对或字节对来构建词汇表。
'''
from collections import defaultdict

def merge(indices, pair, new_index):
    '''
    将常见的词对组合赋予一个新的idx，然后代替原先的多个idx
    ※ 注意是用一个新的idx替换多个idx！！！
    indices: 原来的语句加密后的表示
    pair: 要合并的词对
    new_index: 新的idx
    '''
    new_indices = []
    skip_step = len(pair)
    i = 0
    while i < len(indices) - 1:
        if indices[i] == pair[0] and indices[i+1] == pair[1]:
            new_indices.append(new_index)
            i += skip_step
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def train_bpe(s: str, num_merges: int):
    indices = list(map(int, s.encode("utf-8")))
    merges = {}
    vocab = {x: bytes([x]) for x in range(256)}
    
    for i in range(num_merges):
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1
        
        pair = max(counts, key=counts.get)
        index1, index2 = pair

        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        print(vocab[new_index])
        indices = merge(indices, pair, new_index)


if __name__ == "__main__":
    s = "the cat in the hat"
    print(len(s))
    train_bpe(s, 3)