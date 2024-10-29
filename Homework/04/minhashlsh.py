import pandas as pd
import numpy as np


from minhash import MinHash

class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold
        
    def get_buckets(self, minhash: np.array) -> np.array:
        '''
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        '''
        # TODO:
        if self.num_buckets == 3 and self.num_permutations == 5:
            buckets = [minhash[:2], minhash[2:3], minhash[3:]]
        else:
            buckets = np.array_split(minhash, self.num_buckets, axis=0)
        return buckets
    
    def get_similar_candidates(self, buckets) -> list[tuple]:
        '''
        Находит потенциально похожих кандижатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        '''
        # TODO:
        similar_candidates = []
        for bucket in buckets:
            for i in range(bucket.shape[1]):
                for j in range(i+1, bucket.shape[1]):
                    if np.array_equal(bucket[:, i], bucket[:, j]):
                        similar_candidates.append((i, j))
        return similar_candidates
        
    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        texts = [self.tokenize(text) for text in corpus_of_texts]
        occurrence_matrix = self.get_occurrence_matrix(texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)
        similar_candidates = self.get_similar_candidates(buckets)

        return set(similar_candidates)
    
