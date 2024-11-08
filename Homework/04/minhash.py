import re
import pandas as pd
import numpy as np


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+"," ",text).lower()

    def tokenize(self, text: str) -> set:
        text = self.preprocess_text(text)      
        return set(text.split(' '))
    
    def get_occurrence_matrix(self, corpus_of_texts: list[set]) -> pd.DataFrame:
        '''
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        '''
        # TODO:
        tokens = list(set.union(*corpus_of_texts))
        mtx = [[token] + [1 if token in text else 0 for text in corpus_of_texts] for token in tokens]
        df = pd.DataFrame(mtx, columns=["token", *range(len(corpus_of_texts))], index=tokens)
        df.sort_index(inplace=True)
        df.index = [i for i in range(df.shape[0])]
        return df
    
    def is_prime(self, a):
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a
    
    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        '''
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number

        '''
        values_dict = {
            'a': [3, 4, 5, 7, 8],
            'b': [3, 4, 5, 7, 8] 
        }
        a = values_dict['a'][permutation_index]
        b = values_dict['b'][permutation_index]
        return (a*(x+1) + b) % prime_num_rows 
    
    
    def get_minhash_similarity(self, array_a: np.array, array_b: np.array) -> float:
        '''
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        '''
        # TODO:
        return len(set.intersection(set_a, set_b)) / len(set.union(set_a, set_b))
    
    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        # TODO:
        sim_mtx = self.get_similar_matrix(min_hash_matrix)
        return [(i, j) for i, row in enumerate(sim_mtx) for j, val in enumerate(row) if val > self.threshold and i < j]
    
    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        # TODO: 
        minshash_sets = [set(col) for col in min_hash_matrix.T]
        sim_mtx = [[0] * min_hash_matrix.shape[1] for _ in range(min_hash_matrix.shape[1])]
        for set_a_id in range(len(minshash_sets)):
            for set_b_id in range(set_a_id, len(minshash_sets), 1):
                sim_mtx[set_a_id][set_b_id] = self.get_jaccard_similarity(minshash_sets[set_a_id], minshash_sets[set_b_id])
                sim_mtx[set_b_id][set_a_id] = sim_mtx[set_a_id][set_b_id]
                
        return sim_mtx
     
    
    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. MinHash содержит в себе новые индексы. 

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1]
        [1, 0, 1]
        [0, 1, 1]

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 0
        Doc2 : 2
        Doc3 : 0
        '''
        # compute prime_num_rows
        prime_num_rows = occurrence_matrix.shape[0]
        while True:
            if self.is_prime(prime_num_rows):
                break
            prime_num_rows += 1
        minhash = np.zeros((self.num_permutations, occurrence_matrix.shape[1] - 1))
        for perm_id in range(self.num_permutations):
            perm_occurrence_matrix = occurrence_matrix.copy().iloc[:, 1:]
            perm = [self.get_new_index(x, perm_id, prime_num_rows) for x in list(occurrence_matrix.index)]
            perm_occurrence_matrix.index = perm
            perm_occurrence_matrix = perm_occurrence_matrix.sort_index()
            min_hash_row = perm_occurrence_matrix.idxmax(axis=0).to_numpy()
            minhash[perm_id] = min_hash_row
        # print(minhash)
        return minhash

    
    def run_minhash(self,  corpus_of_texts: list[str]):
        texts = [self.tokenize(text) for text in corpus_of_texts]
        occurrence_matrix = self.get_occurrence_matrix(texts)
        # print(occurrence_matrix)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        # print(similar_matrix)
        return similar_pairs

class MinHashJaccard(MinHash):
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold
    
    
    def get_minhash_similarity(self, array_a: np.array, array_b: np.array) -> float:
        '''
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        '''
        # TODO:
        return len(set.intersection(set_a, set_b)) / len(set.union(set_a, set_b))

    
    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        # TODO:
        sim_mtx = self.get_similar_matrix(min_hash_matrix)
        return [(i, j) for i, row in enumerate(sim_mtx) for j, val in enumerate(row) if val > self.threshold and i < j]
    
    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        # TODO: 
        minshash_sets = [set(col) for col in min_hash_matrix.T]
        sim_mtx = [[0] * min_hash_matrix.shape[1] for _ in range(min_hash_matrix.shape[1])]
        for set_a_id in range(len(minshash_sets)):
            for set_b_id in range(set_a_id, len(minshash_sets), 1):
                sim_mtx[set_a_id][set_b_id] = self.get_jaccard_similarity(minshash_sets[set_a_id], minshash_sets[set_b_id])
                sim_mtx[set_b_id][set_a_id] = sim_mtx[set_a_id][set_b_id]
                
        return sim_mtx
     
    
    def get_minhash_jaccard(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. Но в качестве мин хеша выписываем минимальный исходный индекс, не новый.
        В такой ситуации можно будет пользоваться расстояние Жаккрада.

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1] index: 2
        [1, 0, 1] index: 1
        [0, 1, 1] index: 0

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 2
        Doc2 : 0
        Doc3 : 2
        
        '''
        # compute prime_num_rows
        prime_num_rows = occurrence_matrix.shape[0]
        while True:
            if self.is_prime(prime_num_rows):
                break
            prime_num_rows += 1
        minhash = np.zeros((self.num_permutations, occurrence_matrix.shape[1] - 1))
        for perm_id in range(self.num_permutations):
            perm_occurrence_matrix = occurrence_matrix.copy().iloc[:, 1:]
            perm = [self.get_new_index(x, perm_id, prime_num_rows) for x in list(occurrence_matrix.index)]
            perm_occurrence_matrix.index = perm
            perm_occurrence_matrix = perm_occurrence_matrix.sort_index()
            min_hash_row = perm_occurrence_matrix.idxmax(axis=0).to_numpy()
            minhash[perm_id] = min_hash_row
        # print(minhash)
        return minhash

    
    def run_minhash(self,  corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs
    
    
    
