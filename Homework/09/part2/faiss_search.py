from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from part1.search_engine import Document, SearchResult


class FAISSSearcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать создание FAISS индекса

        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        self.documents = documents
        self.embeddings = self.model.encode(
            [doc.title + " " + doc.text for doc in documents]
        )
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(self.dimension), self.dimension, 6
        )
        self.index.train(self.embeddings)
        self.index.add(self.embeddings)

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса

        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        with open(path, "wb") as f:
            pickle.dump((self.documents, faiss.serialize_index(self.index)), f)

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса

        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        with open(path, "rb") as f:
            self.documents, self.index = pickle.load(f)
            self.index = faiss.deserialize_index(self.index)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск

        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        print(distances)
        return [
            SearchResult(
                self.documents[i].id,
                float(
                    np.clip(distance, 0, 1)
                ),  # тут не понял почему [0,1] диапазон значений, расстояние же может быть [0,2]
                self.documents[i].title,
                self.documents[i].text,
            )
            for i, distance in zip(indices[0], distances[0])
        ]

    def batch_search(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[SearchResult]]:
        """
        TODO: Реализовать batch-поиск

        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        query_embeddings = self.model.encode(queries)
        faiss.normalize_L2(query_embeddings)
        distances, indices = self.index.search(query_embeddings, top_k)
        return [
            [
                SearchResult(
                    self.documents[i].id,
                    float(distance),
                    self.documents[i].title,
                    self.documents[i].text,
                )
                for i, distance in zip(indices[j], distances[j])
            ]
            for j in range(len(queries))
        ]
