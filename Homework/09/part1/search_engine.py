from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    id: str
    title: str
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    doc_id: str
    score: float
    title: str
    text: str


def load_documents(path: str) -> List[Document]:
    """Загрузка документов из json файла"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(
            id=article["id"],
            title=article["title"],
            text=article["text"],
            embedding=None,
        )
        for article in data["articles"]
    ]


class Indexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать индексацию документов
        1. Сохранить документы в self.documents
        2. Получить эмбеддинги для документов используя self.model.encode()
           Подсказка: для каждого документа нужно объединить title и text
        3. Сохранить эмбеддинги в self.embeddings
        """
        self.documents = documents
        self.embeddings = self.model.encode(
            [doc.title + " " + doc.text for doc in documents]
        )

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса
        1. Сохранить self.documents и self.embeddings в pickle файл
        """
        with open(path, "wb") as f:
            pickle.dump((self.documents, self.embeddings), f)

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса
        1. Загрузить self.documents и self.embeddings из pickle файла
        """
        with open(path, "rb") as f:
            self.documents, self.embeddings = pickle.load(f)


class Searcher:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        TODO: Реализовать инициализацию поиска
        1. Загрузить индекс из index_path
        2. Инициализировать sentence-transformers
        """
        self.model = SentenceTransformer(model_name)
        with open(index_path, "rb") as f:
            self.documents, self.embeddings = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск документов
        1. Получить эмбеддинг запроса через self.model.encode()
        2. Вычислить косинусное сходство между запросом и документами
        3. Вернуть top_k наиболее похожих документов
        """
        query_embedding = self.model.encode(query)
        cosine_similarities = np.dot(self.embeddings, query_embedding)
        top_k_indices = np.argsort(cosine_similarities)[::-1][:top_k]
        search_results = []

        for i in top_k_indices:
            search_results.append(
                SearchResult(
                    doc_id=self.documents[i].id,
                    score=float(cosine_similarities[i]),
                    title=self.documents[i].title,
                    text=self.documents[i].text,
                )
            )
        return search_results
