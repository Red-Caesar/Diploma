from typing import List


class CharactersTextSplitter:
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        if chunk_size <= 0:
            raise ValueError("Chunk length must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be a positive integer")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start += self._chunk_size - self._chunk_overlap

        return chunks
