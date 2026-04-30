from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SEP = " "


class TimestampAwareTextSplitter:
    """Wraps RecursiveCharacterTextSplitter and restores start_ts/end_ts per chunk.

    Off-the-shelf splitters discard per-segment metadata when merging segments
    into chunks.  This class rebuilds timestamps by mapping each output chunk
    back to the source segments it was drawn from.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        # Build combined text and record each source doc's char range within it.
        parts: list[str] = []
        positions: list[tuple[int, int]] = []  # (start_char, end_char) inclusive-exclusive
        cursor = 0
        for doc in documents:
            text = doc.page_content
            parts.append(text)
            positions.append((cursor, cursor + len(text)))
            cursor += len(text) + len(_SEP)

        combined = _SEP.join(parts)
        raw_chunks = self._splitter.split_text(combined)

        result: list[Document] = []
        search_from = 0
        for chunk_text in raw_chunks:
            idx = combined.find(chunk_text, search_from)
            if idx == -1:
                idx = combined.find(chunk_text)
            if idx == -1:
                continue

            chunk_start = idx
            chunk_end = idx + len(chunk_text)
            # Advance cursor by 1 so next chunk (which may overlap) is found
            # at or after this position, but never earlier.
            search_from = idx + 1

            overlapping = [
                doc
                for doc, (s, e) in zip(documents, positions, strict=True)
                if s < chunk_end and e > chunk_start
            ]
            if not overlapping:
                continue

            metadata = {
                **overlapping[0].metadata,
                "start_ts": overlapping[0].metadata["start_ts"],
                "end_ts": overlapping[-1].metadata["end_ts"],
            }
            result.append(Document(page_content=chunk_text, metadata=metadata))

        return result
