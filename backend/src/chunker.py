import uuid
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.tokens import count_tokens

_SEP = " "
_DEFAULT_MODEL = "gpt-4o-mini"


class TimestampAwareTextSplitter(RecursiveCharacterTextSplitter):
    """RecursiveCharacterTextSplitter that preserves start_ts / end_ts metadata.

    The base splitter splits each document independently and copies its metadata
    to every resulting chunk.  This subclass instead treats all source segments
    as one stream, splits the combined text, then walks back through the source
    segments to assign the correct start_ts and end_ts to each chunk.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model: str = _DEFAULT_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: count_tokens(text, model),
            **kwargs,
        )

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        docs = list(documents)
        if not docs:
            return []

        # 1. Build combined text; record each source doc's char range.
        parts: list[str] = []
        positions: list[tuple[int, int]] = []
        cursor = 0
        for doc in docs:
            text = doc.page_content
            parts.append(text)
            positions.append((cursor, cursor + len(text)))
            cursor += len(text) + len(_SEP)

        combined = _SEP.join(parts)

        # 2. Delegate text splitting to the parent implementation.
        raw_chunks = self.split_text(combined)

        # 3. Walk raw_chunks left-to-right; map each chunk back to the source
        #    segments it spans, then rebuild timestamps and assign chunk_id.
        result: list[Document] = []
        search_from = 0
        for chunk_text in raw_chunks:
            idx = combined.find(chunk_text, search_from)
            if idx == -1:
                idx = combined.find(chunk_text)
            if idx == -1:
                continue

            chunk_end = idx + len(chunk_text)
            search_from = idx + 1

            overlapping = [
                doc
                for doc, (s, e) in zip(docs, positions, strict=True)
                if s < chunk_end and e > idx
            ]
            if not overlapping:
                continue

            metadata = {
                **overlapping[0].metadata,
                "start_ts": overlapping[0].metadata["start_ts"],
                "end_ts": overlapping[-1].metadata["end_ts"],
                "chunk_id": str(uuid.uuid4()),
            }
            result.append(Document(page_content=chunk_text, metadata=metadata))

        return result


def chunk_documents(
    docs: list[Document],
    target_tokens: int = 500,
    overlap_tokens: int = 50,
    model: str = _DEFAULT_MODEL,
) -> list[Document]:
    """Split segment Documents into token-bounded chunks.

    Each output Document has:
      page_content  — concatenated text from one or more source segments
      metadata:
        start_ts    — start of the first contributing segment (seconds)
        end_ts      — end of the last contributing segment (seconds)
        chunk_id    — unique UUID4 string
        video_id    — propagated from source segments
    """
    return TimestampAwareTextSplitter(
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        model=model,
    ).split_documents(docs)
