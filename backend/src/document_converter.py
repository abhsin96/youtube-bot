from langchain_core.documents import Document

from src.transcript_service import TranscriptSegment


def segments_to_documents(video_id: str, segments: list[TranscriptSegment]) -> list[Document]:
    return [
        Document(
            page_content=seg["text"],
            metadata={
                "video_id": video_id,
                "segment_index": idx,
                "start_ts": seg["start"],
                "end_ts": round(seg["start"] + seg["duration"], 6),
            },
        )
        for idx, seg in enumerate(segments)
    ]
