"""Fixture transcript and pre-chunked documents for E2E tests.

Simulates a 10-minute Python tutorial video (ID: fixture_python_e2e_01).
The fixture covers 10 topics so Q/A pairs can target specific chunks.
"""

from __future__ import annotations

from langchain_core.documents import Document

FIXTURE_VIDEO_ID = "fixture_python_e2e_01"

# Raw transcript segments (mirroring youtube-transcript-api shape)
FIXTURE_SEGMENTS: list[dict] = [
    {
        "text": "Welcome to this Python tutorial. Today we cover list comprehensions.",
        "start": 0.0,
        "duration": 6.0,
    },
    {
        "text": "A list comprehension creates a list in a single concise line of code.",
        "start": 6.0,
        "duration": 6.0,
    },
    {
        "text": "For example: squares = [x**2 for x in range(10)] produces ten squares.",
        "start": 12.0,
        "duration": 6.0,
    },
    {
        "text": "You can add a filter: even_squares = [x**2 for x in range(10) if x % 2 == 0].",
        "start": 18.0,
        "duration": 6.0,
    },
    {
        "text": "Dictionary comprehensions work the same way: {k: v for k, v in pairs}.",
        "start": 60.0,
        "duration": 6.0,
    },
    {
        "text": "Lambda functions are anonymous one-line functions using the lambda keyword.",
        "start": 120.0,
        "duration": 6.0,
    },
    {
        "text": "Example: double = lambda x: x * 2. Calling double(5) returns 10.",
        "start": 126.0,
        "duration": 6.0,
    },
    {
        "text": "Decorators wrap a function to add behaviour without changing its source.",
        "start": 180.0,
        "duration": 6.0,
    },
    {
        "text": "Apply a decorator using the @ symbol above the function definition.",
        "start": 186.0,
        "duration": 6.0,
    },
    {
        "text": "Context managers use the with statement to handle setup and teardown.",
        "start": 240.0,
        "duration": 6.0,
    },
    {
        "text": "with open('file.txt') as f: ensures the file closes even on exceptions.",
        "start": 246.0,
        "duration": 6.0,
    },
    {
        "text": "Generator expressions are memory-efficient: (x**2 for x in range(10)).",
        "start": 300.0,
        "duration": 6.0,
    },
    {
        "text": "Type hints document argument types: def greet(name: str) -> str.",
        "start": 360.0,
        "duration": 6.0,
    },
    {
        "text": "f-strings embed expressions inline: f'Hello {name}' is concise and fast.",
        "start": 420.0,
        "duration": 6.0,
    },
    {
        "text": "Unpacking: a, b, *rest = [1, 2, 3, 4, 5] assigns values in one step.",
        "start": 480.0,
        "duration": 6.0,
    },
]


def make_fixture_docs() -> list[Document]:
    """Return pre-chunked Documents ready for direct insertion into Chroma."""
    topics = [
        (
            0.0,
            24.0,
            "A list comprehension creates a list in a single concise line. squares = [x**2 for x in range(10)] produces squares. Filter with: [x**2 for x in range(10) if x % 2 == 0].",
        ),
        (
            60.0,
            72.0,
            "Dictionary comprehensions work like list comprehensions: {k: v for k, v in pairs} builds a dict in one line.",
        ),
        (
            120.0,
            138.0,
            "Lambda functions are anonymous one-line functions. Example: double = lambda x: x * 2. Calling double(5) returns 10.",
        ),
        (
            180.0,
            198.0,
            "Decorators wrap a function to add behaviour without changing its source. Apply using the @ symbol above the function definition.",
        ),
        (
            240.0,
            258.0,
            "Context managers use the with statement for setup and teardown. with open('file.txt') as f: ensures the file closes on exceptions.",
        ),
        (
            300.0,
            312.0,
            "Generator expressions are memory-efficient lazy sequences: (x**2 for x in range(10)) does not create the list immediately.",
        ),
        (
            360.0,
            372.0,
            "Type hints document argument types: def greet(name: str) -> str. They help IDEs and type checkers catch bugs.",
        ),
        (
            420.0,
            432.0,
            "f-strings embed expressions inline: f'Hello {name}' is concise and fast. Introduced in Python 3.6.",
        ),
        (
            480.0,
            492.0,
            "Unpacking assigns values in one step: a, b, *rest = [1, 2, 3, 4, 5]. The star gathers remaining items into a list.",
        ),
        (
            540.0,
            552.0,
            "Class inheritance: class Dog(Animal): allows Dog to reuse and extend Animal's methods and attributes.",
        ),
    ]
    docs = []
    for i, (start, end, text) in enumerate(topics):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "video_id": FIXTURE_VIDEO_ID,
                    "chunk_id": f"{FIXTURE_VIDEO_ID}_{i:03d}",
                    "start_ts": start,
                    "end_ts": end,
                },
            )
        )
    return docs
