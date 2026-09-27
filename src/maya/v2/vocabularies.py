"""Closed vocabularies for the v2 Understanding contract (#82 C3–C5, C8).

The model cannot emit a label outside these sets — pydantic ``Literal``
types reject anything else at parse time, which is the whole point of the
v2 contract: fifteen regex and vocabulary gates reduced to a schema.

The genre set is the dataset's own (``MovieDatabase().distinct_genres()``,
19 labels, TV Movie included). It is frozen here as a static tuple so this
module stays pure (ADR 0006 — no storage imports in domain code);
``tests/unit/test_v2_vocabularies.py`` pins the tuple to the live dataset
so the two cannot drift.
"""

from typing import Literal

#: #82 C3: closed mood set with the hint text the v2 prompt shows the model.
MOOD_HINTS: dict[str, str] = {
    "edge-of-your-seat": "tense, suspenseful, gripping",
    "thrilling": "exciting, action-packed, adrenaline",
    "funny": "comedy, laugh-out-loud",
    "feel-good": "warm, uplifting, comforting",
    "scary": "horror, frightening, chilling",
    "romantic": "love stories, romance-forward",
    "tearjerker": "emotional, moving, sad",
    "epic": "grand scale, sweeping, monumental",
}

#: #82 C4: closed audience set.
AUDIENCES: tuple[str, ...] = ("solo", "date night", "family", "kids", "adults")

#: #82 C5: the nineteen dataset genres — see module docstring for the
#: purity trade and the pinning test.
GENRES: tuple[str, ...] = (
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western",
)

Mood = Literal[
    "edge-of-your-seat", "thrilling", "funny", "feel-good",
    "scary", "romantic", "tearjerker", "epic",
]
Audience = Literal["solo", "date night", "family", "kids", "adults"]
Genre = Literal[
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western",
]

#: #82 C8: the four Narrowing Axes — the only slots the funnel may chase.
Axis = Literal["mood", "audience", "genres", "era"]
