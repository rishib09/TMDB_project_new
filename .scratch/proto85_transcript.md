# Prototype #85 — v2 Understand vs v1 on three documented conversations

Model: `z-ai/glm-5.3-flash` (reasoning `low`) · temperature 0.0 · knobs: era_old_year_max=2000, era_recent_year_min=2015, funnel_retrieve_axes=2, MAX_PROBE_TURNS=2

## A - Report conversation (#78/#79/#80, trace 0839ce6c)

### Turn 1: `show me some movies`

**v1 did:** v1: funnel entry, mood probe asked (no era captured; n/a here)

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "show me some movies",
  "filters": null,
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": false,
  "missing_slots": [
    "mood",
    "audience",
    "genres",
    "era"
  ],
  "clarifying_question": "What kind of vibe are you after \u2014 something funny, scary, romantic, or edge-of-your-seat?",
  "referenced_titles": [],
  "confidence": 0.6
}
```
- disposition: no changes
- turn decision: **ask** (missing slots ['mood', 'audience', 'genres', 'era'])
- merged prefs: mood=-, audience=-, genres=-, years=(None, None)
- clarifying question: What kind of vibe are you after — something funny, scary, romantic, or edge-of-your-seat?
- usage: in=780 out=174 (reasoning=10)

### Turn 2: `feel good`

**v1 did:** v1: mood=feel-good set via probe; genre candidates confirm stage

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "feel-good movies, warm and uplifting",
  "filters": null,
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": "feel-good",
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": false,
  "missing_slots": [
    "audience"
  ],
  "clarifying_question": "Nice, feel-good it is \u2014 who's watching: solo, date night, or family?",
  "referenced_titles": [],
  "confidence": 0.95
}
```
- disposition: no changes
- turn decision: **ask** (missing slots ['audience'])
- merged prefs: mood=feel-good, audience=-, genres=-, years=(None, None)
- clarifying question: Nice, feel-good it is — who's watching: solo, date night, or family?
- usage: in=810 out=1971 (reasoning=1773)

### Turn 3: `date night`

**v1 did:** v1: audience=date night; retrieved 5 incl. Date Movie 2006, Holidate 2020, Date Night 2010

```json
{
  "intent": "ATTRIBUTE_FILTER",
  "standalone_query": "feel-good date night movie",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": "date night",
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.97
}
```
- disposition: no changes
- turn decision: **retrieve** (ready + at least one axis known)
- merged prefs: mood=feel-good, audience=date night, genres=-, years=(None, None)
- clarifying question: -
- usage: in=808 out=1011 (reasoning=746)
- NOTE: probe budget exhausted (2/2); ready_to_retrieve forced

### Turn 4: `give me recent movie`

**v1 did:** v1: router failed TWICE (length limit, 979 hidden reasoning tokens), fallback conf 0.1 no filters; prefs year_min=2015 never applied by retrieve; 3/5 posters repeated from turn 3

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "recent feel-good movie for date night",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": "recent",
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.9
}
```
- disposition: disposition: era 'recent' -> year_min=2015
- turn decision: **retrieve** (ready + at least one axis known)
- merged prefs: mood=feel-good, audience=date night, genres=-, years=(2015, None)
- clarifying question: -
- usage: in=813 out=226 (reasoning=36)
- NOTE: probe budget exhausted (2/2); ready_to_retrieve forced

## B - #56 walkthrough ('show me old classic' -> 'just me')

### Turn 1: `show me old classic`

**v1 did:** v1: era NOT captured on funnel entry (extract_era ran only in funnel_node) -> no year_max

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "old classic movies",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": "old",
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [
    "mood",
    "audience",
    "genres"
  ],
  "clarifying_question": "Any particular genre or mood you're in the mood for \u2014 drama, thriller, comedy?",
  "referenced_titles": [],
  "confidence": 0.7
}
```
- disposition: disposition: era 'old' -> year_max=2000
- turn decision: **retrieve** (ready + at least one axis known)
- merged prefs: mood=-, audience=-, genres=-, years=(None, 2000)
- clarifying question: Any particular genre or mood you're in the mood for — drama, thriller, comedy?
- usage: in=780 out=319 (reasoning=32)

### Turn 2: `feel good`

**v1 did:** v1: mood probe answered; genre candidates Comedy/Drama/Family/Romance offered

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "feel-good movies released in 2000 or earlier",
  "filters": null,
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": "feel-good",
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": false,
  "missing_slots": [
    "audience",
    "genres"
  ],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.9
}
```
- disposition: no changes
- turn decision: **retrieve** (axes ['mood', 'era'] >= 2)
- merged prefs: mood=feel-good, audience=-, genres=-, years=(None, 2000)
- clarifying question: -
- usage: in=781 out=140 (reasoning=4)

### Turn 3: `all of them`

**v1 did:** v1: 4-genre INTERSECTION (genre_match=all) -> exactly 1 movie

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "feel-good movies released before 2000",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": 2000,
    "genres": [],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": "old",
  "decade": null,
  "preference_delta": {
    "set_mood": "feel-good",
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": false,
  "missing_slots": [
    "audience",
    "genres"
  ],
  "clarifying_question": "Happy to show everything that fits \u2014 did you mean all feel-good films from before 2000, or all of something else?",
  "referenced_titles": [],
  "confidence": 0.4
}
```
- disposition: no changes
- turn decision: **retrieve** (explicit filters present)
- merged prefs: mood=feel-good, audience=-, genres=-, years=(None, 2000)
- clarifying question: Happy to show everything that fits — did you mean all feel-good films from before 2000, or all of something else?
- usage: in=782 out=248 (reasoning=30)

### Turn 4: `just me`

**v1 did:** v1: audience=solo; retrieved Hannah Montana: The Movie (2009) - era never applied, single-genre-deep result

```json
{
  "intent": "ATTRIBUTE_FILTER",
  "standalone_query": "feel-good movies to watch solo",
  "filters": null,
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": "solo",
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.9
}
```
- disposition: no changes
- turn decision: **retrieve** (ready + at least one axis known)
- merged prefs: mood=feel-good, audience=solo, genres=-, years=(None, 2000)
- clarifying question: -
- usage: in=780 out=141 (reasoning=15)

## C - #26 walkthrough (mid-session: mood=scary, audience=kids)

### Turn 1: `suggest me horror movies`

**v1 did:** v1: chip OUT_OF_SCOPE 1.00 + movie cards shown (Scream 1996, Zombie Kids 2013) - incoherent

```json
{
  "intent": "ATTRIBUTE_FILTER",
  "standalone_query": "scary horror movies suitable for kids",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [
      "Horror"
    ],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [
      "Horror"
    ],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [
    "era"
  ],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.9
}
```
- disposition: no changes
- turn decision: **retrieve** (explicit filters present)
- merged prefs: mood=scary, audience=kids, genres=['Horror'], years=(None, None)
- clarifying question: -
- usage: in=778 out=1869 (reasoning=1599)

### Turn 2: `show me horror movies for kids`

**v1 did:** v1: chip SEMANTIC_SEARCH 1.00, 5 movies retrieved, but pivot text 'outside my reel' shown

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "scary horror movies suitable for kids",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [
      "Horror"
    ],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.9
}
```
- disposition: no changes
- turn decision: **retrieve** (explicit filters present)
- merged prefs: mood=scary, audience=kids, genres=['Horror'], years=(None, None)
- clarifying question: -
- usage: in=780 out=253 (reasoning=12)

### Turn 3: `scary movies for kids`

**v1 did:** v1: chip OUT_OF_SCOPE 1.00, 0 movies, pivot text

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "scary movies for kids",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.95
}
```
- disposition: no changes
- turn decision: **retrieve** (ready + at least one axis known)
- merged prefs: mood=scary, audience=kids, genres=['Horror'], years=(None, None)
- clarifying question: -
- usage: in=779 out=290 (reasoning=13)

### Turn 4: `horror movies`

**v1 did:** v1: ATTRIBUTE_FILTER 1.00, 5 grounded cards - correct

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "kid-friendly horror movies that are scary but appropriate for children",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": null,
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.9
}
```
- disposition: no changes
- turn decision: **retrieve** (ready + at least one axis known)
- merged prefs: mood=scary, audience=kids, genres=['Horror'], years=(None, None)
- clarifying question: -
- usage: in=776 out=302 (reasoning=35)

### Turn 5: `show me horror movies for kids`

**v1 did:** v1: ATTRIBUTE_FILTER 1.00, 0 movies, pivot text - incoherent

```json
{
  "intent": "SEMANTIC_SEARCH",
  "standalone_query": "horror movies suitable for kids, spooky but not too frightening",
  "filters": {
    "exact_year": null,
    "year_min": null,
    "year_max": null,
    "genres": [
      "Horror"
    ],
    "genre_match": "any",
    "director": null,
    "cast_member": null,
    "person": null,
    "excluded_genres": [],
    "excluded_actors": [],
    "runtime_max": null,
    "rating_min": null
  },
  "era": null,
  "decade": null,
  "preference_delta": {
    "set_mood": null,
    "clear_mood": false,
    "set_audience": "kids",
    "add_genres": [],
    "remove_genres": [],
    "add_excluded_genres": [],
    "add_excluded_actors": [],
    "revoke_exclusions": [],
    "add_donts": []
  },
  "reset_context": false,
  "ready_to_retrieve": true,
  "missing_slots": [],
  "clarifying_question": null,
  "referenced_titles": [],
  "confidence": 0.9
}
```
- disposition: no changes
- turn decision: **retrieve** (explicit filters present)
- merged prefs: mood=scary, audience=kids, genres=['Horror'], years=(None, None)
- clarifying question: -
- usage: in=780 out=260 (reasoning=14)

## Totals

- schema-failure fallbacks (C12): 0
- tokens: in=10227, out=7204