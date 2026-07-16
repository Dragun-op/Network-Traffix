Endpoints to implement:

### `GET /api/health`

No params.

```json
{ "status": "ok", "time": "2026-07-16T18:13:36.752320+00:00" }
```

### `GET /api/incidents`

Query params (all optional): `severity` (repeatable — `?severity=high&severity=critical`), `attack_type`, `q` (matches source or destination IP), `status`, `limit` (default 50), `offset` (default 0).

```json
{
  "total": 42,
  "items": [
    {
      "id": "INC-F20493",
      "timestamp": "2026-07-16T16:04:38.100012+00:00",
      "src_ip": "45.53.71.50",
      "dst_ip": "82.92.210.193",
      "protocol": "UDP",
      "attack_type": "Port Scan",
      "severity": "critical",
      "confidence": 96,
      "status": "new",
      "packet_count": 19462,
      "explanation": null
    }
  ]
}
```

- `total` is the count _after_ filtering, before pagination — the frontend uses it for "Showing X of Y."
- `severity` ∈ `low | medium | high | critical`, `status` ∈ `new | investigating | resolved`.
- `explanation` is always `null` here — the list endpoint deliberately omits it to stay light.

### `GET /api/incidents/{id}`

Path param: `id` (e.g. `INC-F20493`).
Same shape as one item above, but `explanation` is populated:

```json
{
  "id": "INC-F20493",
  "timestamp": "2026-07-16T16:04:38.100012+00:00",
  "src_ip": "45.53.71.50",
  "dst_ip": "82.92.210.193",
  "protocol": "UDP",
  "attack_type": "Port Scan",
  "severity": "critical",
  "confidence": 96,
  "status": "new",
  "packet_count": 19462,
  "explanation": [
    { "feature": "SYN Flag Count", "contribution": 52 },
    { "feature": "Flow Duration", "contribution": 31 },
    { "feature": "Bwd Packets/s", "contribution": 17 }
  ]
}
```

Contributions in `explanation` sum to 100 and are pre-sorted descending — the frontend just renders them as bars in that order. 404 with `{"detail": "Incident not found"}` if the id doesn't exist.

### `PATCH /api/incidents/{id}`

Body:

```json
{ "status": "investigating" }
```

Returns the full updated incident (same shape as the GET-by-id response, `status` must be one of `new | investigating | resolved`). 400 if the status value is invalid, 404 if the id doesn't exist. _Not yet wired up in the frontend UI — endpoint exists for when you add a status-change control._

### `GET /api/summary`

No params.

```json
{ "total": 42, "low": 7, "medium": 8, "high": 13, "critical": 14 }
```

Drives the five stat cards at the top of the dashboard.
