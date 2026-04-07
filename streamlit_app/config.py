from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Langfuse CSV export time window defaults (used by `export_langfuse_csv.py`).
#
# Keep an end buffer to avoid partially indexed fresh data.
LANGFUSE_EXPORT_END_BUFFER: timedelta = timedelta(minutes=15)

# Earliest timestamp to export from.
LANGFUSE_EXPORT_START_DT: datetime = datetime(2026, 3, 8, tzinfo=timezone.utc)

