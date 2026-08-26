#!/usr/bin/env python3
"""Export FL and TSF paper lists from neko-library's canonical catalog."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from pathlib import Path


class PaperExporter:
    """Export canonical papers into separate FL and TSF CSV files."""

    COLUMNS = [
        "ID",
        "Title",
        "Venue",
        "Year",
        "URL",
        "DOI",
        "OpenReview",
        "Arxiv",
        "Code",
        "Video",
        "Slide",
        "Poster",
    ]
    QUERIES = {
        "fl": "federat*",
        "tsf": (
            '"time series" OR timeseries OR forecast* OR temporal* OR '
            'spatiotemporal* OR "sequence model" OR (trend AND predict*)'
        ),
    }

    @staticmethod
    def connect(db: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def selected_work_ids(connection: sqlite3.Connection, query: str) -> set[int]:
        return {
            work_id
            for (work_id,) in connection.execute(
                """
                SELECT DISTINCT wm.work_id
                FROM metadata_fts
                JOIN work_members AS wm USING (target_id)
                WHERE metadata_fts MATCH ?
                """,
                (query,),
            )
        }

    @staticmethod
    def preferred_member(members: list[sqlite3.Row]) -> sqlite3.Row:
        return max(
            members,
            key=lambda member: (
                member["publisher"].lower() not in {"arxiv", "openreview", "other"},
                bool(member["doi"]),
                bool(member["venue"]),
                bool(member["year"]),
            ),
        )

    @staticmethod
    def source_url(members: list[sqlite3.Row], publisher: str) -> str:
        for member in members:
            if member["publisher"].lower() == publisher and member["url"]:
                return PaperExporter.normalize_url(value=member["url"])
        return ""

    @staticmethod
    def doi_url(members: list[sqlite3.Row]) -> str:
        doi = next((member["doi"] for member in members if member["doi"]), "")
        doi = doi.removeprefix("doi:").strip()
        if not doi or doi.startswith(("http://", "https://")):
            return doi
        return f"https://doi.org/{doi}"

    @staticmethod
    def normalize_url(value: str) -> str:
        value = value.strip()
        if value.startswith("10.") and "/" in value:
            return f"https://doi.org/{value}"
        return value

    @staticmethod
    def make_row(
        work_id: int, title: str, members: list[sqlite3.Row]
    ) -> dict[str, str]:
        preferred = PaperExporter.preferred_member(members=members)
        arxiv = PaperExporter.source_url(members=members, publisher="arxiv")
        openreview = PaperExporter.source_url(
            members=members,
            publisher="openreview",
        )
        doi = PaperExporter.doi_url(members=members)
        url = PaperExporter.normalize_url(
            value=preferred["url"]
            or next((member["url"] for member in members if member["url"]), ""),
        )
        if url in {arxiv, openreview, doi}:
            url = ""
        return {
            "ID": str(work_id).zfill(5),
            "Title": html.unescape(title or preferred["title"]),
            "Venue": html.unescape(preferred["venue"]),
            "Year": preferred["year"],
            "URL": url,
            "DOI": doi,
            "OpenReview": openreview,
            "Arxiv": arxiv,
            "Code": "",
            "Video": "",
            "Slide": "",
            "Poster": "",
        }

    @staticmethod
    def first_field_value(value_json: str) -> str:
        value = json.loads(value_json)
        values = value if isinstance(value, list) else [value]
        return next((item for item in values if isinstance(item, str) and item), "")

    @staticmethod
    def build_rows(
        connection: sqlite3.Connection,
        topic_ids: dict[str, set[int]],
    ) -> dict[str, list[dict[str, str]]]:
        all_ids = sorted(set().union(*topic_ids.values()))
        connection.execute(
            "CREATE TEMP TABLE selected_work(work_id INTEGER PRIMARY KEY, fl INTEGER, tsf INTEGER)"
        )
        connection.executemany(
            "INSERT INTO selected_work VALUES(?,?,?)",
            (
                (work_id, work_id in topic_ids["fl"], work_id in topic_ids["tsf"])
                for work_id in all_ids
            ),
        )

        rows_by_id: dict[int, dict[str, str]] = {}
        current_id = -1
        current_title = ""
        members: list[sqlite3.Row] = []
        for member in connection.execute("""
            SELECT sw.work_id, w.canonical_title, m.publisher, m.title,
                   m.year, m.venue, m.url, m.doi
            FROM selected_work AS sw
            JOIN works AS w USING (work_id)
            JOIN work_members AS wm USING (work_id)
            JOIN metadata AS m USING (target_id)
            ORDER BY sw.work_id, wm.member_order
            """):
            if member["work_id"] != current_id and members:
                rows_by_id[current_id] = PaperExporter.make_row(
                    work_id=current_id,
                    title=current_title,
                    members=members,
                )
                members = []
            current_id = member["work_id"]
            current_title = member["canonical_title"]
            members.append(member)
        if members:
            rows_by_id[current_id] = PaperExporter.make_row(
                work_id=current_id,
                title=current_title,
                members=members,
            )

        for field in connection.execute("""
            SELECT wf.work_id, wf.field, wf.value_json
            FROM selected_work AS sw
            JOIN work_fields AS wf USING (work_id)
            WHERE wf.field IN ('code','video','slide','poster')
            """):
            rows_by_id[field["work_id"]][field["field"].title()] = (
                PaperExporter.first_field_value(value_json=field["value_json"])
            )

        return {
            topic: [rows_by_id[work_id] for work_id in sorted(work_ids)]
            for topic, work_ids in topic_ids.items()
        }

    @staticmethod
    def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=PaperExporter.COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--library",
            type=Path,
            default=Path(r"E:\neko941\neko-library"),
        )
        parser.add_argument(
            "--out-dir", type=Path, default=Path(__file__).resolve().parent
        )
        args = parser.parse_args()
        db = args.library / "papers" / "catalog.sqlite"
        if not db.is_file():
            parser.error(f"paper catalog not found: {db}")

        with PaperExporter.connect(db=db) as connection:
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata_fts'"
            ).fetchone():
                parser.error(
                    "metadata_fts is missing; rebuild neko-library's paper search index"
                )
            topic_ids = {
                topic: PaperExporter.selected_work_ids(
                    connection=connection,
                    query=query,
                )
                for topic, query in PaperExporter.QUERIES.items()
            }
            rows = PaperExporter.build_rows(
                connection=connection,
                topic_ids=topic_ids,
            )

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for topic, topic_rows in rows.items():
            PaperExporter.write_csv(
                path=args.out_dir / f"papers-{topic}.csv",
                rows=topic_rows,
            )
        print(
            json.dumps({topic: len(topic_rows) for topic, topic_rows in rows.items()})
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(PaperExporter.main())
