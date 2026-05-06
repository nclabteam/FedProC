import csv
import math
import os

from .base import Topology


class MekongGeoKNN(Topology):
    """KNN topology for Mekong salinity stations using metadata coordinates."""

    k = 8
    river_bonus = 0.75

    def _station_metadata(self):
        metadata_path = os.path.join(
            "datasets",
            "MekongSalinity",
            "metadata",
            "hydrology_stations.csv",
        )
        rows = {}
        with open(metadata_path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                code = str(row["hydrology_station_code"]).strip()
                rows[code] = {
                    "lon": float(row["longtitude"]),
                    "lat": float(row["latitude"]),
                    "river": row.get("river_name", "").strip(),
                }
        return rows

    def _client_station_codes(self):
        info_path = os.path.join(
            "datasets",
            "MekongSalinity",
            "seq_14-offset_0-pred_7",
            "info.json",
        )
        if not os.path.exists(info_path):
            return []

        import json

        with open(info_path, encoding="utf-8") as handle:
            info = json.load(handle)

        codes = []
        for item in info[: self.num_nodes]:
            filename = os.path.basename(item["file"])
            codes.append(os.path.splitext(filename)[0])
        return codes

    @staticmethod
    def _haversine_km(a, b):
        radius = 6371.0
        lat1 = math.radians(a["lat"])
        lat2 = math.radians(b["lat"])
        d_lat = lat2 - lat1
        d_lon = math.radians(b["lon"] - a["lon"])
        h = (
            math.sin(d_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        )
        return 2 * radius * math.asin(math.sqrt(h))

    def _gen(self):
        metadata = self._station_metadata()
        codes = self._client_station_codes()
        if len(codes) != self.num_nodes:
            return self._fallback_ring()

        directed = {}
        for idx, code in enumerate(codes):
            current = metadata.get(code)
            if current is None:
                directed[idx] = [
                    other for other in range(self.num_nodes) if other != idx
                ][: self.k]
                continue

            scored = []
            for other_idx, other_code in enumerate(codes):
                if idx == other_idx:
                    continue
                other = metadata.get(other_code)
                if other is None:
                    continue
                distance = self._haversine_km(current, other)
                same_river = current["river"] and current["river"] == other["river"]
                score = distance * (self.river_bonus if same_river else 1.0)
                scored.append((score, other_idx))
            directed[idx] = [other for _, other in sorted(scored)[: self.k]]

        neighbors = {node: set(values) for node, values in directed.items()}
        for node, values in directed.items():
            for neighbor in values:
                neighbors.setdefault(neighbor, set()).add(node)

        return {
            node: sorted(neighbors.get(node, set()) - {node})
            for node in range(self.num_nodes)
        }

    def _fallback_ring(self):
        neighbors = {}
        for node in range(self.num_nodes):
            values = set()
            for step in range(1, min(self.k, self.num_nodes - 1) + 1):
                values.add((node - step) % self.num_nodes)
                values.add((node + step) % self.num_nodes)
            values.discard(node)
            neighbors[node] = sorted(values)
        return neighbors
