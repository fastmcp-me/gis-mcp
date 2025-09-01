### Movement Data (via `osmnx`)

Currently, the **Movement Data** tool in GIS-MCP allows you to download and analyze street networks and perform routing using [OSMnx](https://osmnx.readthedocs.io/en/stable/).

---

#### Installation

To enable movement data tools, install GIS-MCP with the **movement** extra:

```bash
pip install gis-mcp[movement]
```

---

#### Available Operations

- `download_street_network` – Download a street network for a given place and save as GraphML.
- `calculate_shortest_path` – Calculate the shortest path between two points using a saved street network.

---

#### Example Usage

**Prompt:**

```bash
Using gis-mcp download the street network for Berlin, Germany for driving and save as GraphML.
```

**Prompt:**

```bash
Using gis-mcp calculate the shortest path between (52.5200, 13.4050) and (52.5155, 13.3777) using the saved Berlin driving network.
```
