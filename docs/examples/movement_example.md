# Example 3: Movement Analysis (OSMnx)

This example demonstrates how to use GIS-MCP's movement tools (powered by OSMnx) to download street networks and calculate shortest paths.

---

## Downloading a Street Network

**Prompt:**

```
Using gis-mcp download the street network for Berlin, Germany for driving and save as GraphML.
```

**Action:**

- The tool will use OSMnx to download the drivable street network for Berlin and save it as a GraphML file.

---

## Calculating a Shortest Path

**Prompt:**

```
Using gis-mcp calculate the shortest path between (52.5200, 13.4050) and (52.5155, 13.3777) using the saved Berlin driving network.
```

**Action:**

- The tool will load the saved GraphML file and compute the shortest path between the two points using OSMnx and NetworkX.

---

## Notes

- OSMnx supports various network types: "drive", "walk", "bike", etc.
- The output path is a list of node IDs; you can visualize it using OSMnx plotting functions or export to other formats.
