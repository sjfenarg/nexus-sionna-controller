# PowerLog / 3GPP ensemble

Load any `scene_000.json` through `scene_199.json` in the GUI and click **Simulate**.
Every file keeps the PowerLog base station, UE antenna and radiomap area from
`configs/radiomap_db.json`. Radiomap computation is disabled in these requests
so the explicit trajectories are simulated.
The configured `Detailed 6D Map.xml` includes both tree instance meshes
(`TREE_V10_Final_00` and `TREE_V10_Final_01`); the generator verifies this.

Each file has 200 moving UEs: 40 lines, 40 arcs, 40 S curves, 40 zigzags,
20 circular loops and 20 elliptical loops. The loops are smooth, closed cubic
Bezier curves. Each scene has a fresh random layout of 1 to 5 moving 3GPP
targets drawn from humans, AGVs and cars. Target count cycles from 1 to 5,
giving 40 scenes of each count.

Cars use straight paths or gentle smooth arcs only. Their complete paths are
checked against the scene's `GROUND_Obj.ply` and `GRASS_Obj.ply` footprints and
the low-height tree-instance footprints, with a 1.5 m clearance from those
boundaries.

| Scene numbers | UE length distribution |
| --- | --- |
| 000–039 | Short |
| 040–079 | Medium |
| 080–119 | Long |
| 120–159 | Bimodal short/long |
| 160–199 | Broad |

`manifest.json` records each scene's target types, family counts and measured
path lengths. The PNG summaries are in `output/powerlog_3gpp_ensemble_200/`.
These requests simulate all 200 UEs simultaneously, so full runs can be large.

To regenerate the ensemble from the current radiomap source:

```powershell
$env:PYTHONPATH='src'
python scripts/create_powerlog_3gpp_ensemble.py
```
