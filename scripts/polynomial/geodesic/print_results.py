import json

data = json.load(open("scripts/polynomial/geodesic/mesh_comparison/comparison_results.json"))

configs = list(data.keys())

for name in configs:
    q = data[name]
    v = q["n_vertices"]
    f = q["n_faces"]
    bad = q["pct_bad"]
    vbad = q["pct_very_bad"]
    war = q["worst_ar"]
    p99 = q["p99_ar"]
    wang = q["worst_min_angle"]
    p1a = q["p1_min_angle"]
    d4 = q["pct_deg_le_4"]
    d3 = q["deg_le_3"]
    gb = q["gauss_bad"]
    t = q["elapsed"]
    print(f"{name:22s} V={v:>7} F={f:>9} bad={bad:5.1f}% vbad={vbad:5.2f}% wAR={war:9.1f} p99AR={p99:5.1f} wAng={wang:7.3f} p1Ang={p1a:5.1f} D4={d4:5.1f}% d3={d3:>4} Gbad={gb:5.1f}% {t:5.0f}s")
