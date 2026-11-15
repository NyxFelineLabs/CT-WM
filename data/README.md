# Dataset access and file isolation

Download [Video Streaming Network KPIs for O-RAN Testing](https://doi.org/10.21227/v8p4-v974)
from IEEE DataPort using your own authorized access. No raw measurements, window
arrays, or credentials are redistributed. The repository's MIT license does not
replace the data provider's terms.

Place the **ten independent CSV files** in `data/raw/`, without concatenating
them. File names, row counts, split membership and SHA-256 fingerprints are in
[`manifest.reference.json`](manifest.reference.json).

Expected columns, in order:

```text
time,phy_mcs,mac_dl_cqi,mac_dl_ri,mac_dl_pmi,mac_ul_buffer,mac_n_prb,rsrq,rsrp,rssi,dl_sinr,se,dl_bler,delay
```

Run from the repository root:

```bash
python scripts/prepare_data.py --window 32
```

The preparation tool validates every file, builds windows within each file, and
records a **7 training / 1 validation / 2 test file split**. Its NPY arrays are
derived model inputs; raw CSVs remain separate and untouched. Window split
boundaries are file boundaries. `run_file_split.py` applies these boundaries
before invoking the unchanged training routine; train-only scaling is fitted by
that loader. Do not directly invoke the upstream trainer's default 70/15/15
window split for the file-isolated benchmark.

Use `--check-hashes` to require byte-identical inputs for historical comparison.
With different data, report new results rather than claiming exact reproduction
of the archived tables. An unavailable data file is an explicit error; the
program never silently substitutes synthetic measurements.
