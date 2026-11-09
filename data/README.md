# Data

The two SNAP signed trust networks are public but are not redistributed in this repository.
Run `python scripts/download_data.py`, or download them manually into this folder:

| file | source |
|---|---|
| `soc-sign-bitcoinotc.csv.gz` | https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html (5,881 users, 35,592 ratings) |
| `soc-sign-bitcoinalpha.csv.gz` | https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html (3,783 users, 24,186 ratings) |

Format: `source,target,rating,time` (ratings in -10..+10, Unix timestamps). Kumar et al., ICDM 2016 / WSDM 2018.
Set `CALITRUST_DATA` to use another folder.
