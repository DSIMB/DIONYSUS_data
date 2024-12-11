# DIONYSUS_data

Additional source data & code used to generate and analyze content of DIONYSUS database : www.dsimb.inserm.fr/DIONYSUS/

Data:
* Sequence clusters in terms of MMseqs2 score: `data/binding_sites_mmseqs2_clusters.csv`
* Table downloadable from Clusters page: `data/binding_sites_mmseqs2_clusters.csv`
* List of core carbohydrates: `data/core_carbohydrates.json` 
* List of all the carbohydrates: `data/carbohydrate_components.csv`
* Pairwise scores of high-quality interfaces in terms of geometrical resemblance (source code for the comparison script was already available at www.github.com/DSIMB/CompareCBS): `data/ava.score`

Code:
* Script for selection of protein residues involved in binding site formation: `PDBParser.py` 
* Source code for clustering algorithm used in our study `hierarchical_spectral_clustering.py`
