## Match the Modality, Mind the Transfer: Linguistically Calibrated Language Distances from URIEL+
### Source code for distance metrics

In `src/querying.py`, we provide classes to query language distances under each proposed representation, for each modality (geographic, genetic, typological). Example usage is provided in `examples.ipynb`.

*Note:* The inputs to `compute_distance` are assumed to be Glottocodes, taken from [Glottolog](https://glottolog.org/) [1].

### Supplementary code files
- `src/speaker_geographic/functions.py` contains all helper functions for constructing speaker distributions from data, and computing the Wasserstein distance between distributions.
- All code for learning hyperbolic embeddings are under `src/hyperbolic_genetic`. The main function is `train_embeddings()` in `src/hyperbolic_genetic/train_logic.py`.
- `src/typological/functions.py` contains all helper functions for learning the latent structure of the URIEL+ typological dataset, and for computing distances given islands.

### Supplementary data files
- `data/country_speaker_subset.csv` contains a subset of Ethnologue data containing language information, country centroids and per-country speaker counts for high- and medium-resource languages, as identified in URIEL+ [2].
- `data/genetic_distance_matrix.csv` contains pre-computed distances between languages in hyperbolic space.
- `data/genetic_adjacency_list.pkl` contains the adjacency list from the Glottolog genealogy tree, as used to learn hyperbolic embeddings.
- `data/islands.pkl` contains pre-computed latent islands for the URIEL+ typological dataset.
- `data/URIELPlus_Union_SoftImpute.csv` contains the URIEL+ typological dataset, taking union aggregation over sources, and with missing values imputed using SoftImpute.

## Licenses
We distribute our code under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY SA 4.0) license](https://creativecommons.org/licenses/by-sa/4.0/).

The URIEL+ typological data is originally distributed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY SA 4.0) license](https://creativecommons.org/licenses/by-sa/4.0/).

The Glottolog genealogy data is originally distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).

The language speakers data from Ethnologue is **partially** redistributed under the [SIL Fair Use Guidelines](https://www.sil.org/sites/default/files/files/sil-org1-fairuseguidelines_0.pdf).

## References
[1] Harald Hammarström, Robert Forkel, Martin Haspelmath, & Sebastian Bank. 2025. Glottolog database 5.2. Zenodo. https://doi.org/10.5281/zenodo.15525265

[2] Aditya Khan, Mason Shipton, David Anugraha, Kaiyao Duan, Phuong H. Hoang, Eric Khiu, A. Seza Doğruöz, and En-Shiun Annie Lee. 2025. URIEL+: Enhancing Linguistic Inclusion and Usability in a Typological and Multilingual Knowledge Base. In Proceedings of the 31st International Conference on Computational Linguistics, pages 6937–6952, Abu Dhabi, UAE. Association for Computational Linguistics.