## Data and labels

**Source.** The Kepler Q1-Q17 DR24 threshold crossing event (TCE) table from the NASA Exoplanet Archive, fetched through its TAP service (`q1_q17_dr24_tce`): 20,367 TCEs. The DR25 table was rejected because its `av_training_set` column is empty for all 34,032 rows, so it carries no usable labels.

**Labels.** The label is `av_training_set`, the Autovetter training-set class (Catanzarite 2015, KSCI-19091-001):

| Class | Meaning | Count | Binary label |
|---|---|---|---|
| PC | planet candidate | 3,600 | 1 |
| AFP | astrophysical false positive (e.g. eclipsing binaries) | 9,596 | 0 |
| NTP | non-transiting phenomenon (noise, instrumental artefacts) | 2,541 | 0 |
| UNK | not in the training set, no trusted label | 4,630 | excluded |

15,737 TCEs are labeled, of which 22.9% are planet candidates (about 1 planet to 3.4 non-planets). The three-way `disposition` is kept alongside the binary `label` so errors can later be split into confusions with eclipsing binaries and confusions with noise. UNK TCEs are kept in a separate table, used only to mask other signals on the same star during preprocessing, never for training, validation or testing.

**Star-level profile.** The 15,737 labeled TCEs come from 9,865 unique stars. Most stars have one TCE (5,505) or two (3,186); 907 have three, 208 have four, 50 have five, and 9 have six to eight. Because 4,360 stars (44.2%) have more than one TCE, the train/validation/test split is made by star, never by TCE, so that no star appears on both sides. 76 stars (0.8%) have TCEs with mixed labels, so a star-level label ("has at least one planet candidate") is well defined for almost every star. 2,716 stars (27.5%) have at least one planet candidate; this is the quantity the split is stratified on. 1,140 stars (11.6%) also have UNK TCEs, so masking other signals on the same star will matter for about one star in nine.

**Inputs.** Each TCE's period (days), epoch (BKJD) and duration (hours, converted to days) are used to fold its light curve. The model sees only the folded light-curve views; no catalog columns are model inputs.

**Label provenance and limitations.** The labels are not pure ground truth. They come from the Autovetter training set, which combines human vetting with automatic rules (Catanzarite 2015, section 6): for example, 135 planet candidates with radius above 25 Earth radii were relabeled AFP, TCEs failing a bootstrap false-alarm test (threshold 1e-11) were labeled NTP, TCEs with a spurious 459-day period were labeled NTP, and 1,437 TCEs matched to eclipsing-binary contaminators on other stars were labeled AFP. Two consequences: a model could learn the rules rather than the physics, and some labels depend on information that is not in a single star's light curve, so some errors are not reducible. Results apply to the labeled subset only.

**Leakage-prone columns.** The Autovetter outputs (`av_vf_*`, `av_pp_*`, `av_pred_class`) come from a classifier trained on these labels, and the bootstrap statistics (`boot_*`) and planet radius (`tce_prad`) helped define labels. None of these is used as an input or as a baseline feature.

**Placeholder fit values.** 1,059 TCEs (5.2%) carry placeholder values from failed transit-model fits: `tce_model_snr` = -1, of which 377 also report zero transits, and two report zero depth. These rows are almost all non-planets: among the 783 flagged labeled TCEs, 779 are non-planets (PC 4, AFP 651, NTP 128). A model given the placeholder could learn "not a planet" from it, which is leakage, so the rows are kept but the placeholders are stored as missing values, a `fit_failed` flag marks them for analysis only, and SNR, depth, transit count and the flag are never used as model or baseline inputs. Their period, epoch and duration look valid. To test for a shortcut through the folded views, results are reported both overall and excluding `fit_failed` rows.