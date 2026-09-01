# Statistics provenance

Where the numbers quoted in the manuscript actually come from. Started 15-07-26 after
losing track of which notebook cell a reported statistic came from while writing the
thesis and updating the paper at the same time. Add an entry here whenever you (or an
agent) compute a number that ends up quoted in manuscript/thesis text, especially if it
was computed ad hoc rather than read straight off an existing notebook cell.

Format per entry: manuscript claim -> exact numbers -> source file + cell/section ->
any caveats.

---

## PFM / NPC dispersion — network area, correlation & group comparison (new fsLR32k pipeline)

**Where it's used:** `/Users/mrenke/obsidian-wiki/projects/dnumrisk/rework_neural_section_13-07-26.tex`,
Results section, second and third paragraphs (PFM / NPC dispersion paragraphs).

**Source data:**
- `/Users/mrenke/data/ds-dnumrisk/derivatives/phenotype/netsPFM_NPC_allNets_atlas-caNets_DDnr_method-individual_area.csv`
  — per-subject network area (cm²) within the NPC mask, new fsLR32k/Infomap pipeline.
- `/Users/mrenke/data/ds-dnumrisk/derivatives/phenotype/netsPFM_wholeBrain_allNets_atlas-caNets_DDnr_method-individual_area.csv`
  — same, whole-brain (for supplementary table 1 replacement).
- `/Users/mrenke/data/ds-dnumrisk/derivatives/phenotype/NPC_dispersion_2D_final.csv` — NPC gradient dispersion per subject.
- `/Users/mrenke/data/ds-dnumrisk/group_assignment.csv` — group labels.

**Source notebook (reproducible):**
`/Users/mrenke/git/parietal_patterns/prep_results/thesis_ch-DD-neuro_01.ipynb`
- Cell `a2a4fd0f` — loads/joins NPC network area + dispersion.
- Cell `91b912a9` — Spearman correlation, NPC dispersion vs. NPC network area (all networks).
- Cell `a44403f0` — mean/SD of NPC network area, all networks.
- Cell `79328db3` / `2b1e259c` (added 15-07-26) — N subjects per network within the NPC
  mask; shows only Visual2, Dorsal-attention, Somatomotor are present in all 66 subjects
  — the other four networks (cingulo-opercular, default mode, frontoparietal, auditory)
  are sparsely represented and excluded from the manuscript text for that reason.
- Cell `e8f0e602` — group comparison (t-test), NPC network area.
- Cell `682270ad` / `2f7598ad` (added 15-07-26) — group comparison (**Mann-Whitney U**,
  matching the convention used elsewhere in the manuscript/Table 1), for the three
  universally-represented networks. This is the source of the exact numbers quoted in
  the manuscript: visual-2 U(33,33)=295.00, p=0.001; dorsal attention U(33,33)=761.00,
  p=0.006.

**Whole-brain group comparison table** (manuscript's "supplementary table 1"):
computed directly from `netsPFM_wholeBrain_allNets_...` (not from any notebook cell —
`network_size_analysis.ipynb` in `parietal_patterns/nets_PFM/` has an equivalent but
stale/superseded cell 25). CSV saved at
`/Users/mrenke/obsidian-wiki/projects/dnumrisk/supp_table1_wholebrain_PFM_newpipeline_15-07-26.csv`.

**Caveats:**
- As of 15-07-26, the phenotype CSVs above were regenerated (mtime ~15:58-16:03) *after*
  `thesis_ch-DD-neuro_01.ipynb`'s cell outputs had last been saved — so the notebook's
  displayed/cached output was briefly stale relative to disk. Re-run cells if in doubt
  rather than trusting a screenshot of old output.
- Default mode's correlation with NPC dispersion was significant under the old
  fsaverage5 pipeline but is **not** significant under this new pipeline (r=0.21,
  p=0.118) — it has been dropped from the manuscript text for this reason, not because
  it was overlooked.

---

## nPRF model $R^2$ (variance explained) — NPC extent-of-fit, group comparison

**Where it's used:** `/Users/mrenke/obsidian-wiki/chapters/2c_study_dyscalculia_neural_23-07-26.tex`,
Results, "Representational account: nPRF modelling — spatial extent" paragraph (and the
corresponding Methods addition to the "Numerical pRF modelling" paragraph). This is the
extent-of-fit test that resolves the old `% R2 maps?!` TODO in earlier chapter drafts —
distinct from decoding precision `r` (null group difference, reported earlier in the
same Results paragraph), this asks how much of the NPC clears a usable model fit at all.

**Source notebook (originally sciencecloud-only; as of 23/24-07-26 the resulting
per-subject values are exported and local):**
`/Users/mrenke/git/parietal_patterns/prep_results/thesis_ch-DD-neuro_02_scloud.ipynb`
- Cells `f11ec4ea` / `007efeb3` — setup; loads the same shared, group-level NPC mask
  (`get_NPC_mask(space='fsaverage5', hemi='both')`, plus per-hemisphere variants) already
  used to define the NPC mask in the Methods paragraph above — not a new/different mask.
- Cell `2f88b8f7` — per-subject `mean_R2` (bilateral, left, right) and vertex-count
  `area(r2>threshold)` at thresholds 0.01, 0.05, 0.1 (bilateral only — the L/R split for
  the area measure is written in this cell but commented out, so not part of the
  executed/saved result).
- Cell `2ba4d267` — group comparison via `parietal_patterns.utils.statistics.between_group_comparison`
  for all six measures. Source of the exact numbers quoted in the chapter.
- Cell `9a11c086` — Pearson correlation of each extent measure with decoding precision `r`.
- Cells `12`/`13` (`009909f2`/`9d8c0391`) — group-average $R^2$ surface plots (Control vs.
  DD, both hemispheres) — the source material for the figure placeholder in the chapter
  (`fig:npc-r2-map`); not yet exported to a standalone file, pending a `savefig` call
  Maiken is adding on the remote machine.

**Exported local CSV (as of 23-07-26):**
`/Users/mrenke/data/ds-dnumrisk/derivatives/phenotype/nPRF_r2area.csv` — one row per
subject, columns `subject, group, mean_R2, mean_R2_L, mean_R2_R, area(r2>0.01),
area(r2>0.05), area(r2>0.1)`. This is what closed the earlier missing-mean±SD gap
(`df.groupby('group')[['mean_R2']].agg(['mean','std'])` on this file gives control:
0.023±0.007 (N=32), DD: 0.020±0.006 (N=33) — now quoted in the chapter text) and what
`thesis_ch-DD-neuro_01.ipynb` cell `6b5fdeb4` loads for the cross-measure correlation
figures (see the next entry below).

**Caveats:**
- N=32 control / 33 DD — one subject dropped upstream in cell `f11ec4ea`
  (`subList = subList[subList != 3]`). The cell's own comment says "remove sub-09" but the
  code excludes subject index 3 (sub-03) — a labelling discrepancy in the notebook itself,
  not corrected here; worth checking which subject was actually meant to be excluded.
- The two hemisphere-split area measures (`area(r2>0.05)_L`, `p=0.053`; `area(r2>0.05)_R`,
  `p=0.014`) mentioned in chat are still not present as columns in `nPRF_r2area.csv` —
  only the bilateral `area(r2>threshold)` measures were exported. Both remain omitted from
  the chapter text for this reason, not just the borderline left one.

---

## Cross-measure correlations — neural × neural, and behaviour × neural (partial, group removed)

**Where it's used:** `/Users/mrenke/obsidian-wiki/chapters/2c_study_dyscalculia_neural_23-07-26.tex`,
Results, "Correlations across neural measures" and "Correlations with behavioural
measures" paragraphs (the latter resolves that section's long-standing
"TODO: not yet run" for the partial-correlation/double-dipping check).

**Source notebook:** `/Users/mrenke/git/parietal_patterns/prep_results/thesis_ch-DD-neuro_01.ipynb`
- Cells `f09b8b07`/`2151951b`/`9d63179f` — builds the combined behaviour+neural dataframe
  (`df_bn`) from local loaders only: `numrisk.behavior_general.measures_registry`
  (`_load_magjudge_bauer`, `load_all`, `_load_decode_r`, `_load_npc_pfm_net_area`,
  `_load_npc_dispersion`) plus `nPRF_r2area.csv` and `gradient-2_SD.csv` from the
  phenotype dir directly.
- Cell `a1d572d2` — plain Spearman correlation + p-value, behaviour × neural (produces
  `cross_measure_heatmap_behav-neural.pdf` — an intermediate figure, not the one in the
  chapter).
- Cells `68acc735`/`62520be8` — **partial** Spearman correlation controlling for `group`,
  squared to r² (variance explained); saved as
  `cross_measure_partial_r2_behav-neural_group-removed.pdf` — this is the figure quoted
  in the chapter.
- Cell `045ea02a` — neural × neural Spearman correlation matrix; saved as
  `cross_measure_heatmap_neural.pdf`, then manually annotated with category divider
  lines in Affinity Designer and re-exported as
  `cross_measure_heatmap_neural_affDocVarGrouping.pdf` (the filename referenced in the
  chapter) — the annotated version is not reproducible from the notebook alone, only the
  underlying (unannotated) heatmap is.
- None of these cells print the underlying correlation/p-value matrices as text (only
  the heatmap figures) — the exact numbers quoted in the chapter prose were obtained by
  re-running the same computation (same loaders, same variable lists) directly, not read
  off any printed notebook cell output.

**Caveats:**
- Behavioural variables are restricted by design to the PMC-RDM noise-dissociation
  parameters (`perceptual_noise_sd`, `memory_noise_sd`, `a`; `t0` excluded) and
  standardized cognitive tests only (IQ, visuospatial WM, Weber fraction, math skill) —
  no questionnaires, raw task accuracy/RT, or probit parameters. This scoping is stated
  in cell `f09b8b07` and carried into the chapter's figure caption.
- Sample sizes vary by pair (N=64-66) depending on which neural/behavioural measures are
  available for a given subject; not restated per-cell in the chapter prose, only in this
  entry and in the figure caption's general note.

---

## DAN patch area — group comparison, all eight bilateral reference patches

**Where it's used:** paper rework — `dyscalc_paper_rework/sections/results.tex`
("Individual DAN topography" paragraph, three patches quoted in prose) and
`dyscalc_paper_rework/sections/supplements.tex` (Supplementary Table 2, all eight).
Equivalent to `tab:dan-patches` in
`obsidian-wiki/chapters/2c_study_dyscalculia_neural_11-08-26.tex`.

**Source data:**
- `/Users/mrenke/data/ds-dnumrisk/derivatives/phenotype/netsPFM_DANpatches_refAtlas-caNets_DDnr.csv`
  — per-subject, per-patch `ind_area` (cm², summed individual anatomical surface area) and
  `n_verts`, 513 rows / 66 subjects / 8 reference patches.
- `/Users/mrenke/data/ds-dnumrisk/group_mapping.csv` — group labels (0 = Control, 1 = Dyscalculic).

**Script (reproducible, added 13-08-26):**
`/Users/mrenke/obsidian-wiki/dyscalc_paper_rework/data/make_dan_patch_table.py`
→ writes `dyscalc_paper_rework/data/supp_table_DANpatches_group_comparison.csv`.
Test selection replicates `parietal_patterns.utils.statistics.between_group_comparison`
(normaltest on the pooled measure, then t-test or Mann-Whitney U at alpha=0.05);
Bonferroni and Benjamini-Hochberg FDR computed over the eight patches.

**Validation:** all eight rows reproduce the thesis table exactly — means, SDs, per-group N,
test choice, statistic, raw p, and both corrected p-values. Until 13-08-26 the numbers existed
only as hand-written LaTeX in the thesis chapter; this is the first time they are derivable
from source data.

**Caveats:**
- N varies by patch (58-66 total) because some reference patches are not recoverable in every
  participant's individual partition. L frontal-lateral — the largest effect — is 33/32, so it
  rests on one fewer subject than the parietal patches. Worth checking whether the missing
  subject is the same one dropped from the nPRF analyses (see the nPRF entry above, where the
  notebook's comment and code disagree about which subject was excluded).
- The two frontal-medial-dorsal and R temporal patches have the smallest N and the largest
  relative SDs; their nulls are correspondingly weakly constrained and should not be read as
  evidence of no effect.
