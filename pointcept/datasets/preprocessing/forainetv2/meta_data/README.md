`train_list.txt`, `val_list.txt` and `test_list.txt` are the official FOR-instanceV2 split lists (47 training,
16 validation and 28 test plots), copied unchanged from `data/ForAINetV2/meta_data/` of ForestFormer3D
(https://github.com/SmartForest-no/ForestFormer3D, commit 6a75c37), which is licensed under CC BY-NC 4.0.
`scripts/prepare_forest.sh` copies them next to the data (the train and val lists with `--train`). The three
`BlueCat_RN_merged_trees_*` plots of the Zenodo archives are in none of the lists, so they are neither trained on
nor evaluated. The order of the lines of `test_list.txt` is the evaluation order; keep it to reproduce the
reported numbers exactly.
