DATASET=COVIDNEWS

python -u perturb_labels.py --dataset_dir $DATASET --perturb_function perturb_label_stratified
