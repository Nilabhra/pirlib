EXAMPLEDIR=$(dirname $0)
ROOTDIR=$EXAMPLEDIR/../..

export DOCKER_USER=ridzy619
export PIRLIB_REPO=tuning

# Dockerize current environment
python $ROOTDIR/bin/pircli dockerize \
    $ROOTDIR \
	--auto \
    --pipeline examples.stacking.pipeline:ml_job \
	--output $EXAMPLEDIR/package_argo.yml \
	--flatten

# Convert EXAMPLEDIR to absolute path since docker can't bind-mount relative paths.
EXAMPLEDIR=$([[ $EXAMPLEDIR = /* ]] && echo "$EXAMPLEDIR" || echo "$PWD/${EXAMPLEDIR#./}")

# Create necessary folders
mkdir -p $EXAMPLEDIR/outputs
mkdir -p $EXAMPLEDIR/cache_dir

INPUT_train_path=$EXAMPLEDIR/inputs/train.csv \
INPUT_prev_app_path=$EXAMPLEDIR/inputs/previous_application.csv \
INPUT_test_path=$EXAMPLEDIR/inputs/test.csv \
INPUT_preproc_hp=$EXAMPLEDIR/inputs/preproc_hp.json \
INPUT_base_model_hp=$EXAMPLEDIR/inputs/base_model_hp.json \
INPUT_meta_model_hp=$EXAMPLEDIR/inputs/meta_model_hp.json \
CACHE=$EXAMPLEDIR/cache_dir \
OUTPUT=$EXAMPLEDIR/outputs \
NFS_SERVER=k8s-master.cm.cluster \
python  $ROOTDIR/bin/pircli generate $EXAMPLEDIR/package_argo.yml \
	--target pirlib.backends.argo_batch:ArgoBatchBackend \
	--output $EXAMPLEDIR/argo-stacking-pipeline.yml