EXAMPLEDIR=$(dirname $0)
ROOTDIR=$EXAMPLEDIR/../..

export DOCKER_USER=nilabhra
export PIRLIB_REPO=tuning

# Dockerize current environment
python $ROOTDIR/bin/pircli dockerize \
    $ROOTDIR \
	--auto \
    --pipeline examples.tuning.ml_pipeline:ml_job \
	--output $EXAMPLEDIR/package_argo.yml \
	--flatten

# Convert EXAMPLEDIR to absolute path since docker can't bind-mount relative paths.
EXAMPLEDIR=$([[ $EXAMPLEDIR = /* ]] && echo "$EXAMPLEDIR" || echo "$PWD/${EXAMPLEDIR#./}")

# Create necessary folders
mkdir -p $EXAMPLEDIR/outputs
mkdir -p $EXAMPLEDIR/cache_dir

INPUT_raw_data=$EXAMPLEDIR/data \
INPUT_preproc_hp=$EXAMPLEDIR/data/preprocess_hp.json \
INPUT_fe_hp=$EXAMPLEDIR/data/fe_hp.json \
INPUT_base_model_hp=$EXAMPLEDIR/data/base_model_hp.json \
INPUT_meta_model_hp=$EXAMPLEDIR/data/meta_model_hp.json \
CACHE=$EXAMPLEDIR/cache_dir \
OUTPUT=$EXAMPLEDIR/outputs \
NFS_SERVER=k8s-master.cm.cluster \
python  $ROOTDIR/bin/pircli generate $EXAMPLEDIR/package_argo.yml \
	--target pirlib.backends.argo_batch:ArgoBatchBackend \
	--output $EXAMPLEDIR/argo-tuning-pipeline.yml