##### Directories Setup
SRUN_ARGS="--ntasks=$SLURM_NNODES --ntasks-per-node=1"
export XDG_CACHE_HOME=${SLURM_TMPDIR}/.cache
export XDG_CONFIG_HOME=${SLURM_TMPDIR}/.config
export TRITON_CACHE_DIR=${SLURM_TMPDIR}/.cache/triton
export TMPDIR=${SLURM_TMPDIR}/.cache/tmp
export VLLM_CACHE_ROOT=${SLURM_TMPDIR}/vllm_cache
export VLLM_CONFIG_ROOT=${SLURM_TMPDIR}/vllm_config
export VLLM_ASSETS_CACHE=${SLURM_TMPDIR}/assets_cache
export FLASHINFER_WORKSPACE_BASE=${SLURM_TMPDIR}/flashinfer
srun $SRUN_ARGS mkdir -p ${XDG_CACHE_HOME}
srun $SRUN_ARGS mkdir -p ${XDG_CONFIG_HOME}
srun $SRUN_ARGS mkdir -p ${TRITON_CACHE_DIR}
srun $SRUN_ARGS mkdir -p ${TMPDIR}
srun $SRUN_ARGS mkdir -p ${VLLM_CACHE_ROOT}
srun $SRUN_ARGS mkdir -p ${VLLM_CONFIG_ROOT}
srun $SRUN_ARGS mkdir -p ${VLLM_ASSETS_CACHE}
srun $SRUN_ARGS mkdir -p ${FLASHINFER_WORKSPACE_BASE}

# pip install --user proxy.py
# kill -9 $(lsof -t -i:8899)
# proxy --hostname 0.0.0.0 --port 8899

kill -9 $(lsof -t -i:8899)
if [[ "$hostname" == *trig* ]]; then
  export http_proxy=http://trig-login01:8899
else
  export http_proxy=http://rorqual1:8899
fi
export https_proxy=$http_proxy
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

##### Env Setup
module --force purge all
module load StdEnv/2023  nvhpc/23.9  openmpi/4.1.5
module load cuda/12.2
source /scratch/rohhs/venvs/SVFR/bin/activate

cd $PROJECTS_DIR/SVFR

##### Run the code

input_path=/scratch/rohhs/downloads/yt-dlp/biker.mp4
output_dir=/scratch/rohhs/downloads/yt-dlp/enhanced

export CUDA_VISIBLE_DEVICES=0

python infer.py --config config/infer.yaml --task_ids 0 --input_path $input_path --output_dir $output_dir --crop_face_region

pip install "diffusers==0.27.2" "huggingface-hub==0.23.4" "transformers==4.40.2" "accelerate==0.30.1" "safetensors==0.4.3"