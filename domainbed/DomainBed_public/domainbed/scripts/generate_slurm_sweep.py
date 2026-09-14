import os
import stat
import math
from datetime import datetime

def generate_slurm_sweep(args, jobs, sweep_output_dir,
                         job_name="Test",
                         role="classic",
                         device="a100",
                         account="ldz@a100",
                         gpus_per_task=1,
                         cpus_per_task=4,
                         time="02:30:00",
                         max_parallel=10,
                        ):

    parts = args.output_dir.strip(os.sep).split(os.sep)
    # Take last two folders
    job_name = "".join(parts[-3:])

    domainbed_dir = "/lustre/fswork/projects/rech/ldz/upr55xw/these/domainbed/DomainBed"

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_dir = os.path.join(sweep_output_dir, "slurm_out", date)
    logs_dir = os.path.join(slurm_dir, "logs")  
    os.makedirs(logs_dir, exist_ok=True)

    # 1️⃣ Écriture commands.txt
    if role == "incomplete":
        commands_file = os.path.join(sweep_output_dir, "commands_incomplete.txt")
    else:
        commands_file = os.path.join(sweep_output_dir, "commands.txt")
    with open(commands_file, "w") as f:
        for job in jobs:
            os.makedirs(job.output_dir, exist_ok=True)
            f.write(job.command_str + "\n")

    n_jobs = len(jobs)

    # Limite parallèle optionnelle
    array_str = f"0-{n_jobs-1}"
    if max_parallel is not None:
        array_str += f"%{max_parallel}"

    # 2️⃣ Génération script sbatch
    if role == "incomplete":
        slurm_script = os.path.join(sweep_output_dir, "submit_slurm_incomplete.sh")
    else:
        slurm_script = os.path.join(sweep_output_dir, "submit_slurm.sh")

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --constraint={device}
#SBATCH --chdir={domainbed_dir}
#SBATCH --ntasks=1
#SBATCH --gpus-per-task={gpus_per_task}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --hint=nomultithread
#SBATCH --time={time}
#SBATCH --output={logs_dir}/%x_%A_%a.out
#SBATCH --error={logs_dir}/%x_%A_%a.err
#SBATCH --array={array_str}


echo "Running on ${{SLURM_JOB_NODELIST}}"
echo "GPU(s): ${{SLURM_GPUS_ON_NODE}}"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
# load conda 
# module purge
# module load miniforge/24.9.0
# conda activate $SCRATCH/envs/domainbed
# module load ffmpeg/4.2.2


COMMAND_FILE="{commands_file}"
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $COMMAND_FILE)

echo "Executing:"
echo $CMD

echo "PWD: $(pwd)"
echo "Python: $(which python)"
echo "uv: $(which uv)"
echo "Project files:"
ls

echo ".venv:"
ls -ld .venv

echo "pyproject:"
ls pyproject.toml

set -ex
eval "$CMD"
echo $?
"""

    with open(slurm_script, "w") as f:
        f.write(script_content)

    # rendre exécutable
    st = os.stat(slurm_script)
    os.chmod(slurm_script, st.st_mode | stat.S_IEXEC)

    print(f"\n✔ Sweep prêt dans : {sweep_output_dir}")
    print(f"✔ Nombre de jobs : {n_jobs}")
    print(f"\nPour lancer :")
    print(f"cd {sweep_output_dir}")
    print(f"sbatch submit_slurm.sh")




def generate_slurm_sweep_dalia(
    args,
    jobs,
    sweep_output_dir,
    job_name="Test",
    role="classic",
    cpus_per_task=36,
    time="02:30:00",
    max_parallel=1,
):
    parts = args.output_dir.strip(os.sep).split(os.sep)
    job_name = "".join(parts[-3:])

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_dir = os.path.join(sweep_output_dir, "slurm_out", date)
    logs_dir = os.path.join(slurm_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Écriture des commandes
    # ---------------------------------------------------------
    if role == "incomplete":
        commands_file = os.path.join(
            sweep_output_dir,
            "commands_incomplete.txt",
        )
    else:
        commands_file = os.path.join(
            sweep_output_dir,
            "commands.txt",
        )

    with open(commands_file, "w") as f:
        for job in jobs:
            os.makedirs(job.output_dir, exist_ok=True)
            f.write(job.command_str + "\n")

    n_jobs = len(jobs)

    if n_jobs == 0:
        raise ValueError("No jobs to submit.")

    # ---------------------------------------------------------
    # 2. Job array
    # ---------------------------------------------------------
    array_str = f"0-{n_jobs - 1}"

    if max_parallel is not None:
        array_str += f"%{max_parallel}"

    # ---------------------------------------------------------
    # 3. Nom du script
    # ---------------------------------------------------------
    if role == "incomplete":
        slurm_script = os.path.join(
            sweep_output_dir,
            "submit_slurm_incomplete.sh",
        )
    else:
        slurm_script = os.path.join(
            sweep_output_dir,
            "submit_slurm.sh",
        )

    # ---------------------------------------------------------
    # 4. Paths
    # ---------------------------------------------------------
    domainbed_dir = "/lustre/work/pdl16831/upr55xw/these/domainbed/DomainBed"

    spack_root = (
        "/lustre/work/pdl16831/shared/ldz/"
        "spack-local-v1.1.1-0"
    )

    ffmpeg_prefix = (
        "/lustre/work/pdl16831/shared/ldz/"
        "spack-local-v1.1.1-0/spack/opt/spack/"
        "linux-neoverse_v2/"
        "ffmpeg-7.1-452ilf4o4mqo6voprmeu235lmd52kpnh"
    )

    venv_dir = f"{domainbed_dir}/.venv"

    # ---------------------------------------------------------
    # 5. Script Slurm
    # ---------------------------------------------------------
    script_content = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --chdir={domainbed_dir}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/%x_%A_%a.out
#SBATCH --error={logs_dir}/%x_%A_%a.err
#SBATCH --array={array_str}


echo "========================================"
echo "Running on: ${{SLURM_JOB_NODELIST}}"
echo "Job ID:     ${{SLURM_JOB_ID}}"
echo "Array ID:   ${{SLURM_ARRAY_JOB_ID}}"
echo "Task ID:   ${{SLURM_ARRAY_TASK_ID}}"
echo "GPU(s):    ${{SLURM_GPUS_ON_NODE}}"
echo "========================================"


# ---------------------------------------------------------
# ohmyjob / modules
# ---------------------------------------------------------
module use /lustre/work/pdl16831/shared/modulefiles/
module load ohmyjob


# ---------------------------------------------------------
# Python environment
# ---------------------------------------------------------
source "{venv_dir}/bin/activate"


# ---------------------------------------------------------
# Spack
# ---------------------------------------------------------
source "{spack_root}/spack/share/spack/setup-env.sh"


# ---------------------------------------------------------
# TorchCodec runtime libraries
# ---------------------------------------------------------

# PyTorch shared libraries
export TORCH_LIB="{venv_dir}/lib/python3.12/site-packages/torch/lib"

# CUDA libraries installed by pip/uv
export CUDA_PYTHON_LIB="{venv_dir}/lib/python3.12/site-packages/nvidia/cu13/lib"

# Known-good FFmpeg 7.1 installation
export FFMPEG_PREFIX="{ffmpeg_prefix}"
export FFMPEG_LIB="$FFMPEG_PREFIX/lib"

# Make all required shared libraries visible
export LD_LIBRARY_PATH="$FFMPEG_LIB:$CUDA_PYTHON_LIB:$TORCH_LIB:$LD_LIBRARY_PATH"


# ---------------------------------------------------------
# Environment diagnostics
# ---------------------------------------------------------
echo ""
echo "========== ENVIRONMENT =========="
echo "HOST:              $(hostname)"
echo "PWD:               $(pwd)"
echo "Python:            $(which python)"
echo "Python version:    $(python --version)"
echo "FFmpeg:            $FFMPEG_PREFIX"
echo "Torch lib:         $TORCH_LIB"
echo "CUDA Python lib:   $CUDA_PYTHON_LIB"
echo "================================="
echo ""


# ---------------------------------------------------------
# Verify FFmpeg
# ---------------------------------------------------------
echo "========== FFMPEG =========="
"$FFMPEG_PREFIX/bin/ffmpeg" -version | head -n 3
echo ""


# ---------------------------------------------------------
# Verify Torch / CUDA
# ---------------------------------------------------------
echo "========== PYTORCH =========="
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
echo ""


# ---------------------------------------------------------
# Verify TorchCodec BEFORE starting the experiment
# ---------------------------------------------------------
echo "========== TORCHCODEC =========="

python -c "
import torchcodec
print('torchcodec:', torchcodec.__version__)
print('torchcodec import: OK')
"

echo ""


# ---------------------------------------------------------
# Verify the FFmpeg-7 TorchCodec binary
# ---------------------------------------------------------
TORCHCODEC_CORE="$VIRTUAL_ENV/lib/python3.12/site-packages/torchcodec/libtorchcodec_core7.so"

echo "TorchCodec core7:"
echo "$TORCHCODEC_CORE"

if [ ! -f "$TORCHCODEC_CORE" ]; then
    echo "ERROR: TorchCodec core7 library not found!"
    exit 1
fi

echo ""
echo "Checking shared libraries..."
MISSING=$(ldd "$TORCHCODEC_CORE" | grep "not found" || true)

if [ -n "$MISSING" ]; then
    echo ""
    echo "ERROR: Missing TorchCodec shared libraries:"
    echo "$MISSING"
    echo ""
    echo "Full ldd output:"
    ldd "$TORCHCODEC_CORE"
    exit 1
fi

echo "All TorchCodec shared libraries found."
echo ""


# ---------------------------------------------------------
# Get command for this array task
# ---------------------------------------------------------
COMMAND_FILE="{commands_file}"

CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMMAND_FILE")

echo "Executing:"
echo "$CMD"

echo ""
echo "PWD: $(pwd)"
echo "Python: $(which python)"
echo "uv: $(which uv)"
echo ""

set -ex

eval "srun ohmyjob $CMD"

STATUS=$?

echo "Exit status: $STATUS"

exit $STATUS
"""

    # ---------------------------------------------------------
    # 6. Écriture du script
    # ---------------------------------------------------------
    with open(slurm_script, "w") as f:
        f.write(script_content)

    # ---------------------------------------------------------
    # 7. Rendre exécutable
    # ---------------------------------------------------------
    st = os.stat(slurm_script)
    os.chmod(slurm_script, st.st_mode | stat.S_IEXEC)

    # ---------------------------------------------------------
    # 8. Résumé
    # ---------------------------------------------------------
    print(f"\\n✔ Sweep prêt dans : {sweep_output_dir}")
    print(f"✔ Nombre de jobs : {n_jobs}")
    print(f"✔ Nombre maximum de jobs simultanés : {max_parallel}")
    print(f"✔ Script : {slurm_script}")

    print("\\nPour lancer :")
    print(f"cd {sweep_output_dir}")
    print("sbatch submit_slurm.sh")