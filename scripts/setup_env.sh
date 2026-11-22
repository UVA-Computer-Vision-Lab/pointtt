#!/usr/bin/env bash
# =============================================================================
# PoinTTT -- environment build (conda + pip): python 3.11, torch, spconv, torch-scatter,
# flash-attn (prebuilt wheel) and pointops (compiled from libs/pointops).
#
# Usage (from any directory):
#   bash scripts/setup_env.sh                                                # ./env, torch 2.5.0+cu124 (A100 / H100 / ...)
#   ENV_PREFIX=$PWD/env-torch2.7 TORCH_VER=2.7.1 bash scripts/setup_env.sh  # Blackwell GPUs (B200, sm_100): torch 2.7.1+cu128
#   ENV_PREFIX=/data/envs/pointtt bash scripts/setup_env.sh                  # env somewhere else
#   USE_CONDA_GCC=1 bash scripts/setup_env.sh                                # host without a gcc 9-13
# The evaluation scripts use ./env/bin/python if it exists; otherwise set PYTHON=<prefix>/bin/python.
# Re-running resumes after the last finished stage (stamp files in $ENV_PREFIX/.setup_stamps).
# One prefix holds one torch version: build another TORCH_VER into its own ENV_PREFIX.
# On a node without GPU the build still finishes; the closing smoke test then runs with --no-forward and
# the script prints the command for the full test, to be run once on a GPU node.
#
# Knobs (env vars):
#   ENV_PREFIX            conda env location                      (default: <repo>/env)
#   TORCH_VER             2.5.0 | 2.7.1                            (default: 2.5.0; 2.7.1 = +cu128 for Blackwell)
#   CONDA_EXE_OVERRIDE    conda front-end to use                   (default: the first of mamba, conda, micromamba
#                         on PATH, then $MAMBA_EXE)
#   USE_CONDA_GCC=1       install gcc/g++ 11 from conda-forge into the env and compile pointops with it
#   TORCH_CUDA_ARCH_LIST  GPU architectures pointops is compiled for (default "8.0 8.6 9.0" = A100, A40/A6000,
#                         H100; with TORCH_VER=2.7.1 "8.0 8.6 9.0 10.0", which adds B200)
#   USE_SYSTEM_CUDA=1     use an nvcc already on PATH / $CUDA_HOME instead of installing the conda toolkit
#   FA_VER                flash-attn release to take the prebuilt wheel from (default 2.8.3; with torch 2.5.0 and
#                         glibc < 2.32 the script falls back to 2.7.4.post1)
#   SKIP_SMOKE=1          do not run scripts/smoke_test_env.py at the end
#   KEEP_CACHE=1          keep $ENV_PREFIX/.cache (pip cache, downloaded wheels) instead of deleting it at the end
# =============================================================================
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${ENV_PREFIX:-$REPO/env}
TORCH_VER=${TORCH_VER:-2.5.0}
CONSTRAINTS=${CONSTRAINTS:-$REPO/requirements/constraints.txt}
SMOKE=$REPO/scripts/smoke_test_env.py
POINTOPS_SRC=$REPO/libs/pointops

# torch 2.5 +cu124 wheels use the old C++ ABI (flash-attn wheel "cxx11abiFALSE"), torch 2.7 the new one.
# Blackwell (B200 = sm_100): the +cu124 torch wheels carry GPU code up to sm_90 only and the CUDA 12.4 nvcc cannot
# target sm_100, so TORCH_VER=2.7.1 installs torch 2.7.1+cu128 and compiles pointops with the CUDA 12.8 nvcc.
# spconv has no cu128 wheel: spconv-cu124 2.3.8 is used with both torch versions.
CU=cu124; SPCONV_CU=cu124; CUDA_LABEL=cuda-12.4.1; FA_ABI=FALSE; ARCH_DEFAULT="8.0 8.6 9.0"; DROP_PINS=""
case "$TORCH_VER" in
  2.5.0) TV_VER=0.20.0; PT_TAG=pt25; FA_TORCH=2.5 ;;
  2.7.1) TV_VER=0.22.1; PT_TAG=pt27; FA_TORCH=2.7
         CU=cu128; CUDA_LABEL=cuda-12.8.1; FA_ABI=TRUE
         ARCH_DEFAULT="8.0 8.6 9.0 10.0"
         DROP_PINS="sympy" ;;                                # torch 2.7 requires sympy >= 1.13.3
  *) echo "unsupported TORCH_VER=$TORCH_VER (use 2.5.0 or 2.7.1)"; exit 2 ;;
esac
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$ARCH_DEFAULT}
FA_VER=${FA_VER:-2.8.3}

# ---- isolation -------------------------------------------------------------------------------------
# keep ~/.local/lib/python3.11/site-packages out of the env
export PYTHONNOUSERSITE=1
# pip cache and temp dir on the env's filesystem: installing a wheel across filesystems can fail with
# "[Errno 18] Invalid cross-device link"
mkdir -p "$(dirname "$ENV_PREFIX")"
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$ENV_PREFIX/.cache/pip}
BUILD_TMP=${BUILD_TMP:-$ENV_PREFIX/.cache/tmp}          # becomes TMPDIR once the env exists
export PIP_DISABLE_PIP_VERSION_CHECK=1
WHEEL_DIR=$ENV_PREFIX/.cache/wheels
STAMPS=$ENV_PREFIX/.setup_stamps

# micromamba is usually a shell function plus $MAMBA_EXE, not a binary on PATH
CONDA=${CONDA_EXE_OVERRIDE:-$(command -v mamba || command -v conda || command -v micromamba || echo "${MAMBA_EXE:-}")}
[ -n "$CONDA" ] || { echo "need conda, mamba or micromamba on PATH (or CONDA_EXE_OVERRIDE=/path/to/one of them)"; exit 2; }

log()   { echo; echo "########## [$(date '+%F %T')] STAGE: $* ##########"; }
stage() { [ -f "$STAMPS/$1" ] && { echo "[skip] $1 (already done)"; return 1; } || return 0; }
done_() { mkdir -p "$STAMPS"; touch "$STAMPS/$1"; }

# ---- compiler preflight (before anything is installed) --------------------------------------------------------
# libs/pointops (stage 7) is compiled from source: the torch headers need gcc >= 9, the CUDA 12.x nvcc refuses
# gcc > 13. C++ -> $CXX or `c++`; host compiler of nvcc -> $CC or `gcc`.
# shellcheck disable=SC2086
cc_version()  { $1 -dumpfullversion -dumpversion 2>/dev/null || $1 -dumpversion 2>/dev/null || true; }
compiler_ok() {   # compiler_ok <driver>: it exists and is version 9-13; otherwise COMPILER_MSG says what is wrong
  local v
  command -v "${1%% *}" >/dev/null 2>&1 || { COMPILER_MSG="no '$1' found"; return 1; }
  v=$(cc_version "$1")
  case "${v%%.*}" in
    9|10|11|12|13) return 0 ;;
    *) COMPILER_MSG="'$1' is version ${v:-unknown}"; return 1 ;;
  esac
}
NEED_COMPILER=0
[ -f "$STAMPS/7_pointops" ] || NEED_COMPILER=1
COMPILER_MSG=
if [ "$NEED_COMPILER" = 1 ]; then
  if [ "${USE_CONDA_GCC:-0}" = "1" ]; then
    echo "[compiler] USE_CONDA_GCC=1: gcc/g++ 11 from conda-forge goes into the env and is used for pointops"
  elif compiler_ok "${CXX:-c++}" && compiler_ok "${CC:-gcc}" && { [ -n "${CXX:-}" ] || command -v g++ >/dev/null 2>&1 ||
         { COMPILER_MSG="no 'g++' found (it links the extension modules)"; false; }; }; then
    echo "[compiler] ${CXX:-c++} $(cc_version "${CXX:-c++}"), ${CC:-gcc} $(cc_version "${CC:-gcc}") (need 9-13): ok"
  else
    cat <<EOF
ERROR: no usable compiler: $COMPILER_MSG.
  libs/pointops is compiled from source and needs gcc/g++ 9-13. Either
    USE_CONDA_GCC=1 bash scripts/setup_env.sh    # installs gcc/g++ 11 from conda-forge into the env and uses it
  or put a gcc 9-13 on PATH (e.g. \`module load gcc/11\`), or point CC and CXX at one (CC=gcc-12 CXX=g++-12).
  Nothing was installed by this run.
EOF
    exit 3
  fi
fi

# ---- guard: one prefix holds one torch version ---------------------------------------------------------------
torch_guard() {   # torch_guard optional|required   (optional: a prefix without torch passes = fresh build)
  local have
  have=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null | tail -1) || true
  if [ "$have" = "${TORCH_VER}+${CU}" ] || { [ -z "$have" ] && [ "$1" = optional ]; }; then return 0; fi
  if [ -z "$have" ]; then
    $PY -c "import torch" || true      # show why
    echo "ERROR: $PY cannot import torch although every stage is finished: delete $ENV_PREFIX and run this command again."
    exit 2
  fi
  cat <<EOF

ERROR: $ENV_PREFIX holds torch $have, but this run asks for TORCH_VER=$TORCH_VER (torch ${TORCH_VER}+${CU}).
  Build the other version into a prefix of its own, e.g.
    ENV_PREFIX=$REPO/env-torch${TORCH_VER%.*} TORCH_VER=$TORCH_VER bash scripts/setup_env.sh
  or delete $ENV_PREFIX and run this command again.
EOF
  exit 2
}

# ---- 1/7 conda env -----------------------------------------------------------------------------------------
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  log "1/7 conda env (python 3.11) at $ENV_PREFIX"
  # conda-forge + --override-channels: no Anaconda terms-of-service prompt on a fresh machine
  "$CONDA" create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
      python=3.11 pip "setuptools=80.10.2" wheel ninja
fi
PY=$ENV_PREFIX/bin/python
PIP="$PY -m pip"
torch_guard optional
mkdir -p "$PIP_CACHE_DIR" "$BUILD_TMP" "$WHEEL_DIR" "$STAMPS"
if [ -n "$DROP_PINS" ]; then   # same constraints minus the pins this torch version cannot live with
  grep -v -E "^(${DROP_PINS// /|})==" "$CONSTRAINTS" > "$STAMPS/constraints.txt"
  CONSTRAINTS=$STAMPS/constraints.txt
fi
export TMPDIR=$BUILD_TMP
export PATH=$ENV_PREFIX/bin:$PATH
done_ 1_env

# ---- optional: gcc/g++ 11 from conda-forge (USE_CONDA_GCC=1) -------------------------------------------------
if [ "${USE_CONDA_GCC:-0}" = "1" ] && [ "$NEED_COMPILER" = 1 ]; then
  CC=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc
  CXX=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++
  if [ ! -x "$CC" ] || [ ! -x "$CXX" ]; then
    log "conda compilers: gxx_linux-64=11 from conda-forge (USE_CONDA_GCC=1)"
    "$CONDA" install -y -p "$ENV_PREFIX" --override-channels -c conda-forge "gxx_linux-64=11"
  fi
  export CC CXX
  if ! { compiler_ok "$CXX" && compiler_ok "$CC"; }; then echo "ERROR: conda compiler not usable: $COMPILER_MSG"; exit 3; fi
  echo "[compiler] CC=$CC CXX=$CXX ($(cc_version "$CXX"))"
fi

# ---- 2/7 torch ------------------------------------------------------------------------------------------------
if stage 2_torch; then
  log "2/7 torch ${TORCH_VER}+${CU} / torchvision ${TV_VER}"
  $PIP install "torch==${TORCH_VER}+${CU}" "torchvision==${TV_VER}+${CU}" \
      --index-url "https://download.pytorch.org/whl/${CU}" --extra-index-url https://pypi.org/simple \
      -c "$CONSTRAINTS"
  $PY -c "import torch; assert torch.__version__=='${TORCH_VER}+${CU}', torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  done_ 2_torch
fi

# ---- 3/7 torch-scatter + spconv ----------------------------------------------------------------------------------
if stage 3_scatter_spconv; then
  log "3/7 torch-scatter (${PT_TAG}${CU}) + spconv-${SPCONV_CU} 2.3.8"
  $PIP install "torch-scatter==2.1.2+${PT_TAG}${CU}" -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CU}.html" -c "$CONSTRAINTS"
  $PIP install "spconv-${SPCONV_CU}==2.3.8" -c "$CONSTRAINTS"   # pulls cumm-cu124 0.7.11, pccm, ccimport, pybind11, fire
  done_ 3_scatter_spconv
fi

# ---- 4/7 python packages ----------------------------------------------------------------------------------------
if stage 4_pydeps; then
  log "4/7 python packages (versions pinned by $CONSTRAINTS)"
  $PIP install -c "$CONSTRAINTS" \
      numpy scipy h5py plyfile pyyaml packaging addict yapf termcolor timm einops tensorboardx tqdm
  done_ 4_pydeps
fi

# ---- 5/7 CUDA toolkit (nvcc + headers) for building pointops ------------------------------------------------------
if stage 5_cuda_toolkit; then
  if [ "${USE_SYSTEM_CUDA:-0}" = "1" ]; then
    log "5/7 cuda-toolkit: using system nvcc ($(command -v nvcc || echo "${CUDA_HOME:-?}/bin/nvcc"))"
  else
    log "5/7 cuda-toolkit ${CUDA_LABEL#cuda-} into the env (nvcc for source builds)"
    # torch's ATen/cuda/CUDAContext.h needs cusparse.h / cublas_v2.h / cusolverDn.h, so nvcc alone is not enough
    # shellcheck disable=SC2086
    "$CONDA" install -y -p "$ENV_PREFIX" --override-channels -c "nvidia/label/$CUDA_LABEL" -c conda-forge \
        ${CUDA_CONDA_PKGS:-cuda-nvcc cuda-cudart-dev libcublas-dev libcusparse-dev libcusolver-dev}
  fi
  done_ 5_cuda_toolkit
fi
if [ "${USE_SYSTEM_CUDA:-0}" = "1" ]; then
  : "${CUDA_HOME:=$(dirname "$(dirname "$(command -v nvcc)")")}"
else
  CUDA_HOME=$ENV_PREFIX
  export CPATH=$ENV_PREFIX/include${CPATH:+:$CPATH}
  export LIBRARY_PATH=$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
  # the conda packages of CUDA 12.8 keep headers and libraries in targets/x86_64-linux
  if [ -d "$ENV_PREFIX/targets/x86_64-linux/include" ]; then
    export CPATH=$ENV_PREFIX/targets/x86_64-linux/include:$CPATH
    export LIBRARY_PATH=$ENV_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
  fi
fi
export CUDA_HOME

# ---- 6/7 flash-attn (prebuilt wheel) ---------------------------------------------------------------------------------
if stage 6_flash_attn; then
  GLIBC=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}'); GLIBC=${GLIBC:-0}
  if [ "$FA_VER" = "2.8.3" ] && [ "$(printf '%s\n' 2.32 "$GLIBC" | sort -V | head -1)" != "2.32" ]; then
    if [ "$TORCH_VER" = "2.5.0" ]; then
      echo "[warn] glibc $GLIBC < 2.32: the v2.8.3 wheel will not import here -> using v2.7.4.post1 (glibc >= 2.14)"
      FA_VER=2.7.4.post1
    else
      echo "[warn] glibc $GLIBC < 2.32: the flash-attn 2.8.3 wheel needs glibc >= 2.32 and may not import here"
    fi
  fi
  log "6/7 flash-attn ${FA_VER} (prebuilt wheel)"
  FA_WHL="flash_attn-${FA_VER}+cu12torch${FA_TORCH}cxx11abi${FA_ABI}-cp311-cp311-linux_x86_64.whl"
  FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${FA_WHL//+/%2B}"
  curl -fL --retry 3 -o "$WHEEL_DIR/$FA_WHL" "$FA_URL"
  $PIP install --no-deps "$WHEEL_DIR/$FA_WHL"        # runtime deps torch + einops are already installed
  # fail fast on an ABI mismatch (the model files swallow this ImportError and only assert when a model is built)
  $PY -c "import torch, flash_attn; from flash_attn import flash_attn_varlen_qkvpacked_func; print('flash_attn', flash_attn.__version__, 'imports ok')"
  done_ 6_flash_attn
fi

# ---- 7/7 pointops ------------------------------------------------------------------------------------------------------
if stage 7_pointops; then
  log "7/7 pointops from libs/pointops (TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\")"
  "$CUDA_HOME/bin/nvcc" --version | tail -2
  # --no-build-isolation: setup.py imports torch at module level
  BUILD=$(mktemp -d "$TMPDIR/pointops.XXXXXX"); cp -r "$POINTOPS_SRC/." "$BUILD/"; rm -rf "$BUILD/build" "$BUILD"/*.egg-info
  ( cd "$BUILD" && $PIP install --no-deps --no-build-isolation . )
  rm -rf "$BUILD"
  done_ 7_pointops
fi

# ---- record + smoke test -------------------------------------------------------------------------------------------------
torch_guard required
$PIP list --format=freeze > "$ENV_PREFIX/pip-freeze.txt"; echo "wrote $ENV_PREFIX/pip-freeze.txt ($(wc -l < "$ENV_PREFIX/pip-freeze.txt") packages)"

SMOKE_RC=0; NO_GPU=0
SMOKE_CMD="PYTHONNOUSERSITE=1 $PY $SMOKE --expect-torch $TORCH_VER"
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  SMOKE_ARGS="--expect-torch $TORCH_VER"
  if $PY -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    log "SMOKE TEST"
  else
    NO_GPU=1; SMOKE_ARGS="$SMOKE_ARGS --no-forward"
    log "SMOKE TEST (--no-forward: no usable GPU on this node)"
  fi
  # shellcheck disable=SC2086
  $PY "$SMOKE" $SMOKE_ARGS || SMOKE_RC=$?
fi

# pip cache / downloaded wheels / build temp live inside the env prefix; removed only after the smoke test
if [ "${KEEP_CACHE:-0}" != "1" ]; then rm -rf "$ENV_PREFIX/.cache"; fi

if [ "$SMOKE_RC" != 0 ]; then
  cat <<EOF

SMOKE TEST FAILED (exit code $SMOKE_RC): the [FAIL] lines above name the checks.
The build stages are finished, so re-running this script only repeats the test (pointops is rebuilt
after \`rm $STAMPS/7_pointops\`, e.g. for another TORCH_CUDA_ARCH_LIST). To repeat only the test:
  $SMOKE_CMD
EOF
  exit "$SMOKE_RC"
fi

cat <<EOF

Environment ready:  $PY
The evaluation scripts use ./env/bin/python if it exists; otherwise run them with PYTHON=$PY
EOF
if [ "$NO_GPU" = 1 ]; then
  cat <<EOF

NOTE: no usable GPU on this node, so only the part of the smoke test that needs none was run (--no-forward).
Run the full smoke test once on a GPU node:
  $SMOKE_CMD
EOF
fi
