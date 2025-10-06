###############################################################################
# Stage 1: STEP1
#   - Start from NVIDIA’s PyTorch image (CUDA 12.8 + cuDNN + PyTorch preinstalled)
#   - Install all system-level libraries (Git, ffmpeg, OpenCV, etc.)
###############################################################################
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS step1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake build-essential \
      libavcodec-dev libavformat-dev libavutil-dev \
      libfmt-dev qtbase5-dev libboost-all-dev libopencv-dev \
      libfdk-aac-dev nasm libass-dev libmp3lame-dev \
      libopus-dev libvorbis-dev libx264-dev libx265-dev \
      libxcb-xinput0 yasm libtool libc6 libc6-dev unzip wget \
      libnuma1 libnuma-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

###############################################################################
# Stage 2: STEP2
#   - Install Miniforge under /opt/conda
#   - Set up environment so "conda" works out of the box
###############################################################################
FROM step1 AS step2

# Download & install Miniforge3 (conda-forge’s lightweight installer)
RUN wget --quiet \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Put conda on the PATH and disable auto-activation of base
ENV PATH="/opt/conda/bin:$PATH" \
    CONDA_AUTO_ACTIVATE_BASE="false"

# Use bash login shell so "conda" and "activate" work naturally
SHELL ["/bin/bash", "-lc"]

###############################################################################
# Stage 3: STEP3
#   - Create the "beesbook" conda env with Python 3.12
#   - Install pip, core Python libraries, and TensorFlow
###############################################################################
FROM step2 AS step3

RUN conda create -n beesbook python=3.12 -y && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate beesbook && \
    conda install -n beesbook pip -y && \
    pip install --upgrade pip setuptools wheel build setuptools-scm && \
    conda install -n beesbook -c conda-forge \
       jupyterlab matplotlib scipy seaborn numpy ffmpeg dill tqdm chardet \
       lapack blas sqlite && \
    python -m pip install cairocffi && \
    python -m pip install tensorflow[and-cuda] && \
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio && \    
    conda clean --all -y

ENV CONDA_DEFAULT_ENV=beesbook \
    PATH="/opt/conda/envs/beesbook/bin:$PATH"

###############################################################################
# Stage 4: STEP4
#   - On top of step3, install all BeesBook Python packages from GitHub
###############################################################################
FROM step3 AS step4

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate beesbook && \
    apt update && \
    apt install -y python3-setuptools python3-distutils-extra

ENV SETUPTOOLS_USE_DISTUTILS=local

RUN pip uninstall -y bb_binary bb_pipeline bb_tracking bb_behavior bb_utils 2>/dev/null || true && \
    pip install --upgrade \
      "git+https://github.com/BioroboticsLab/bb_binary#egg=bb_binary" \
      "git+https://github.com/BioroboticsLab/bb_pipeline#egg=bb_pipeline" \
      "git+https://github.com/BioroboticsLab/bb_tracking#egg=bb_tracking" \
      "git+https://github.com/BioroboticsLab/bb_behavior#egg=bb_behavior" \
      "git+https://github.com/BioroboticsLab/bb_utils#egg=bb_utils"

###############################################################################
# Stage 5: Download & patch model files into the conda env
###############################################################################
FROM step4 AS step5

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate beesbook && \
    \
    # 1) Locate pipeline package dir, suppressing noisy TF logs
    BB_PIPELINE_DIR=$(python - <<<'import sys;from io import StringIO;sys.stdout=StringIO();import pipeline;sys.stdout=sys.__stdout__;print(pipeline.__path__[0])') && \
    \
    # 2) Prepare model dir
    MODEL_DIR="$CONDA_PREFIX/pipeline_models" && \
    mkdir -p "$MODEL_DIR" && \
    \
    # 3) Download all model files
    echo "Downloading model files…" && \
    for f in \
      decoder_2019_keras3.h5 \
      localizer_2019_keras3.h5 \
      localizer_2019_attributes.json \
      detection_model_4.json \
      tracklet_model_8.json; do \
      case "$f" in \
        decoder_2019_keras3.h5)                  subdir="decoder"  ;; \
        localizer_2019_keras3.h5|localizer_2019_attributes.json) subdir="saliency" ;; \
        detection_model_4.json|tracklet_model_8.json)           subdir="tracking" ;; \
      esac && \
      wget -q -O "$MODEL_DIR/$f" \
        "https://raw.githubusercontent.com/BioroboticsLab/bb_pipeline_models/master/models/$subdir/$f"; \
    done && \
    \
    # 4) Patch config.ini in-place
    echo "Patching config.ini…" && \
    CONFIG_FILE="$BB_PIPELINE_DIR/config.ini" && \
    sed -i \
      -e "s|^model_path=decoder_.*\.h5\$|model_path=$MODEL_DIR/decoder_2019_keras3.h5|" \
      -e "s|^model_path=localizer_.*\.h5\$|model_path=$MODEL_DIR/localizer_2019_keras3.h5|" \
      -e "s|^attributes_path=.*\.json\$|attributes_path=$MODEL_DIR/localizer_2019_attributes.json|" \
      "$CONFIG_FILE"

###############################################################################
# Stage 6: STEP6
#   - Install Cairo and FreeType (as per updated install script)
#   - Install parquet support
###############################################################################
FROM step5 AS step6

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate beesbook && \
    conda install -n beesbook -c conda-forge cairo freetype -y && \
    conda install -n beesbook -c conda-forge pyarrow fastparquet -y && \
    conda clean --all -y

###############################################################################
# Stage 7: STEP7 (FINAL)
#   - Remove any remaining caches to slim the image
#   - Set working directory & default command
###############################################################################
FROM step6 AS step7

RUN rm -rf ~/.cache ~/.conda /opt/conda/pkgs

WORKDIR /workspace
CMD ["/bin/bash"]