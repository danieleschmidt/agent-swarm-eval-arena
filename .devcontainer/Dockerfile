ARG VARIANT="3.9-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        vim \
        htop \
        tree \
        jq \
        postgresql-client \
        redis-tools \
        graphviz \
        libjpeg-dev \
        libpng-dev \
        libfreetype6-dev \
        pkg-config \
        libhdf5-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
    && apt-get autoremove -y && apt-get clean -y

# Install Python development tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Ray with all extras for distributed computing
RUN python -m pip install "ray[default,serve,tune,rllib]==2.8.0"

# Install Jupyter extensions and kernels
RUN python -m pip install \
    jupyterlab \
    jupyterlab-git \
    jupyterlab-lsp \
    jupyter-ai \
    nbconvert \
    papermill \
    ipywidgets

# Install additional development tools
RUN python -m pip install \
    pre-commit \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pytest-xdist \
    coverage \
    bandit \
    safety

# Install performance and profiling tools
RUN python -m pip install \
    memory-profiler \
    py-spy \
    scalene \
    line-profiler \
    pympler

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set up git safe directory
RUN git config --global --add safe.directory /workspace

# Configure shell
RUN echo 'export PATH="/workspace/scripts:$PATH"' >> /home/vscode/.bashrc
RUN echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /home/vscode/.bashrc

# Switch to vscode user
USER vscode

# Configure git for the vscode user
RUN git config --global init.defaultBranch main
RUN git config --global pull.rebase false

# Install Oh My Zsh plugins
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure Zsh
RUN sed -i 's/plugins=(git)/plugins=(git python docker kubectl zsh-autosuggestions zsh-syntax-highlighting)/g' ~/.zshrc
RUN echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.zshrc
RUN echo 'export PATH="/workspace/scripts:$PATH"' >> ~/.zshrc
RUN echo 'alias ll="ls -la"' >> ~/.zshrc
RUN echo 'alias la="ls -la"' >> ~/.zshrc
RUN echo 'alias pytest="python -m pytest"' >> ~/.zshrc
RUN echo 'alias black="python -m black"' >> ~/.zshrc
RUN echo 'alias isort="python -m isort"' >> ~/.zshrc
RUN echo 'alias mypy="python -m mypy"' >> ~/.zshrc