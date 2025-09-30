FROM openeuler/openeuler:latest

SHELL ["/bin/bash", "-c"]
RUN /usr/bin/dnf update -y
RUN /usr/bin/dnf install -y python3 nodejs npm curl go sqlite htop
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN . "$HOME/.cargo/env"
RUN /usr/bin/dnf group install -y "Development Tools"

WORKDIR /work

ENTRYPOINT /bin/sh