FROM openeuler/openeuler:24.03-lts-sp3

SHELL ["/bin/bash", "-c"]
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN dnf install -y python3 python3-pip bash curl findutils coreutils grep sed \
    && dnf clean all
RUN python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

COPY work/requirements.txt /tmp/samantha-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/samantha-requirements.txt

WORKDIR /work
COPY work/ /work/
RUN printf '%s\n' 'source /work/samantha.sh' >> /root/.bashrc

CMD ["/bin/bash"]
