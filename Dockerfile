FROM openeuler/openeuler:24.03-lts-sp3

SHELL ["/bin/bash", "-c"]
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ARG OPENEULER_REPO_URL=https://repo.openeuler.org
RUN sed -i -e '/^metalink=/d' \
        -e "s|^baseurl=https://repo.openeuler.org/|baseurl=${OPENEULER_REPO_URL%/}/|" \
        /etc/yum.repos.d/openEuler.repo \
    && dnf --setopt=timeout=30 --setopt=retries=2 \
        --disablerepo=EPOL,debuginfo,source,update-source install -y \
        python3 python3-pip bash curl findutils coreutils grep sed \
    && dnf clean all
RUN python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

ARG CONTENT_HYBRID=false
COPY work/requirements.txt work/requirements-content-hybrid.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && if [ "$CONTENT_HYBRID" = "true" ]; then \
        python3 -m pip install --no-cache-dir -r /tmp/requirements-content-hybrid.txt; \
    fi

WORKDIR /work
COPY work/ /work/
RUN printf '%s\n' 'source /work/samantha.sh' >> /root/.bashrc

CMD ["/bin/bash"]
