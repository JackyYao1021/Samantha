FROM openeuler/openeuler:24.03-lts-sp3 AS buildbase

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

FROM buildbase AS runtime
WORKDIR /work
COPY work/ /work/
RUN printf '%s\n' 'source /work/samantha.sh' >> /root/.bashrc

CMD ["/bin/bash"]

# Test services use image code, a writable results volume, and an unprivileged UID.
FROM runtime AS task-tests
RUN command -v tar && command -v gzip \
    && mkdir -p /test-results \
    && chown 10001:10001 /test-results
ENV HOME=/tmp/samantha-test-home PYTHONUTF8=1 SAMANTHA_TEST_RESULTS=/test-results
USER 10001:10001
WORKDIR /test-results
ENTRYPOINT ["bash", "/work/tests/real_tasks/container.sh"]
CMD ["shell"]

# Evaluation does not inherit runtime's COPY work, its datasets, or test oracles.
# Input arrives over stdin and generated commands execute only in this container.
FROM buildbase AS evalsandbox
COPY work/src/ /opt/samantha/src/
COPY work/evals/sandbox_worker.py /opt/samantha/sandbox_worker.py
RUN mkdir -p /fixture && chown 10001:10001 /fixture
ENV HOME=/fixture TMPDIR=/fixture PYTHONUTF8=1 SAMANTHA_EVAL_CONTAINER=1
USER 10001:10001
WORKDIR /fixture
ENTRYPOINT ["python3", "/opt/samantha/sandbox_worker.py"]

# Standalone docker build keeps the regular application as its default target.
FROM runtime AS default-runtime
