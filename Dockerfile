# eeg-seizures: SzCORE-format seizure detection on scalp EEG (CPU only, no network at run time).
FROM --platform=linux/amd64 python:3.12.16-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4

WORKDIR /processor

# Pinned Python dependencies, binary wheels only, no dependency resolution at build time
# (requirements.txt lists every package). Test folders are dropped; Matplotlib's font cache is
# built into a read-only path and copied to $TMPDIR at start-up (Lambda-safe).
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --root-user-action=ignore -r /processor/requirements.txt \
    && pip check \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -depth -type d \( -name tests -o -name testing \) -path "*/mne/*" -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/pandas/tests /usr/local/lib/python3.12/site-packages/scipy/*/tests \
              /usr/local/lib/python3.12/site-packages/numpy/*/tests /usr/local/lib/python3.12/site-packages/matplotlib/tests \
              /usr/local/lib/python3.12/site-packages/onnxruntime/test* \
    && mkdir -p /opt/eeg-seizures/matplotlib \
    && MPLCONFIGDIR=/opt/eeg-seizures/matplotlib python3 -c "import matplotlib.font_manager" \
    && chmod -R a+rX /opt/eeg-seizures \
    && useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPL_FONT_CACHE=/opt/eeg-seizures/matplotlib \
    MNE_DONTWRITE_HOME=true \
    MNE_USE_CUDA=false \
    HOME=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Runs as uid 1000 on ECS and as an arbitrary user on Lambda; nothing here needs root or a
# writable image path (scratch goes to $TMPDIR, Matplotlib's cache to $TMPDIR/mpl).
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
