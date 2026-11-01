# A throwaway lab container to reconstruct the xz timing symptom on Linux, where perf/strace are real.
# It contains NO xz backdoor and downloads no xz tarballs. It builds a benign CPU-burning shim
# (slow_recon.c) and an sshd to time. linux/amd64 (the ThinkPad engine).
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server gcc libc6-dev \
        linux-perf strace time hyperfine \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# an sshd that will always reject (no keys, no passwords) — we only ever time the FAILED login path,
# so nothing needs to authenticate and the container holds no secrets.
RUN mkdir -p /run/sshd \
    && ssh-keygen -A \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config

WORKDIR /lab
COPY slow_recon.c measure.sh challenge.sh demo.sh /lab/
RUN cc -O2 -fPIC -shared slow_recon.c -o /lab/libslow-recon.so && chmod +x /lab/*.sh

CMD ["bash"]
