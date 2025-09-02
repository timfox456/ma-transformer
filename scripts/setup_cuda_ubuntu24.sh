#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Setup NVIDIA driver and CUDA toolkit on Ubuntu 24.04 (noble) for T4
# - Installs latest recommended NVIDIA driver via ubuntu-drivers
# - Installs CUDA Toolkit 12.1 from Ubuntu repositories
# - Appends environment exports to ~/.bashrc
# Requires: sudo

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run with sudo: sudo bash $0"
  exit 1
fi

echo "== Detecting Ubuntu version =="
. /etc/os-release
echo "${NAME} ${VERSION} (${VERSION_CODENAME})"
if [ "${VERSION_ID}" != "24.04" ]; then
  echo "Warning: this script targets Ubuntu 24.04; continuing anyway..."
fi

echo "== Installing NVIDIA driver =="
apt-get update
apt-get install -y ubuntu-drivers-common
ubuntu-drivers autoinstall || true
echo "Driver installation initiated. A reboot will be required."

echo "== Installing CUDA Toolkit 12.1 =="
apt-get update
apt-get install -y cuda-toolkit-12-1

echo "== Configuring environment for CUDA =="
USER_HOME=$(getent passwd $(logname 2>/dev/null || echo $SUDO_USER) | cut -d: -f6)
if [ -z "${USER_HOME}" ]; then USER_HOME=/home/$SUDO_USER; fi
if [ -z "${USER_HOME}" ] || [ ! -d "${USER_HOME}" ]; then USER_HOME=/root; fi

CUDA_HOME=/usr/local/cuda-12.1
PROFILE_FILE="${USER_HOME}/.bashrc"
if ! grep -q "CUDA_HOME" "$PROFILE_FILE" 2>/dev/null; then
  {
    echo "# CUDA 12.1 environment"
    echo "export CUDA_HOME=${CUDA_HOME}"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
  } >> "$PROFILE_FILE"
  chown $(logname 2>/dev/null || echo $SUDO_USER): "$PROFILE_FILE" || true
fi

echo
echo "== Done =="
echo "- Reboot the system to load the NVIDIA driver: sudo reboot"
echo "- After reboot, verify: nvidia-smi and nvcc --version"
echo "- Then reinstall PyTorch CUDA wheels matching CUDA 12.1 and rebuild the project"
