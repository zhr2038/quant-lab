#!/usr/bin/env bash
set -euo pipefail

SYSTEMCTL_BIN="${QUANT_LAB_FORWARD_SYSTEMCTL:-systemctl}"
TEST_MODE="${QUANT_LAB_FORWARD_INSTALL_TEST_MODE:-0}"
DESTDIR="${QUANT_LAB_FORWARD_DESTDIR:-}"

if [[ "${TEST_MODE}" != "1" && "${EUID}" -ne 0 ]]; then
  echo "uninstall_forward_v221_systemd.sh must run as root" >&2
  exit 77
fi

if [[ "${TEST_MODE}" != "1" ]]; then
  ${SYSTEMCTL_BIN} disable --now quant-lab-forward-v221.timer 2>/dev/null || true
fi
rm -f "${DESTDIR}/etc/systemd/system/quant-lab-forward-v221.service"
rm -f "${DESTDIR}/etc/systemd/system/quant-lab-forward-v221.timer"
rm -f "${DESTDIR}/etc/quant-lab/forward-v221.env"
if [[ "${TEST_MODE}" != "1" ]]; then
  ${SYSTEMCTL_BIN} daemon-reload
  ${SYSTEMCTL_BIN} reset-failed quant-lab-forward-v221.service 2>/dev/null || true
fi

echo "Forward v2.2.1 units removed. Evidence and parameter locks were preserved."
