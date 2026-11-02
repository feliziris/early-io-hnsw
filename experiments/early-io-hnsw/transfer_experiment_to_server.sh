#!/usr/bin/env bash
# Run on the source machine. Copies index/sidecar/query/k10 model only.
# Generate the dummy payload on the server; do not transfer it.
set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: bash $0 USER HOST [REMOTE_REPO] [--skip-payload]" \
        "" \
        "REMOTE_REPO: absolute clone path; default is the remote login home/early-io-hnsw." \
        "Copies index, sidecar, query and k10 model only; payload is always excluded." \
        "--skip-payload: accepted for compatibility; no longer required." \
        "Environment: SSH_PORT=22, SOURCE_ROOT=local experiment root." \
        "Existing destination files with the same names will be overwritten."
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then usage >&2; exit 2; fi
server_user="$1"
server_host="$2"
shift 2
remote_repo=""
for argument in "$@"; do
    case "${argument}" in
        --skip-payload) : ;; # Payload is now always excluded.
        -*) fail "Unknown option: ${argument}" ;;
        *)
            [[ -z "${remote_repo}" ]] || fail "Only one REMOTE_REPO is allowed."
            remote_repo="${argument}"
            ;;
    esac
done
[[ "${server_user}" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || fail "Invalid SSH user."
[[ "${server_host}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "Use a domain name or IPv4 address."
ssh_port="${SSH_PORT:-22}"
[[ "${ssh_port}" =~ ^[0-9]{1,5}$ ]] || fail "Invalid SSH_PORT."
(( 10#${ssh_port} >= 1 && 10#${ssh_port} <= 65535 )) || fail "Invalid SSH_PORT."
for command in ssh scp stat; do
    command -v "${command}" >/dev/null || fail "Missing command: ${command}"
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="${SOURCE_ROOT:-${script_dir}/../../../..}"
source_root="$(cd -- "${source_root}" && pwd)"
files=(
    "data/index/bigann_base10M_hnsw_M32_efC128.faiss"
    "data/index/bigann_base10M_hnsw_M32_efC128.level0_avg_neighbor_distance.bin"
    "output/sift10M/bigann_query_test1000_seed42.fvecs"
    "model/sift10M/lstm_io/ef200/k10/model.onnx"
)

total_bytes=0
for file in "${files[@]}"; do
    [[ -r "${source_root}/${file}" && -f "${source_root}/${file}" ]] ||
        fail "Source file missing or unreadable: ${source_root}/${file}"
    bytes="$(stat -Lc %s -- "${source_root}/${file}")"
    total_bytes=$((total_bytes + bytes))
done

destination="${server_user}@${server_host}"
ssh_options=(-p "${ssh_port}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
scp_options=(-P "${ssh_port}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
if [[ -z "${remote_repo}" ]]; then
    # Resolve the real remote login home instead of assuming /home/USER.
    remote_home="$(ssh "${ssh_options[@]}" "${destination}" 'printf "%s\n" "$HOME"')"
    remote_repo="${remote_home}/early-io-hnsw"
fi
# Restrict remote paths to avoid shell interpretation in ssh / legacy scp.
[[ "${remote_repo}" =~ ^/[A-Za-z0-9_./-]+$ ]] ||
    fail "REMOTE_REPO must be absolute, with no spaces or shell metacharacters."
remote_repo="${remote_repo%/}"
[[ -n "${remote_repo}" ]] || fail "REMOTE_REPO cannot be /."

printf 'Source: %s\nDestination: %s:%s\nFiles: %d\nTotal: %s bytes\n' \
    "${source_root}" "${destination}" "${remote_repo}" "${#files[@]}" "${total_bytes}"
printf '%s\n' 'Existing files with the same names will be overwritten. SSH may ask for authentication.'
printf '%s\n' 'Payload is excluded; generate the dummy payload on the server.'
ssh "${ssh_options[@]}" "${destination}" \
    "test -f '${remote_repo}/CMakeLists.txt'" ||
    fail "Clone not found at ${remote_repo}; specify its absolute path as the third argument."
printf '\nRemote free space (allow at least %s bytes for a fresh copy):\n' "${total_bytes}"
ssh "${ssh_options[@]}" "${destination}" "df -h -- '${remote_repo}'"

for file in "${files[@]}"; do
    remote_dir="${remote_repo}/${file%/*}"
    ssh "${ssh_options[@]}" "${destination}" "mkdir -p -- '${remote_dir}'"
    printf '\nTransferring: %s\n' "${file}"
    scp "${scp_options[@]}" "${source_root}/${file}" "${destination}:${remote_dir}/"
    # Check size after each successful scp. This is not a checksum test.
    local_bytes="$(stat -Lc %s -- "${source_root}/${file}")"
    remote_bytes="$(ssh "${ssh_options[@]}" "${destination}" \
        "stat -Lc %s -- '${remote_repo}/${file}'")"
    [[ "${remote_bytes}" == "${local_bytes}" ]] || fail "Size mismatch: ${file}"
done
printf '\nTransfer complete: %d files copied; file sizes verified.\n' "${#files[@]}"
printf '%s\n' 'Source files were not removed. scp does not resume partially transferred files.'
