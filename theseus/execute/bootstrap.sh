#!/usr/bin/env bash

set -euo pipefail
umask 077

BOOTSTRAP_STAGING=""
BOOTSTRAP_WORKDIR=""
JUICEFS_MOUNT_POINT=""
JUICEFS_CACHE_PATH=""
MAIN_CHILD_PID=""
LOG_SINK_PID=""
CLEANUP_IN_PROGRESS=0

terminate_main() {
    [[ -n "$MAIN_CHILD_PID" ]] || return 0
    kill -0 "$MAIN_CHILD_PID" 2>/dev/null || return 0

    echo "[bootstrap] stopping process group $MAIN_CHILD_PID"
    kill -TERM -- "-$MAIN_CHILD_PID" 2>/dev/null || \
        kill -TERM "$MAIN_CHILD_PID" 2>/dev/null || true
    sleep "${THESEUS_MAIN_CHILD_TERM_GRACE_SECONDS:-3}"
    kill -KILL -- "-$MAIN_CHILD_PID" 2>/dev/null || \
        kill -KILL "$MAIN_CHILD_PID" 2>/dev/null || true
}

cleanup() {
    local exit_code="${1:-$?}"
    [[ "$CLEANUP_IN_PROGRESS" -eq 0 ]] || return 0
    CLEANUP_IN_PROGRESS=1
    set +e

    echo "[bootstrap] cleanup started with exit code $exit_code"
    terminate_main
    if [[ -n "$BOOTSTRAP_STAGING" ]]; then
        echo "[bootstrap] removing staging directory $BOOTSTRAP_STAGING"
        rm -rf "$BOOTSTRAP_STAGING"
    fi
    if [[ "$exit_code" -eq 0 ]] && [[ -n "$BOOTSTRAP_WORKDIR" ]]; then
        echo "[bootstrap] removing work directory $BOOTSTRAP_WORKDIR"
        rm -rf "$BOOTSTRAP_WORKDIR"
    elif [[ -n "$BOOTSTRAP_WORKDIR" ]]; then
        echo "[bootstrap] preserving failed work directory: $BOOTSTRAP_WORKDIR"
    fi

    if [[ -n "$LOG_SINK_PID" ]]; then
        echo "[bootstrap] draining log sink"
        exec 1>&3 2>&4
        wait "$LOG_SINK_PID"
        exec 3>&- 4>&-
    fi
    if [[ -n "$JUICEFS_MOUNT_POINT" ]] && \
        mountpoint -q "$JUICEFS_MOUNT_POINT" 2>/dev/null; then
        echo "[bootstrap] unmounting JuiceFS from $JUICEFS_MOUNT_POINT"
        juicefs umount "$JUICEFS_MOUNT_POINT" 2>/dev/null || \
            juicefs umount --force "$JUICEFS_MOUNT_POINT" 2>/dev/null || \
            echo "[bootstrap] WARNING: failed to unmount $JUICEFS_MOUNT_POINT"
    fi
    if [[ -n "$JUICEFS_CACHE_PATH" ]]; then
        if [[ -n "$JUICEFS_MOUNT_POINT" ]] && \
            mountpoint -q "$JUICEFS_MOUNT_POINT" 2>/dev/null; then
            echo "[bootstrap] WARNING: preserving cache for mounted JuiceFS client: $JUICEFS_CACHE_PATH"
        else
            echo "[bootstrap] removing JuiceFS client cache $JUICEFS_CACHE_PATH"
            rm -rf -- "$JUICEFS_CACHE_PATH"
        fi
    fi
}

handle_signal() {
    local signal_name="$1"
    local exit_code=143
    [[ "$signal_name" == INT ]] && exit_code=130
    [[ "$signal_name" == HUP ]] && exit_code=129
    trap '' TERM INT HUP
    terminate_main
    exit "$exit_code"
}

python_works() {
    "$1" - <<'PY' >/dev/null 2>&1
import asyncio
import base64
import concurrent.futures
import ctypes
import dataclasses
import email
import encodings
import hashlib
import http.client
import importlib.metadata
import json
import logging
import multiprocessing
import pathlib
import pickle
import shutil
import socket
import sqlite3
import ssl
import subprocess
import tarfile
import tempfile
import threading
import uuid
import venv
import xml.etree.ElementTree
import zipfile
import zlib
PY
}

uv_with_timeout() {
    local status=0
    timeout --signal=TERM --kill-after=10s \
        "$THESEUS_UV_OPERATION_TIMEOUT_SECONDS" uv "$@" || status=$?
    if [[ "$status" == 124 || "$status" == 137 ]]; then
        reset_uv_cache || return
        timeout --signal=TERM --kill-after=10s \
            "$THESEUS_UV_OPERATION_TIMEOUT_SECONDS" uv "$@"
    else
        return "$status"
    fi
}

reset_uv_cache() {
    echo "[bootstrap] cleaning UV cache with UV's in-use protection" >&2
    # Never force removal: a timed-out waiter does not prove its owner is hung.
    timeout --signal=TERM --kill-after=10s \
        "$THESEUS_UV_OPERATION_TIMEOUT_SECONDS" uv cache clean >&2
}

sync_environment() {
    if [[ -f uv.lock ]]; then
        uv_with_timeout sync --no-default-groups --python "$PYTHON_BIN" --frozen \
            "${UV_GROUP_ARGUMENTS[@]}"
    else
        uv_with_timeout sync --no-default-groups --python "$PYTHON_BIN" \
            "${UV_GROUP_ARGUMENTS[@]}"
    fi
}

trap 'echo "[bootstrap] ERROR exit=$? line=${BASH_LINENO[0]:-unknown}"' ERR
trap 'cleanup "$?"' EXIT
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap 'handle_signal HUP' HUP

echo
echo " ░▀█▀░█░█░█▀▀░█▀▀░█▀▀░█░█░█▀▀ "
echo " ░░█░░█▀█░█▀▀░▀▀█░█▀▀░█░█░▀▀█ "
echo " ░░▀░░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀ "
echo
echo "[bootstrap] starting on $(hostname)"
echo "[bootstrap] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-NOT_SET}"
echo "[bootstrap] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT_SET}"

if [[ "$#" -ne 1 ]]; then
    echo "[bootstrap] ERROR: usage: bootstrap.sh dispatchspec.json"
    exit 2
fi
DISPATCH_SPEC_INPUT="$1"
if [[ ! -r "$DISPATCH_SPEC_INPUT" ]]; then
    echo "[bootstrap] ERROR: DispatchSpec is not readable: $DISPATCH_SPEC_INPUT"
    exit 2
fi

#### local staging ####

# Some SLURM installations export a small or stale per-user TMPDIR. Probe local
# scratch locations in order, then use the dispatch's shared directory only as a
# last resort so bootstrap can still report and recover from a full node tmpfs.
echo "[bootstrap] phase=staging selecting temporary storage"
staging_roots=(
    "${SLURM_TMPDIR:-}"
    "${TMPDIR:-}"
    /tmp
    /var/tmp
    "$(dirname "$DISPATCH_SPEC_INPUT")"
)
for staging_root in "${staging_roots[@]}"; do
    [[ -n "$staging_root" && -d "$staging_root" && -w "$staging_root" ]] || continue
    if BOOTSTRAP_STAGING="$(
        mktemp -d "$staging_root/theseus-bootstrap.XXXXXX" 2>/dev/null
    )"; then
        break
    fi
done
[[ -n "$BOOTSTRAP_STAGING" ]] || {
    echo "[bootstrap] ERROR: no writable temporary staging directory"
    exit 1
}
export TMPDIR="$BOOTSTRAP_STAGING"
echo "[bootstrap] staging directory: $BOOTSTRAP_STAGING"

#### bootstrap tools ####

echo "[bootstrap] phase=tools checking bootstrap dependencies"
missing_packages=()
command -v jq >/dev/null 2>&1 || missing_packages+=(jq)
if ! command -v uv >/dev/null 2>&1 && \
    ! command -v curl >/dev/null 2>&1 && \
    ! command -v wget >/dev/null 2>&1; then
    missing_packages+=(curl)
fi
if [[ "${#missing_packages[@]}" -gt 0 ]]; then
    echo "[bootstrap] missing tools: ${missing_packages[*]}"
    if [[ "$(uname -s)" != Linux ]] || \
        ! command -v apt-get >/dev/null 2>&1; then
        echo "[bootstrap] ERROR: automatic prerequisite installation is available only on Linux with apt-get and non-interactive installation privileges"
        exit 1
    fi
    if [[ "$EUID" -eq 0 ]]; then
        apt_install=(apt-get)
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        apt_install=(sudo -n apt-get)
    else
        echo "[bootstrap] ERROR: installing ${missing_packages[*]} requires root or passwordless sudo"
        exit 1
    fi
    "${apt_install[@]}" update -qq
    "${apt_install[@]}" install -y -qq "${missing_packages[@]}"
fi
command -v jq >/dev/null 2>&1 || {
    echo "[bootstrap] ERROR: jq installation failed"
    exit 1
}

if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] installing uv"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        wget -qO- https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
    echo "[bootstrap] ERROR: uv installation failed"
    exit 1
}

#### packed dispatch ####

echo "[bootstrap] phase=dispatch unpacking DispatchSpec"
DISPATCH_SPEC_PATH="$BOOTSTRAP_STAGING/dispatch.json"
cp "$DISPATCH_SPEC_INPUT" "$DISPATCH_SPEC_PATH"

dispatch_fields="$BOOTSTRAP_STAGING/dispatch-fields"
mkdir -p "$dispatch_fields"
MACHINE_INDEX="${THESEUS_MACHINE_INDEX:-${SLURM_PROCID:-0}}"
if [[ ! "$MACHINE_INDEX" =~ ^[0-9]+$ ]]; then
    echo "[bootstrap] ERROR: invalid machine index: $MACHINE_INDEX"
    exit 1
fi
MACHINE_INDEX="$((10#$MACHINE_INDEX))"

if ! jq -ej --argjson machine_index "$MACHINE_INDEX" '
    def nul_free: (explode | index(0)) == null;
    def text($name):
        if type == "string" and nul_free then .
        else error("Dispatch field \($name) must be a NUL-free string")
        end;
    def optional_text($name):
        if . == null then "" else text($name) end;

    .hardware.hosts as $hosts
    | if ($hosts | type) != "array" or ($hosts | length) == 0 then
        error("DispatchSpec.hardware.hosts must not be empty")
      elif $machine_index >= ($hosts | length) then
        error("Machine index \($machine_index) is outside DispatchSpec.hardware.hosts")
      else . end
    | $hosts[$machine_index] as $machine
    | $machine.cluster as $cluster
    | [
        ["PROJECT", (.project | text("project"))],
        ["GROUP", (.group | text("group"))],
        ["NAME", (.name | text("name"))],
        ["NONCE", (.nonce | text("nonce"))],
        ["WORK_ROOT", ($cluster.work | text("cluster.work"))],
        ["LOG_DIR", (($cluster.log // ($cluster.work + "/logs")) | text("cluster.log"))],
        ["HOST_COUNT", ($hosts | length | tostring)],
        ["CLUSTER_ROOT", ($cluster.root | text("cluster.root"))],
        ["JUICEFS_URL", ($cluster.mount | optional_text("cluster.mount"))],
        ["JUICEFS_CACHE_SIZE", ($cluster.cache_size | optional_text("cluster.cache_size"))],
        ["JUICEFS_CACHE_DIR", ($cluster.cache_dir | optional_text("cluster.cache_dir"))],
        ["JUICEFS_ALL_SQUASH", ($cluster.all_squash | optional_text("cluster.all_squash"))]
      ]
    | .[]
    | "\(.[0])\u0000\(.[1])\u0000"
' "$DISPATCH_SPEC_PATH" > "$dispatch_fields/shell"; then
    echo "[bootstrap] ERROR: invalid DispatchSpec"
    exit 1
fi
while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    printf -v "$name" '%s' "$value"
done < "$dispatch_fields/shell"

if ! jq -ej --argjson machine_index "$MACHINE_INDEX" '
    def nul_free: (explode | index(0)) == null;
    def environment_value($name):
        if type == "string" and nul_free then .
        else error("Environment variable \($name) must be a NUL-free string")
        end;

    .hardware.hosts[$machine_index] as $machine
    | ({"THESEUS_ROOT": $machine.cluster.root} + ($machine.env // {}))
    | to_entries[]
    | .key as $name
    | if ($name | test("^[A-Za-z_][A-Za-z0-9_]*$")) then
        "\($name)\u0000\(.value | environment_value($name))\u0000"
      else error("Invalid environment variable name: \($name)")
      end
' "$DISPATCH_SPEC_PATH" > "$dispatch_fields/environment"; then
    echo "[bootstrap] ERROR: invalid dispatch environment"
    exit 1
fi

if ! jq -j --argjson machine_index "$MACHINE_INDEX" '
    def nul_free: (explode | index(0)) == null;
    .hardware.hosts[$machine_index].uv_groups // []
    | .[]
    | if type == "string" and nul_free then
        . + "\u0000"
      else error("uv groups must be NUL-free strings")
      end
' "$DISPATCH_SPEC_PATH" > "$dispatch_fields/uv-groups"; then
    echo "[bootstrap] ERROR: invalid uv groups"
    exit 1
fi

if ! jq -ejr '
    .job.config
    | if type == "string" and ((explode | index(0)) == null) then .
      else error("Job config must be a NUL-free string")
      end
' \
    "$DISPATCH_SPEC_PATH" > "$BOOTSTRAP_STAGING/config.yaml"; then
    echo "[bootstrap] ERROR: invalid job config"
    exit 1
fi

echo "[bootstrap] phase=environment applying runtime environment"
while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    printf -v "$name" '%s' "$value"
    export "$name"
done < "$dispatch_fields/environment"

: "${THESEUS_UV_OPERATION_TIMEOUT_SECONDS:=300}"
if [[ ! "$THESEUS_UV_OPERATION_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[bootstrap] ERROR: invalid UV operation timeout: $THESEUS_UV_OPERATION_TIMEOUT_SECONDS"
    exit 1
fi
: "${UV_LOCK_TIMEOUT:=$THESEUS_UV_OPERATION_TIMEOUT_SECONDS}"
export UV_LOCK_TIMEOUT
PYTHON_BIN="$(uv_with_timeout python find --no-project 3.11 2>/dev/null || true)"
if [[ -z "$PYTHON_BIN" ]] || ! python_works "$PYTHON_BIN"; then
    echo "[bootstrap] repairing Python 3.11 through uv"
    uv_with_timeout python install --reinstall 3.11
    PYTHON_BIN="$(uv_with_timeout python find --managed-python --no-project 3.11)"
fi
if [[ -z "$PYTHON_BIN" ]] || ! python_works "$PYTHON_BIN"; then
    echo "[bootstrap] ERROR: uv could not provide a complete Python 3.11"
    exit 1
fi

echo "[bootstrap] phase=logging opening durable log"
mkdir -p "$LOG_DIR"
log_project="${PROJECT//\//_}"
log_group="${GROUP//\//_}"
log_name="${NAME//\//_}"
log_nonce="${NONCE//\//_}"
LOG_FILE="$LOG_DIR/${log_project}-${log_group}-${log_name}-${log_nonce}.${MACHINE_INDEX}.log"
if [[ "${THESEUS_STDOUT_MANAGED:-0}" != 1 ]]; then
    exec 3>&1 4>&2
    exec > >(tee -a "$LOG_FILE" >&3) 2>&1
    LOG_SINK_PID=$!
fi
echo "[bootstrap] dispatch nonce=$NONCE log=$LOG_FILE"

echo "[bootstrap] phase=repository validating packed repository"
REPO_BUNDLE_PATH="$BOOTSTRAP_STAGING/repo.tar.gz"
base64 -d > "$REPO_BUNDLE_PATH" <<'__REPO_BUNDLE_EOF__'
__REPO_BUNDLE__
__REPO_BUNDLE_EOF__
tar -tzf "$REPO_BUNDLE_PATH" >/dev/null

#### juicefs ####

echo "[bootstrap] phase=storage checking JuiceFS"
if [[ -n "$JUICEFS_URL" ]] && ! command -v juicefs >/dev/null 2>&1; then
    if [[ "$(uname -s)" == Linux ]]; then
        echo "[bootstrap] installing JuiceFS"
        mkdir -p "$HOME/.local/bin"
        if command -v curl >/dev/null 2>&1; then
            curl -sSL https://d.juicefs.com/install | sh -s "$HOME/.local/bin"
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://d.juicefs.com/install | sh -s "$HOME/.local/bin"
        else
            echo "[bootstrap] ERROR: installing JuiceFS requires curl or wget"
            exit 1
        fi
    else
        echo "[bootstrap] ERROR: automatic JuiceFS installation is available only on Linux and uses a user-local, sudoless install"
        exit 1
    fi
fi
[[ -z "$JUICEFS_URL" ]] || command -v juicefs >/dev/null 2>&1 || {
    echo "[bootstrap] ERROR: JuiceFS installation failed"
    exit 1
}

#### cluster setup ####

echo "[bootstrap] phase=storage preparing cluster mount"
if [[ -n "$JUICEFS_URL" ]]; then
    MOUNT_POINT="$CLUSTER_ROOT"
    if ! mountpoint -q "$MOUNT_POINT"; then
        echo "[bootstrap] mounting JuiceFS at $MOUNT_POINT"
        mkdir -p "$MOUNT_POINT"
        juicefs_arguments=(--quiet mount --umask=0000 -d)
        [[ -z "$JUICEFS_CACHE_SIZE" ]] || \
            juicefs_arguments+=(--cache-size "$JUICEFS_CACHE_SIZE")
        if [[ -n "$JUICEFS_CACHE_DIR" ]]; then
            # Each JuiceFS client maintains its own in-memory cache index. Give
            # concurrent dispatches separate disk caches while preserving the
            # configured directory as their common storage root.
            cache_nonce="${NONCE//[^[:alnum:]_.-]/_}"
            if [[ "$JUICEFS_CACHE_DIR" == memory ]]; then
                client_cache_dir=memory
            else
                client_cache_dir="${JUICEFS_CACHE_DIR%/}/theseus-$cache_nonce-$MACHINE_INDEX"
                JUICEFS_CACHE_PATH="$client_cache_dir"
            fi
            juicefs_arguments+=(--cache-dir "$client_cache_dir")
        fi
        [[ -z "$JUICEFS_ALL_SQUASH" ]] || \
            juicefs_arguments+=(--all-squash "$JUICEFS_ALL_SQUASH")
        juicefs_arguments+=("$JUICEFS_URL" "$MOUNT_POINT")
        juicefs "${juicefs_arguments[@]}"
        JUICEFS_MOUNT_POINT="$MOUNT_POINT"
    else
        echo "[bootstrap] JuiceFS mount already available at $MOUNT_POINT"
    fi
else
    echo "[bootstrap] no JuiceFS mount requested"
fi

UV_GROUP_ARGUMENTS=()
while IFS= read -r -d '' group; do
    UV_GROUP_ARGUMENTS+=(--group "$group")
done < "$dispatch_fields/uv-groups"

: "${XLA_PYTHON_CLIENT_MEM_FRACTION:=0.95}"
export XLA_PYTHON_CLIENT_MEM_FRACTION

# Opt out of Theseus defaults without overriding user-supplied JAX settings.
if [[ "${THESEUS_DISABLE_OPTIMIZATIONS:-0}" != 1 ]]; then
    : "${JAX_OPTIMIZATION_LEVEL:=O2}"
    : "${JAX_ENABLE_PGLE:=true}"
    : "${JAX_PGLE_PROFILING_RUNS:=3}"
    export JAX_OPTIMIZATION_LEVEL JAX_ENABLE_PGLE JAX_PGLE_PROFILING_RUNS
fi

# If CUDA_ERROR_ILLEGAL_ADDRESS recurs around cross-node NCCL collectives,
# temporarily set XLA_PYTHON_CLIENT_ALLOCATOR=platform and
# XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER,ALLGATHER"
# through the dispatch environment overrides; these are not safe global defaults.

echo "[bootstrap] phase=repository extracting work tree"
RUN_ROOT="$WORK_ROOT/$PROJECT/$GROUP/$NAME/$NONCE"
if ! mkdir -p "$RUN_ROOT"; then
    echo "[bootstrap] WARNING: configured work root is unavailable: $WORK_ROOT"
    RUN_ROOT="$BOOTSTRAP_STAGING/work"
    mkdir -p "$RUN_ROOT"
    echo "[bootstrap] using temporary work root: $RUN_ROOT"
fi
BOOTSTRAP_WORKDIR="$RUN_ROOT/$MACHINE_INDEX"
if [[ -e "$BOOTSTRAP_WORKDIR" ]]; then
    preserved="$BOOTSTRAP_WORKDIR.failed.$(date +%Y%m%d_%H%M%S).$$"
    echo "[bootstrap] preserving previous work directory: $preserved"
    mv "$BOOTSTRAP_WORKDIR" "$preserved"
fi
mkdir "$BOOTSTRAP_WORKDIR"
tar -xzf "$REPO_BUNDLE_PATH" -C "$BOOTSTRAP_WORKDIR"
cp "$DISPATCH_SPEC_PATH" "$BOOTSTRAP_WORKDIR/dispatch.json"
DISPATCH_SPEC_PATH="$BOOTSTRAP_WORKDIR/dispatch.json"

cd "$BOOTSTRAP_WORKDIR"
echo "[bootstrap] work directory: $BOOTSTRAP_WORKDIR"

echo "[bootstrap] phase=dependencies syncing environment"
if ! sync_environment; then
    # Preserve the existing unfrozen fallback for a stale lockfile.
    [[ -f uv.lock ]] || exit 1
    uv_with_timeout sync --no-default-groups --python "$PYTHON_BIN" \
        "${UV_GROUP_ARGUMENTS[@]}"
fi
if ! python_works .venv/bin/python; then
    reset_uv_cache
    rm -rf -- .venv
    sync_environment
    python_works .venv/bin/python || {
        echo "[bootstrap] ERROR: synced Python environment is incomplete"
        exit 1
    }
fi

# Bootstrap inputs remain owner-only, but scientific artifacts must preserve
# the cluster filesystem's ordinary cross-client readability.
umask 022

#### autobatch ####

echo "[bootstrap] phase=autobatch inspecting batch-size configuration"
: "${THESEUS_AUTOBATCH_SIZE_CANDIDATES:=1024 512 256 128 64 32 16 10 8 4 3 2 1}"
read -r -a BATCH_SIZE_CANDIDATES <<< "$THESEUS_AUTOBATCH_SIZE_CANDIDATES"
for candidate in "${BATCH_SIZE_CANDIDATES[@]}"; do
    if [[ ! "$candidate" =~ ^[1-9][0-9]*$ ]]; then
        echo "[bootstrap] ERROR: invalid autobatch candidate: $candidate"
        exit 1
    fi
done
batch_size_override=""
probe_overrides=()
failed_effective_batch=""
probe_overrides+=("logging.report_interval=1")
if grep -Eq '^[[:space:]]*remote[[:space:]]*:[[:space:]]*true' \
    "$BOOTSTRAP_STAGING/config.yaml"; then
    probe_overrides+=("logging.remote=false")
fi
if grep -Eq \
    '^[[:space:]]*per_device_batch_size[[:space:]]*[:=][[:space:]]*-1([[:space:]]|$)' \
    "$BOOTSTRAP_STAGING/config.yaml"; then
    if [[ "$HOST_COUNT" -gt 1 ]]; then
        echo "[bootstrap] ERROR: autobatch is unsafe for multi-host dispatches"
        exit 1
    fi

    probe_timeout="${THESEUS_DISPATCH_INITIAL_STABILITY_TIMEOUT:-3600}"
    probe_id="${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)-$$}"
    for candidate in "${BATCH_SIZE_CANDIDATES[@]}"; do
        if [[ -n "$failed_effective_batch" ]] && \
            [[ "$candidate" -ge "$failed_effective_batch" ]]; then
            echo "[bootstrap] skipping duplicate effective batch candidate $candidate"
            continue
        fi
        probe_spec="$BOOTSTRAP_STAGING/dispatch-autobatch-${candidate}.json"
        probe_log="$BOOTSTRAP_STAGING/autobatch-${candidate}.log"
        "$PYTHON_BIN" -c \
            'import json, sys; d=json.load(open(sys.argv[1])); d["nonce"]=sys.argv[3]; json.dump(d, open(sys.argv[2], "w"))' \
            "$DISPATCH_SPEC_PATH" "$probe_spec" \
            "${NONCE}-${probe_id}-autobatch-${candidate}"

        echo "[bootstrap] probing training.per_device_batch_size=$candidate"
        # Create the buffer before polling, and keep the writer PID so a
        # fast-failing probe cannot be classified before its output drains.
        : > "$probe_log"
        probe_pipe="$BOOTSTRAP_STAGING/autobatch-${candidate}.fifo"
        mkfifo "$probe_pipe"
        tee "$probe_log" < "$probe_pipe" &
        probe_log_sink_pid=$!
        set +e
        # The probe tee keeps a classification buffer; its stdout still
        # flows through the outer tee into the durable cluster log.
        LC_ALL=C setsid timeout --verbose --signal=TERM --kill-after=10 \
            "$probe_timeout" \
            uv run --no-sync python -m theseus.execute.run "$probe_spec" \
                "training.per_device_batch_size=$candidate" \
                "${probe_overrides[@]}" \
            > "$probe_pipe" 2>&1 &
        MAIN_CHILD_PID=$!
        probe_completed=0
        while kill -0 "$MAIN_CHILD_PID" 2>/dev/null; do
            if grep -Eq 'TRAIN \| [0-9]+/[0-9]+ \| loss ' "$probe_log"; then
                probe_completed=1
                terminate_main
                break
            fi
            sleep 1
        done
        wait "$MAIN_CHILD_PID"
        probe_exit_code=$?
        MAIN_CHILD_PID=""
        wait "$probe_log_sink_pid"
        probe_log_exit_code=$?
        set -e
        rm -f "$probe_pipe"

        if [[ "$probe_log_exit_code" -ne 0 ]]; then
            echo "[bootstrap] ERROR: autobatch log sink failed with exit $probe_log_exit_code"
            exit "$probe_log_exit_code"
        fi
        if [[ "$probe_exit_code" -eq 0 || "$probe_completed" -eq 1 ]]; then
            batch_size_override="training.per_device_batch_size=$candidate"
            break
        fi
        if ! grep -Eq \
            "RESOURCE_EXHAUSTED|JaxRuntimeError: INTERNAL:|Can't reduce memory use below" \
            "$probe_log"; then
            echo "[bootstrap] ERROR: autobatch probe failed with exit $probe_exit_code"
            exit "$probe_exit_code"
        fi
        failed_effective_batch="$(
            sed -nE \
                's/.*BATCHING \| ([0-9]+) batchsize\/node.*/\1/p' \
                "$probe_log" | tail -1
        )"
        [[ -n "$failed_effective_batch" ]] || failed_effective_batch="$candidate"

        sleep "${THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS:-8}"
    done

    if [[ -z "$batch_size_override" ]]; then
        echo "[bootstrap] ERROR: no viable per-device batch size found"
        exit 1
    fi
    sleep "${THESEUS_DISPATCH_PROBE_COOLDOWN_SECONDS:-8}"
fi

#### dispatch! ####

echo "[bootstrap] phase=execution launching ${batch_size_override:-configured batch size}"
dispatch_command=(
    uv run --no-sync python -m theseus.execute.run "$DISPATCH_SPEC_PATH"
)
if [[ -n "$batch_size_override" ]]; then
    dispatch_command+=("$batch_size_override")
fi

setsid "${dispatch_command[@]}" &
MAIN_CHILD_PID=$!
set +e
wait "$MAIN_CHILD_PID"
main_exit_code=$?
set -e
MAIN_CHILD_PID=""
exit "$main_exit_code"
