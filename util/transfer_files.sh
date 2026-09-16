#!/usr/bin/env bash
set -euo pipefail

die() { echo "Error: $*" >&2; exit 1; }

# ======================
# Usage & Config loading
# ======================
if [ "$#" -ne 1 ]; then
  die "Usage: $0 <config_file>"
fi

config_file="$1"
[ -f "$config_file" ] || die "Configuration file not found: $config_file"

# shellcheck source=/dev/null
source "$config_file"

: "${source:?source path is required in config}"
: "${destination:?destination path is required in config}"
file_extension="${file_extension:-*}"           # "" or "*" => copy everything
check_and_remove="${check_and_remove:-false}"   # default: false

# ===================================
# Local vs. remote destination
# ===================================
# rsync's own rule: a spec is remote when it contains a ":" before the first "/".
# So "user@host:/path" and "host:/path" are remote, "/mnt/a:b/c" is local.
is_remote_spec() { [[ $1 == *:* && ${1%%:*} != */* ]]; }

case "$source" in rsync://*) die "rsync daemon URLs are not supported: $source" ;; esac
case "$destination" in rsync://*) die "rsync daemon URLs are not supported: $destination" ;; esac

if is_remote_spec "$source"; then
  die "Remote source is not supported: $source
     The source must be a local path; only the destination may be [user@]host:/path."
fi

dest_is_remote=false
dest_host=""
if is_remote_spec "$destination"; then
  dest_is_remote=true
  dest_host="${destination%%:*}"
  echo "Remote destination: host='$dest_host' path='${destination#*:}'"
  echo "(ssh key-based auth and rsync on the remote host are assumed)"
fi

# Create a destination directory, locally or over ssh.
# $1 = full destination spec (with the "host:" prefix when remote).
dest_mkdir() {
  if [ "$dest_is_remote" = true ]; then
    ssh "$dest_host" "mkdir -p -- $(printf '%q' "${1#*:}")"
  else
    mkdir -p -- "$1"
  fi
}

# Save a copy of the configuration for record-keeping
timestamp=$(date +"%Y%m%d_%H%M%S")
config_backup_dir="./transfer_files_config_backups"
mkdir -p "$config_backup_dir"
cp -f -- "$config_file" "$config_backup_dir/config_${timestamp}.sh"

# Ensure paths exist (subdirs are created per-dir below).
# For a remote destination this first ssh call also surfaces connection/auth
# problems immediately, instead of after N failed rsync runs.
[ -d "$source" ] || die "Source directory does not exist: $source"
dest_mkdir "${destination%/}" || die "Could not create destination: ${destination%/}"

# ====================================
# Directory discovery (top-level only)
# ====================================
# If directories[] is unset or empty, discover all top-level directories under $source
if [ -z "${directories+x}" ] || [ "${#directories[@]}" -eq 0 ]; then
    echo "No directories specified; discovering all top-level directories in: $source"
    # NUL-safe discovery + sort, then store only basenames
    mapfile -d '' -t directories < <(find "$source" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    for i in "${!directories[@]}"; do
        directories[$i]=$(basename "${directories[$i]}")
    done
    # No subdirectories at all: fall back to copying the source root itself
    if [ "${#directories[@]}" -eq 0 ]; then
        directories=("")
    fi
fi

# ===================
# Rsync option sets
# ===================
# Each array is kept non-empty so it expands safely under `set -u`.
rsync_copy_opts=(-av --progress --partial --inplace --no-whole-file)
rsync_move_opts=(-av --progress --partial --remove-source-files --prune-empty-dirs)
rsync_verify_opts=(-a --dry-run --itemize-changes)

use_all_files=true
if [ -n "$file_extension" ] && [ "$file_extension" != "*" ]; then
  use_all_files=false
  ext_filter=(--include='*/' --include="*.${file_extension#*.}" --exclude='*')
  rsync_copy_opts+=("${ext_filter[@]}")
  rsync_move_opts+=("${ext_filter[@]}")
  rsync_verify_opts+=("${ext_filter[@]}")
  echo "Including only *.${file_extension#*.} files."
else
  echo "Including ALL files (no extension filtering)."
fi

# ===========================
# Verification step
# ===========================
# Ask rsync itself what would still have to be transferred. This works the same
# way for a local and a remote destination, and unlike a stat/size comparison it
# also catches files that are missing at the destination entirely.
#
# Returns: 0 = destination matches source, 1 = files still differ, 2 = rsync error.
# A failed rsync (dropped ssh connection, etc.) must never be read as "verified".
verify_transferred() {
  local src_dir=$1 dest_dir=$2 report=$3
  local rc=0

  rsync "${rsync_verify_opts[@]}" "$src_dir/" "$dest_dir/" \
    >"${report}.raw" 2>"${report}.err" || rc=$?

  if [ "$rc" -ne 0 ]; then
    echo "rsync dry-run failed (exit $rc); see ${report}.err" >&2
    sed 's/^/    /' "${report}.err" >&2 || true
    return 2
  fi

  # Only "<f"/">f" itemize lines mean a file would still be sent:
  #   >f+++++++++  missing at the destination
  #   >f.st......  size / timestamp differ
  # Attribute-only lines (e.g. ".f...og..." when the remote user cannot set
  # owner/group) are not a transfer problem and are ignored.
  grep -E '^[<>]f' "${report}.raw" >"$report" || true
  [ ! -s "$report" ]
}

# =================
# Transfer per dir
# =================
results=()
record() { results+=("$1|$2"); }

for dir in "${directories[@]}"; do
  if [ -n "$dir" ]; then
    src_dir="${source%/}/$dir"
    dest_dir="${destination%/}/$dir"
    label="$dir"
  else
    # Fallback case: copy the root when no subdirs exist
    src_dir="${source%/}"
    dest_dir="${destination%/}"
    label="(root)"
  fi

  if [ ! -d "$src_dir" ]; then
    echo "Skipping missing directory: $src_dir"
    record "$label" "SKIPPED_NO_SOURCE"
    continue
  fi

  echo "------------------------------------------------------------"
  echo "Transferring: $src_dir  →  $dest_dir"

  if ! dest_mkdir "$dest_dir"; then
    echo "❌ Could not create destination directory: $dest_dir"
    record "$label" "MKDIR_FAILED"
    continue
  fi

  # First pass: copy
  echo "First rsync pass..."
  if ! rsync "${rsync_copy_opts[@]}" "$src_dir/" "$dest_dir/"; then
    echo "❌ rsync failed for: $label — leaving source files in place."
    record "$label" "TRANSFER_FAILED"
    continue
  fi

  if [ "$check_and_remove" != true ]; then
    record "$label" "TRANSFERRED"
    continue
  fi

  # Verify with a dry run before deleting anything
  echo "Verifying with an rsync dry run before removal..."
  report="/tmp/transfer_verify_${timestamp}_$(echo "$label" | tr '/ ' '__').txt"

  verify_rc=0
  verify_transferred "$src_dir" "$dest_dir" "$report" || verify_rc=$?

  case "$verify_rc" in
    0)
      echo "✅ Destination matches source — proceeding to remove source files."
      echo "Second rsync pass with --remove-source-files..."
      if ! rsync "${rsync_move_opts[@]}" "$src_dir/" "$dest_dir/"; then
        echo "❌ Removal pass failed for: $label"
        record "$label" "REMOVE_FAILED"
        continue
      fi
      # Remove any empty directories left behind
      find "$src_dir" -type d -empty -delete || true
      record "$label" "VERIFIED+REMOVED"
      ;;
    1)
      pending=$(wc -l <"$report" | tr -d ' ')
      echo "❌ $pending file(s) still differ — NOT removing source files for: $label"
      echo "You can inspect: $report"
      record "$label" "MISMATCH ($pending files)"
      ;;
    *)
      echo "❌ Could not verify $label — NOT removing source files."
      record "$label" "VERIFY_ERROR"
      ;;
  esac
done

# =================
# Summary
# =================
echo
echo "============================================================"
echo "Summary"
echo "============================================================"

failed=0
skipped=0
if [ "${#results[@]}" -eq 0 ]; then
  echo "  (nothing to do)"
else
  for entry in "${results[@]}"; do
    printf "  %-14s %s\n" "${entry%%|*}" "${entry#*|}"
    case "${entry#*|}" in
      TRANSFERRED|VERIFIED+REMOVED) ;;
      SKIPPED_NO_SOURCE) skipped=$((skipped + 1)) ;;
      *) failed=$((failed + 1)) ;;
    esac
  done
fi

echo "------------------------------------------------------------"
if [ "$failed" -gt 0 ]; then
  echo "❌ Transfer finished with $failed problem(s) ($skipped skipped)."
  exit 1
fi

echo "✅ Transfer complete ($skipped skipped)."
