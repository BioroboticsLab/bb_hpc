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

# Save a copy of the configuration for record-keeping
timestamp=$(date +"%Y%m%d_%H%M%S")
config_backup_dir="./transfer_files_config_backups"
mkdir -p "$config_backup_dir"
cp -f -- "$config_file" "$config_backup_dir/config_${timestamp}.sh"

# Ensure paths exist (destination parent ok; subdirs created per-dir below)
[ -d "$source" ] || die "Source directory does not exist: $source"
mkdir -p "$(dirname "$destination")"

# ====================================
# Directory discovery (top-level only)
# ====================================
# If directories[] is unset or empty, discover all top-level directories under $source
if [ -z "${directories+x}" ] || [ "${#directories[@]}" -eq 0 ]; then
    echo "No directories specified; discovering all top-level directories in: $source"
    if [ ! -d "$source" ]; then
        echo "Source directory $source does not exist!"
        exit 1
    fi
    # NUL-safe discovery + sort, then store only basenames
    mapfile -d '' -t directories < <(find "$source" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    for i in "${!directories[@]}"; do
        directories[$i]=$(basename "${directories[$i]}")
    done
fi

# ===========================
# Helpers / verification step
# ===========================
# Build two *sorted* lists of "size path" for files present in BOTH source & dest.
# - NUL-safe traversal, but we format output lines as "size relative_path\n" to diff quickly.
# - Optional extension filter: "" or "*" => all files; else only *.<ext>
generate_common_file_sizes() {
  local source_dir=$1
  local destination_dir=$2
  local src_size_list=$3
  local dest_size_list=$4
  local ext_filter=$5   # "" or "*" or "mp4" etc.

  rm -f -- "$src_size_list" "$dest_size_list"
  : >"$src_size_list"
  : >"$dest_size_list"

  # Select predicate for find
  local find_predicate=(-type f)
  if [ -n "$ext_filter" ] && [ "$ext_filter" != "*" ]; then
    find_predicate=(-type f -name "*.${ext_filter#*.}")
  fi

  # We’ll collect candidates from source, check if they exist in dest, and dump "size relpath"
  # NUL-safe walk
  while IFS= read -r -d '' src_file; do
    # Compute relative path; handle root-copy case cleanly
    local rel=${src_file#"$source_dir"/}
    local dst="$destination_dir/$rel"
    if [ -f "$dst" ]; then
      # stat sizes (portable flags: GNU coreutils in most linuxes)
      local s1 s2
      s1=$(stat --format="%s" -- "$src_file") || s1=""
      s2=$(stat --format="%s" -- "$dst")      || s2=""
      if [ -n "$s1" ] && [ -n "$s2" ]; then
        printf "%s %s\n" "$s1" "$rel" >>"$src_size_list"
        printf "%s %s\n" "$s2" "$rel" >>"$dest_size_list"
      fi
    fi
  done < <(find "$source_dir" -mindepth 1 "${find_predicate[@]}" -print0 | sort -z)
  sort -o "$src_size_list" "$src_size_list"
  sort -o "$dest_size_list" "$dest_size_list"
}

# ===================
# Rsync include rules
# ===================
# Build rsync args for extension filtering.
rsync_filter_args_all=()   # copy everything
rsync_filter_args_ext=(
  --include='*/'
  --include="*.${file_extension#*.}"
  --exclude='*'
)

use_all_files=true
if [ -n "$file_extension" ] && [ "$file_extension" != "*" ]; then
  use_all_files=false
fi

# =================
# Transfer per dir
# =================
for dir in "${directories[@]}"; do
  if [ -n "$dir" ]; then
    src_dir="${source%/}/$dir"
    dest_dir="${destination%/}/$dir"
  else
    # Fallback case: copy the root when no subdirs exist
    src_dir="${source%/}"
    dest_dir="${destination%/}"
  fi

  if [ ! -d "$src_dir" ]; then
    echo "Skipping missing directory: $src_dir"
    continue
  fi

  echo "------------------------------------------------------------"
  echo "Transferring: $src_dir  →  $dest_dir"
  mkdir -p "$dest_dir"

  # First pass: copy
  echo "First rsync pass..."
  if $use_all_files; then
    echo "Including ALL files (no extension filtering)."
    rsync -av --progress --partial --inplace --no-whole-file \
      "$src_dir/" "$dest_dir/"
  else
    echo "Including only *.$file_extension files."
    rsync -av --progress --partial --inplace --no-whole-file \
      "${rsync_filter_args_ext[@]}" \
      "$src_dir/" "$dest_dir/"
  fi

  # Verify via size-lists (fast) before deleting
  if [ "$check_and_remove" = true ]; then
    echo "Verifying by size lists before removal..."
    src_size_list="/tmp/src_size_list_${timestamp}_$(echo "$dir" | tr '/ ' '__').txt"
    dest_size_list="/tmp/dest_size_list_${timestamp}_$(echo "$dir" | tr '/ ' '__').txt"

    generate_common_file_sizes "$src_dir" "$dest_dir" "$src_size_list" "$dest_size_list" \
      "$([ "$use_all_files" = true ] && echo "" || echo "$file_extension")"

    if diff -q "$src_size_list" "$dest_size_list" >/dev/null; then
      echo "✅ Size lists match — proceeding to remove source files for this selection."
      echo "Second rsync pass with --remove-source-files..."
      if $use_all_files; then
        rsync -av --remove-source-files --prune-empty-dirs --progress --partial \
          "$src_dir/" "$dest_dir/"
      else
        rsync -av --remove-source-files --prune-empty-dirs --progress --partial \
          "${rsync_filter_args_ext[@]}" \
          "$src_dir/" "$dest_dir/"
      fi

      # Remove any empty directories left behind
      find "$src_dir" -type d -empty -delete || true
    else
      echo "❌ Size lists differ — NOT removing source files for: $dir"
      echo "You can inspect: $src_size_list  vs  $dest_size_list"
    fi
  fi
done

echo "✅ Transfer complete."