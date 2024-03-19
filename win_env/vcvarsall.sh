set -x
#!/usr/bin/env bash
#
# Source: https://github.com/nathan818fr/vcvars-bash
# Author: Nathan Poirier <nathan@poirier.io>
#
set -Eeuo pipefail
shopt -s inherit_errexit

function detect_platform() {
  case "${OSTYPE:-}" in
  cygwin* | msys* | win32)
    declare -gr bash_platform='win_cyg'
    ;;
  *)
    if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
      declare -gr bash_platform='win_wsl'
    else
      printf 'error: Unsupported platform (%s)\n' "${OSTYPE:-}" >&2
      printf 'hint: This script only supports Bash on Windows (Git Bash, WSL, etc.)\n' >&2
      return 1
    fi
    ;;
  esac
}

function detect_mode() {
  # Detect the mode depending on the name called
  if [[ "$(basename -- "$0")" == vcvarsrun* ]]; then
    declare -gr script_mode='run' # vcvarsrun.sh
  else
    declare -gr script_mode='default' # vcvarsall.sh
  fi
}

function print_usage() {
  case "$script_mode" in
  default)
    cat <<EOF
Usage: eval "\$(vcvarsall.sh [vcvarsall.bat arguments])"

Load MSVC environment variables using vcvarsall.bat and export them.
The script writes a list of export commands to stdout, to be evaluated by a
POSIX-compliant shell.

Example: eval "\$(vcvarsall.sh x86)"
EOF
    ;;
  run)
    cat <<EOF
Usage: vcvarsrun.sh [vcvarsall.bat arguments] -- command [arguments...]

Load MSVC environment variables using vcvarsall.bat and execute a command with
them.

Example: vcvarsrun.sh x64 -- cl /nologo /EHsc /Fe:hello.exe hello.cpp
EOF
    ;;
  esac
  cat <<EOF

Environment variables:
  VSINSTALLDIR
    The path to the Visual Studio installation directory.
    Example: "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community"
    Default: The latest Visual Studio installation directory found by vswhere.exe
EOF
  local vcvarsall
  vcvarsall=$(find_vcvarsall)
  printf '\nvcvarsall.bat arguments: %s\n' "$({ cmd "$(cmdesc "$vcvarsall")" -help </dev/null 2>/dev/null || true; } | fix_crlf | sed '/^Syntax:/d; s/^    vcvarsall.bat //')"
}

function main() {
  detect_platform
  detect_mode

  # Parse arguments
  if [[ $# -eq 0 ]]; then
    print_usage >&2
    return 1
  fi

  local arg vcvarsall_args=()
  for arg in "$@"; do
    shift
    if [[ "$arg" == '--' ]]; then
      if [[ "$script_mode" == 'default' ]]; then
        printf 'error: Unexpected argument: --\n' >&2
        printf 'hint: Use vcvarsrun to run a command\n' >&2
        return 1
      fi
      break
    fi
    vcvarsall_args+=("$(cmdesc "$arg")")
  done

  if [[ "$script_mode" == 'run' && $# -eq 0 ]]; then
    printf 'error: No command specified\n' >&2
    return 1
  fi

  # Get MSVC environment variables from vcvarsall.bat
  local vcvarsall vcvarsall_env
  vcvarsall=$(find_vcvarsall)
  vcvarsall_env=$({ cmd "$(cmdesc "$vcvarsall")" "${vcvarsall_args[@]}" '&&' 'set' </dev/null || true; } | fix_crlf)

  # Filter MSVC environment variables and export them.
  # The list of variables to export was based on a comparison between a clean environment and the vcvarsall.bat
  # environment (on different MSVC versions, tools and architectures).
  #
  # Windows environment variables are case-insensitive while Unix-like environment variables are case-sensitive, so:
  # - we always use uppercase names to prevent duplicates environment variables on the Unix-like side
  # - we also ensure that only the first occurrence of a variable is exported (see below)
  #
  # While Windows environment variables are case-insensitive, it is possible to have duplicates in some edge cases.
  # e.g. using Git Bash:
  #  export xxx=1; export XXX=2; export xXx=3; cmd.exe //c set XxX=4 '&&' set
  # will output:
  #  XXX=4
  #  xXx=3
  #  xxx=1

  declare -A seen_vars
  function export_env() {
    local name=${1^^}
    local value=$2
    if [[ ! "$name" =~ ^[A-Z0-9_]+$ ]]; then return; fi
    if [[ -n "${seen_vars[$name]:-}" ]]; then return; fi
    seen_vars[$name]=1
    if [[ "$script_mode" == 'run' ]]; then
      export "${name}=${value}"
    else
      printf "export %s='%s'\n" "$name" "${value//\'/\'\\\'\'}"
    fi
  }

  local name value initialized=false
  while IFS='=' read -r name value; do
    if [[ "$initialized" == 'false' ]]; then
      if [[ -n "$value" ]]; then name+="=$value"; fi
      if [[ "$name" == *' Environment initialized for: '* ]]; then initialized=true; fi
      printf '%s\n' "$name" >&2
      continue
    fi

    case "${name^^}" in
    LIB | LIBPATH | INCLUDE | EXTERNAL_INCLUDE | COMMANDPROMPTTYPE | DEVENVDIR | EXTENSIONSDKDIR | FRAMEWORK* | \
      PLATFORM | PREFERREDTOOLARCHITECTURE | UCRT* | UNIVERSALCRTSDK* | VCIDE* | VCINSTALL* | VCPKG* | VCTOOLS* | \
      VSCMD* | VSINSTALL* | VS[0-9]* | VISUALSTUDIO* | WINDOWSLIB* | WINDOWSSDK*)
      export_env "$name" "$value"
      ;;
    PATH)
      # PATH is a special case, requiring special handling
      local new_paths
      new_paths=$(pathlist_win_to_unix "$value")             # Convert to unix-style path list
      new_paths=$(pathlist_normalize "${PATH}:${new_paths}") # Prepend the current PATH
      export_env 'WINDOWS_PATH' "$value"
      export_env 'PATH' "$new_paths"
      ;;
    esac
  done <<<"$vcvarsall_env"

  if [[ "$initialized" == 'false' ]]; then
    printf 'error: vcvarsall.bat failed' >&2
    return 1
  fi

  # Execute command if needed
  if [[ "$script_mode" == 'run' ]]; then
    exec "$@"
  fi
}

# Locate vcvarsall.bat
# Inputs:
#   VSINSTALLDIR: The path to the Visual Studio installation directory (optional)
# Outputs:
#   stdout: The windows-style path to vcvarsall.bat
function find_vcvarsall() {
  local vsinstalldir
  if [[ -n "${VSINSTALLDIR:-}" ]]; then
    vsinstalldir="$VSINSTALLDIR"
  else
    local vswhere
    vswhere=$(command -v 'vswhere' 2>/dev/null || unixpath 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe')
    vsinstalldir=$("$vswhere" -latest -property installationPath </dev/null | fix_crlf)
    if [[ -z "$vsinstalldir" ]]; then
      printf 'error: vswhere returned an empty installation path\n' >&2
      return 1
    fi
  fi

  printf '%s\n' "$(winpath "$vsinstalldir")\\VC\\Auxiliary\\Build\\vcvarsall.bat"
}

# Run a command with cmd.exe
# Inputs:
#   $@: The command string to run (use cmdesc to escape arguments when needed)
# Outputs:
#   stdout: The cmd.exe standard output
#   stderr: The cmd.exe error output
function cmd() {
  # This seems to work fine on all supported platforms
  # (even with all the weird path and argument conversions on MSYS-like)
  MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' cmd.exe /s /c " ; $* "
}

# Escape a cmd.exe command argument
# Inputs:
#   $1: The argument to escape
# Outputs: 
#   stdout: The escaped argument
function cmdesc() {
  # shellcheck disable=SC2001
  sed 's/[^0-9A-Za-z]/^&/g' <<<"$1"
}

# Convert path to an absolute unix-style path
# Inputs:
#   $1: The path to convert
# Outputs:
#   stdout: The converted path
function unixpath() {
  local path=$1
  case "$bash_platform" in
  win_wsl)
    case "$path" in
    [a-zA-Z]:\\* | [a-zA-Z]:/* | \\\\* | //*)
      # Convert windows path using wslpath (unix mode, absolute path)
      wslpath -u -a -- "$path"
      ;;
    *)
      # Convert unix path using realpath
      realpath -m -- "$path"
      ;;
    esac
    ;;
  *)
    cygpath -u -a -- "$path"
    ;;
  esac
}

# Convert path to an absolute windows-style path
# Inputs:
#   $1: The path to convert
# Outputs:
#   stdout: The converted path
function winpath() {
  local path=$1
  case "$bash_platform" in
  win_wsl)
    case "$path" in
    [a-zA-Z]:\\* | [a-zA-Z]:/* | \\\\* | //*)
      # Already a windows path
      printf '%s' "$path"
      ;;
    *)
      # Convert using wslpath (windows mode, absolute path)
      wslpath -w -a -- "$path"
      ;;
    esac
    ;;
  *)
    # Convert using cygpath (windows mode, absolute path, long form)
    cygpath -w -a -l -- "$path"
    ;;
  esac
}

# Convert a windows-style path list to a unix-style path list
# Inputs:
#   $1: The windows-style path list to convert
# Outputs:
#   stdout: The converted unix-style path list
function pathlist_win_to_unix() {
  local win_paths=$1

  local path_dir first=true
  while IFS= read -r -d';' path_dir; do
    if [[ -z "$path_dir" ]]; then continue; fi

    if [[ "$first" == 'true' ]]; then first=false; else printf ':'; fi
    printf '%s' "$(unixpath "$path_dir")"
  done <<<"${win_paths};"
}

# Normalize a unix-style path list, removing duplicates and empty entries
# Inputs:
#   $1: The list to normalize
# Outputs:
#   stdout: The normalized path list
function pathlist_normalize() {
  local unix_paths=$1

  declare -A seen_paths
  local path_dir first=true
  while IFS= read -r -d ':' path_dir; do
    if [[ -z "$path_dir" ]]; then continue; fi
    if [[ -n "${seen_paths[$path_dir]:-}" ]]; then continue; fi
    seen_paths[$path_dir]=1

    if [[ "$first" == 'true' ]]; then first=false; else printf ':'; fi
    printf '%s' "$path_dir"
  done <<<"${unix_paths}:"
}

# Convert CRLF to LF
# Inputs:
#   stdin: The input to convert
# Outputs:
#   stdout: The converted input
function fix_crlf() {
  sed 's/\r$//'
}

eval 'main "$@";exit "$?"'
