#!/usr/bin/env bash
# experiment.sh — FC-GPU AMD workflow (mm + stencil)

set -e  # exit on any error

# ----------------------------------------
#  CLEANUP
# ----------------------------------------
cleanup_shared_memory(){
  echo "[INFO] Cleaning shared memory segments..."
  for seg in $(ipcs -m | awk 'NR>3{print $2}'); do
    ipcrm -m "$seg" 2>/dev/null || true
  done
}
clear_logs(){
  echo "[INFO] Clearing logs..."
  rm -rf logs && mkdir -p logs
}
kill_processes(){
  echo "[INFO] Killing stray processes..."
  pkill -f './t1'        || true
  pkill -f './t2'        || true
  pkill -f './server'    || true
}

# ----------------------------------------
#  HIPIFY + COMPILE
# ----------------------------------------
hipify_and_compile_tasks(){
  echo "[INFO] Converting & compiling CUDA→HIP tasks..."
  rm -f t1 t2
  local idx=1

  for task in "$@"; do
    local srccu="${task}.cu"
    local srccpp="${task}.cpp"
    local hipcpp="${task}.cpp"
    local bin="t${idx}"
    local define="-DT${idx}"

    # 1) Generate HIP source (fallback to .cpp if no .cu)
    if [[ -f "$srccu" ]]; then
      echo "  [HIPIFY] $srccu → $hipcpp"
      hipify-perl "$srccu" > "$hipcpp"
    elif [[ -f "$srccpp" ]]; then
      echo "  [COPY ] $srccpp → $hipcpp"
      cp "$srccpp" "$hipcpp"
    else
      echo "[ERROR] Neither $srccu nor $srccpp exists!"; exit 1
    fi

    # 2) Patch casts & wrap HIP_CALLs
    echo "  [PATCH ] Applying cast & wrap to $hipcpp"
    ./patch_casts.pl "$hipcpp"

    # 3) Ensure main()
    echo "  [CHECK ] Looking for main() in $hipcpp"
    grep -q 'int main' "$hipcpp" \
      || { echo "[ERROR] No main() in $hipcpp"; exit 1; }

    # 4) Compile with hipcc +DT1/DT2
    echo "  [COMPILE] hipcc $define $hipcpp -o $bin -lpthread -lrt"
    hipcc $define "$hipcpp" -o "$bin" -lpthread -lrt \
      && echo "  [BUILT ] $bin"
    ((idx++))
  done
}

# ----------------------------------------
#  MAIN
# ----------------------------------------
cleanup_shared_memory
clear_logs
kill_processes
chmod +x patch_casts.pl

echo "[INFO] Building controller/server..."
g++ -std=c++17 -pthread -lrt server.cpp -o server

echo "[INFO] Building GPU tasks (mm + stencil)..."
hipify_and_compile_tasks mm stencil

# ----------------------------------------
#  RUN
# ----------------------------------------
periods=(10 15)         # in ms
setpoints=(0.9 0.9)
duration=200            # in seconds

# launch T1 → wait 1s
# echo "[LAUNCH] t1 (${setpoints[0]}, ${periods[0]})  $duration s"
# ./t1 "${setpoints[0]}" "${periods[0]}" "$duration" & t1_pid=$!

# # launch T2 → wait 2s
# echo "[LAUNCH] t2 (${setpoints[1]}, ${periods[1]}) $duration s"
# ./t2 "${setpoints[1]}" "${periods[1]}" "$duration" & t2_pid=$!
# sleep 2

# # launch controller
# echo "[LAUNCH] server (2, $t1_pid, $t2_pid, ${setpoints[0]}, ${setpoints[1]})"
# ./server 2 "$t1_pid" "$t2_pid" "${setpoints[0]}" "${setpoints[1]}" & ctrl_pid=$!

# # let it run
# echo "[INFO] Running for $duration seconds..."
# sleep "$duration"

# # teardown
# echo "[INFO] Duration reached – terminating..."
# kill "$t1_pid" "$t2_pid" "$ctrl_pid" 2>/dev/null || true
# wait "$t1_pid" "$t2_pid" "$ctrl_pid" 2>/dev/null || true

# echo "[DONE] Experiment finished."
args=(2 t1 t2 "${setpoints[0]}" "${periods[0]}" "${setpoints[1]}" "${periods[1]}" "$duration")
python3 runit.py "${args[@]}"
if [[ $? -ne 0 ]]; then
    echo "Error: runit.py execution failed."
    exit 1
fi
