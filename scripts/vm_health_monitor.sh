#!/bin/bash
# VM Health Monitor — runs in background, sends Slack updates every 5 min.
# Launched by the GCE startup script. Reads SLACK_WEBHOOK_URL from env.

WEBHOOK_URL="${SLACK_WEBHOOK_URL}"
INTERVAL=300
BUCKET="f1-predictor-artifacts-jowin"
STAGING="staging/training-run"

slack_notify() {
    if [ -z "$WEBHOOK_URL" ]; then return; fi
    curl -s -X POST "$WEBHOOK_URL" \
        -H 'Content-Type: application/json' \
        -d "{\"text\": \"$1\"}" 2>/dev/null || true
}

while true; do
    # GPU metrics
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    if [ "$GPU_INFO" != "N/A" ]; then
        GPU_UTIL=$(echo "$GPU_INFO" | awk -F', ' '{print $1}')
        GPU_MEM_USED=$(echo "$GPU_INFO" | awk -F', ' '{print $2}')
        GPU_MEM_TOTAL=$(echo "$GPU_INFO" | awk -F', ' '{print $3}')
        GPU_TEMP=$(echo "$GPU_INFO" | awk -F', ' '{print $4}')
        GPU_LINE="GPU: ${GPU_UTIL}% util, ${GPU_MEM_USED}/${GPU_MEM_TOTAL} MiB, ${GPU_TEMP}C"
    else
        GPU_LINE="GPU: N/A"
    fi

    # RAM
    RAM_LINE=$(free -m | awk '/Mem:/ {printf "RAM: %d/%dMB (%.0f%%)", $3, $2, $3/$2*100}')

    # Disk
    DISK_LINE=$(df -h /opt 2>/dev/null | tail -1 | awk '{printf "Disk: %s/%s (%s)", $3, $2, $5}')

    # Current model (detect from running nbconvert processes)
    CURRENT_MODEL="idle"
    for key in g h i; do
        if pgrep -f "05${key}_model_" > /dev/null 2>&1; then
            CURRENT_MODEL="Model ${key^^}"
        fi
    done

    # Last progress line from any model log
    LAST_PROGRESS=""
    for key in g h i; do
        LOG="/var/log/f1-model-${key}-progress.log"
        if [ -f "$LOG" ]; then
            LINE=$(tail -1 "$LOG" 2>/dev/null)
            if [ -n "$LINE" ]; then
                LAST_PROGRESS="$LINE"
            fi
        fi
    done

    # Artifact counts
    G_PARQ=$(ls /opt/f1-training/data/training/model_G_*.parquet 2>/dev/null | wc -l)
    H_PARQ=$(ls /opt/f1-training/data/training/model_H_*.parquet 2>/dev/null | wc -l)
    I_PARQ=$(ls /opt/f1-training/data/training/model_I_*.parquet 2>/dev/null | wc -l)
    G_PKL=$(ls /opt/f1-training/data/raw/model/Model_G_*.pkl 2>/dev/null | wc -l)
    H_PKL=$(ls /opt/f1-training/data/raw/model/Model_H_*.pkl 2>/dev/null | wc -l)
    I_PKL=$(ls /opt/f1-training/data/raw/model/Model_I_*.pkl 2>/dev/null | wc -l)

    # DONE signals
    DONE_STATUS=""
    for key in G H I; do
        SIGNAL=$(gsutil cat "gs://${BUCKET}/${STAGING}/MODEL_${key}_DONE" 2>/dev/null || echo "")
        if [ -n "$SIGNAL" ]; then
            DONE_STATUS="${DONE_STATUS} ${key}=${SIGNAL}"
        fi
    done

    # Build message
    TS=$(date -u +"%H:%M UTC")
    MSG=":heartbeat: *VM Health — ${TS}*\n"
    MSG+="${GPU_LINE}\n"
    MSG+="${RAM_LINE}\n"
    MSG+="${DISK_LINE}\n"
    MSG+="Training: ${CURRENT_MODEL}\n"
    MSG+="Artifacts: G=${G_PARQ}p/${G_PKL}m  H=${H_PARQ}p/${H_PKL}m  I=${I_PARQ}p/${I_PKL}m"
    if [ -n "$DONE_STATUS" ]; then
        MSG+="\nDone:${DONE_STATUS}"
    fi
    if [ -n "$LAST_PROGRESS" ]; then
        MSG+="\nLast: ${LAST_PROGRESS}"
    fi

    slack_notify "$MSG"

    # Also write status to GCS for external polling
    echo -e "${TS}\nGPU: ${GPU_INFO}\n${RAM_LINE}\nTraining: ${CURRENT_MODEL}\nG: parq=${G_PARQ} pkl=${G_PKL}\nH: parq=${H_PARQ} pkl=${H_PKL}\nI: parq=${I_PARQ} pkl=${I_PKL}" \
        | gsutil -q cp - "gs://${BUCKET}/${STAGING}/vm_status.txt" 2>/dev/null || true

    sleep $INTERVAL
done
