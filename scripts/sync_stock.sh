#!/usr/bin/env bash
# VM(awarenet, KOREN) 이 모으고 있는 타슈 재고 스냅샷을 로컬로 가져온다.
#   사용법:  bash scripts/sync_stock.sh
# 가져온 뒤:  python src/build_stock_panel.py
set -euo pipefail

VM_HOST=${VM_HOST:-116.89.187.190}
VM_PORT=${VM_PORT:-26022}
VM_USER=${VM_USER:-ubuntu}
REMOTE_DIR=${REMOTE_DIR:-tashu_collect}
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw/stock"

mkdir -p "$LOCAL_DIR"

echo "== VM 수집 상태"
ssh -o BatchMode=yes -p "$VM_PORT" "$VM_USER@$VM_HOST" \
    "cd ~/$REMOTE_DIR && python3 collect_stock.py --check" || true

echo "== 내려받기 -> $LOCAL_DIR"
if command -v rsync >/dev/null 2>&1; then
  rsync -az --info=stats1 -e "ssh -p $VM_PORT" \
    "$VM_USER@$VM_HOST:~/$REMOTE_DIR/data/" "$LOCAL_DIR/"
else
  scp -q -P "$VM_PORT" "$VM_USER@$VM_HOST:~/$REMOTE_DIR/data/stock/*.csv.gz" "$LOCAL_DIR/" || true
  scp -q -P "$VM_PORT" "$VM_USER@$VM_HOST:~/$REMOTE_DIR/data/stations_log.csv" "$LOCAL_DIR/" || true
fi

ls -la "$LOCAL_DIR" | tail -5
echo "== 다음: python src/build_stock_panel.py"
