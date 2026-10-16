#!/usr/bin/env bash
# Fetch the pinned Nemotron 3.5 Super VL checkpoint and, for SPEC=lilicorr, its drafter into the Hugging Face cache.
# hf download resumes and skips files that are already complete, so a later start is a quick no-op.
set -euo pipefail

hf download "$MODEL_ID" --revision "$MODEL_REVISION"
if [[ ${SPEC:-lilicorr} == lilicorr ]]; then
  hf download "$DRAFT_ID" --revision "$DRAFT_REVISION"
fi
exit 0
