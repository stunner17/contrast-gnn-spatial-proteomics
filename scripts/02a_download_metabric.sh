#!/usr/bin/env bash
# Download METABRIC IMC processed data (Danenberg et al. 2022, Nat Genet), Zenodo record
# 7324285 = v2.1.0, the latest processed-data release (CC BY 4.0). Resumable; verifies md5.
set -euo pipefail
cd "$(dirname "$0")/../data"
mkdir -p metabric && cd metabric
URL="https://zenodo.org/api/records/7324285/files/MBTMEStrPublicProcessedDataCode.zip/content"
F="MBTMEStrPublicProcessedDataCode.zip"
MD5=$(curl -s https://zenodo.org/api/records/7324285 | grep -o '"checksum": *"md5:[0-9a-f]*"' | head -1 | grep -o '[0-9a-f]\{32\}')
curl -L -C - -o "$F" "$URL"
echo "expected md5 $MD5"
GOT=$(md5sum "$F" | cut -d' ' -f1); echo "got      md5 $GOT"
[ "$GOT" = "$MD5" ] || { echo "MD5 MISMATCH"; exit 1; }
unzip -o -q "$F" "MBTMEIMCPublic/SingleCells.csv" "MBTMEIMCPublic/CellNeighbours.csv" \
  "MBTMEIMCPublic/IMCClinical.csv" "MBTMEIMCPublic/AbPanel.csv" "MBTMEIMCPublic/DataAnnotation.pdf" \
  "MBTMEIMCPublic/SingleCellsAnnotation.xlsx" "MBTMEIMCPublic/CellNeighboursAnnotation.xlsx"
ls -la MBTMEIMCPublic
echo DONE
