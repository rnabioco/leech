#!/bin/bash
# Quick diagnostic to check read counts in POD5 vs BAM

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <bam_file> <pod5_file>"
    exit 1
fi

BAM_FILE="$1"
POD5_FILE="$2"

echo "=========================================="
echo "POD5/BAM Read Count Comparison"
echo "=========================================="
echo ""
echo "BAM file:  $BAM_FILE"
echo "POD5 file: $POD5_FILE"
echo ""

# Count reads in BAM (primary alignments only)
echo "Counting BAM reads (primary alignments)..."
BAM_COUNT=$(samtools view -F 0x900 -c "$BAM_FILE")
echo "  BAM reads: $BAM_COUNT"

# Count reads in POD5
echo ""
echo "Counting POD5 reads..."
POD5_COUNT=$(pod5 view "$POD5_FILE" --include "read_id" --output /dev/stdout | tail -n +2 | wc -l)
echo "  POD5 reads: $POD5_COUNT"

# Calculate difference
echo ""
echo "=========================================="
DIFF=$((BAM_COUNT - POD5_COUNT))
if [ $DIFF -eq 0 ]; then
    echo "✓ COUNTS MATCH: $BAM_COUNT reads in both files"
elif [ $DIFF -gt 0 ]; then
    echo "❌ PROBLEM: BAM has $DIFF MORE reads than POD5"
    echo "   This should be impossible if pipeline is correct!"
    echo ""
    echo "   Possible causes:"
    echo "   - BAM was created from different POD5 files"
    echo "   - POD5 merge is incomplete"
    echo "   - Wrong POD5 file provided"
else
    DIFF=$((POD5_COUNT - BAM_COUNT))
    echo "⚠️  POD5 has $DIFF MORE reads than BAM"
    echo "   This is OK - some reads failed alignment/QC"
fi
echo "=========================================="
