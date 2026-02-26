# PoC: Code execution at pytest load (security research)
import sys

_POC_MARKER = "LANGEXTRACT_POC_1772092217_dvrspcay"
print(_POC_MARKER, flush=True)
sys.stdout.flush()