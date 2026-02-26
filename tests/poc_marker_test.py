"""
PoC test - security research. Proves untrusted PR code executes in CI.
"""
import sys


def test_poc_marker():
    marker = "LANGEXTRACT_POC_1772092217_dvrspcay"
    print(marker, flush=True)
    sys.stdout.flush()
    assert True
