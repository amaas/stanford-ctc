import time
from subprocess import check_call
from run_cfg import RUN_DIR, VIEWER_DIR

WAIT_INTERVAL_SECS = 10*60

while True:
    print 'Generating runs page'
    check_call('python browse_runs.py --run_dir %s --viewer_dir %s --figs' % (RUN_DIR, VIEWER_DIR), shell=True)
    time.sleep(WAIT_INTERVAL_SECS)
