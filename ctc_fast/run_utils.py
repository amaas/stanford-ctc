import os
import re
import json
import subprocess
import time
import datetime
from os.path import join as pjoin


def dump_config(cfg, fname):
    json.dump(cfg, open(fname, 'w'), sort_keys=True, indent=4,
            separators=(',', ':'))


def load_config(fname):
    return json.load(open(fname, 'r'))


class CfgStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def add_config_val(key, val, fname):
    with open(fname, 'r') as fin:
        cfg = json.load(fin)
    cfg[key] = val
    dump_config(cfg, fname)


def get_git_revision():
    return subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
            stdout=subprocess.PIPE).communicate()[0]


def get_hostname():
    import socket
    hostname = socket.gethostname()
    hostname = hostname.split('.')[0]
    return hostname


def touch_file(fname):
    try:
        os.utime(fname, None)
    except:
        open(fname, 'a').close()


def last_modified(fname):
    return os.path.getmtime(fname)


def file_alive(fname, max_dur_sec=60*60*2):
    return (int(time.time()) - last_modified(fname)) < max_dur_sec


class TimeString(object):

    def __init__(self, time=datetime.datetime.today()):
        self.time = time

    def __str__(self):
        # NOTE Currently down to the second, e.g. 20130405153042
        s = str(datetime.datetime.today())
        s = s.split('.')[0].replace(' ', '')
        s = s.replace('-', '')
        s = s.replace(':', '')
        return s

    @classmethod
    def match(cls, s):
        return re.match('\d{14}$', s)

    @classmethod
    def from_string(cls, string):
        year = int(string[0:4])
        month = int(string[4:6])
        day = int(string[6:8])
        hour = int(string[8:10])
        minute = int(string[10:12])
        second = int(string[12:14])
        return TimeString(time=datetime.datetime(year, month, day, hour, minute, second))


def get_run_dirs(parent_dir):
    run_dirs = list()
    for d in os.listdir(parent_dir):
        if TimeString.match(d) and not d.endswith('bak'):
            run_dirs.append(pjoin(parent_dir, d))
    return run_dirs
