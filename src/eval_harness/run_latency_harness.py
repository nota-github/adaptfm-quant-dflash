#!/usr/bin/env python3
"""Thin wrapper: run run_eval_local.py's EXACT latency eval against a local
container, stubbing the (unused-for-latency) lm_eval import so it loads without
lm_eval installed. All latency logic/config comes from run_eval_local.py."""
import sys, types, os

# stub lm_eval.api.model.LM (top-level import in run_eval_local.py; only quality uses it)
for name in ('lm_eval', 'lm_eval.api', 'lm_eval.api.model'):
    sys.modules.setdefault(name, types.ModuleType(name))
class _LM:  # noqa
    pass
sys.modules['lm_eval.api.model'].LM = _LM

os.environ['EVAL_MODE'] = 'latency'
os.environ.setdefault('CONTAINER_URL', 'http://localhost:18096')

import runpy
runpy.run_path(os.path.join(os.path.dirname(__file__), 'run_eval_local.py'), run_name='__main__')
