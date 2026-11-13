#!/bin/bash
# Every arm, one after another (case 3 arms share one NVMe, so never run them in parallel).
D=$(dirname "$0")
"$D/case3.sh" eager formal_c88
"$D/case3.sh" lazy formal_c88
"$D/case1.sh" lmcache
"$D/case2.sh" lmcache
"$D/case1.sh" native
"$D/case2.sh" native
