import resource
# 在 swift/transformers import 之前, 进程内直接把 fd 上限抬到硬上限,
# 不依赖 shell 的 ulimit 传递(实测脚本里 shell->python 传不过去)
_s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_h, _h))
print("[run_grpo] RLIMIT_NOFILE %s -> %s" % (_s, resource.getrlimit(resource.RLIMIT_NOFILE)[0]), flush=True)

import sys, runpy
# 等价于: python -m swift.cli.main <args>
sys.argv = ["swift"] + sys.argv[1:]
runpy.run_module("swift.cli.main", run_name="__main__")
