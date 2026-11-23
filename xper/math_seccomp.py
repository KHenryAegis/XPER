"""Opt-in computation-only filter for model Python children on the x86-64 host.

Read-only file access is allowed. Filesystem mutation, subprocesses, signals,
network creation and privileged host operations are denied. Parent-side timeout
handling is outside this filter and signals only the child it just forked.
"""
import ctypes
import errno
import os


def install():
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    class Compare(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                    ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint, ctypes.POINTER(Compare)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO
    def rule(name, comparisons=()):
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            return  # syscall absent on this architecture
        arr = (Compare * len(comparisons))(*comparisons)
        if lib.seccomp_rule_add_array(ctx, deny, number, len(arr), arr) < 0:
            raise RuntimeError(f"seccomp rule failed: {name}")
    try:
        for name in ("creat openat2 truncate ftruncate unlink unlinkat rename renameat renameat2 "
                     "mkdir mkdirat rmdir link linkat symlink symlinkat mknod mknodat "
                     "chmod fchmod fchmodat chown fchown lchown fchownat utime utimes utimensat futimesat "
                     "setxattr lsetxattr fsetxattr removexattr lremovexattr fremovexattr "
                     "kill tkill tgkill pidfd_send_signal pidfd_getfd ptrace process_vm_readv process_vm_writev "
                     "fork vfork clone clone3 execve execveat socket socketpair connect "
                     "mount umount2 pivot_root chroot unshare setns open_by_handle_at "
                     "ioctl io_uring_setup io_uring_enter io_uring_register bpf userfaultfd "
                     "setpriority sched_setscheduler sched_setparam sched_setattr sched_setaffinity prlimit64 "
                     "ioperm iopl quotactl syslog fanotify_init fanotify_mark keyctl add_key request_key "
                     "reboot swapon swapoff acct settimeofday clock_settime adjtimex "
                     "init_module finit_module delete_module kexec_load kexec_file_load "
                     "fsopen fsconfig fsmount fspick open_tree move_mount mount_setattr").split():
            rule(name)
        for syscall, arg in (("open", 1), ("openat", 2)):
            for bit in (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC, os.O_APPEND):
                rule(syscall, [Compare(arg, 7, bit, bit)])  # SCMP_CMP_MASKED_EQ
        if lib.seccomp_load(ctx) < 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        lib.seccomp_release(ctx)
