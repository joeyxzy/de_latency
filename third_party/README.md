# Third-party material

- `libbpf/include/bpf/`: the existing vendored libbpf headers, relocated without changing their contents or copyright notices.
- `libbpf/lib/`: the existing `libbpf.so.1.3.0` binary and `libbpf.so` symlink, plus the `libbpf.so.1` SONAME symlink required by the runtime loader. The native build prefers a system libbpf and falls back to this copy; the monitor's RUNPATH also includes this directory.
- `linux/vmlinux.h`: the historical BTF-generated kernel header, preserved but **not used by the current build**.

These files are not original project code or disposable build output. Their upstream origin, version, and applicable license texts must be verified before public redistribution. Relocating them does not establish permission to redistribute them, and no project-wide license is assigned here.
