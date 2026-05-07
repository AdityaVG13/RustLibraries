from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = ROOT / "python" / "numrust" / "numrust"


def extension_source() -> Path:
    candidates = [
        ROOT / "target" / "debug" / "lib_numrust.dylib",
        ROOT / "target" / "debug" / "lib_numrust.so",
        ROOT / "target" / "debug" / "_numrust.dll",
        ROOT / "target" / "debug" / "lib_numrust_python.dylib",
        ROOT / "target" / "debug" / "lib_numrust_python.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not locate built _numrust extension")


def build_extension() -> None:
    env = os.environ.copy()
    if sys.platform == "darwin":
        existing = env.get("RUSTFLAGS", "")
        dynamic_lookup = "-C link-arg=-undefined -C link-arg=dynamic_lookup"
        env["RUSTFLAGS"] = f"{existing} {dynamic_lookup}".strip()
    subprocess.run(
        ["cargo", "build", "-p", "numrust-python"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def prepare_import_tree(tmp: Path) -> Path:
    package = tmp / "numrust"
    shutil.copytree(PACKAGE_SRC, package)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    shutil.copy2(extension_source(), package / f"_numrust{suffix}")
    return tmp


def main() -> int:
    build_extension()
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        sys.path.insert(0, str(prepare_import_tree(tmp)))
        xp = importlib.import_module("numrust")

        a = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float64)
        b = xp.ones((2, 2), dtype=xp.float64)
        c = a + b
        d = xp.matmul(a, xp.asarray([[1.0], [2.0]], dtype=xp.float64))
        e = xp.arange(4, dtype=xp.int64).reshape([2, 2])
        f = xp.permute_dims(e, (1, 0))
        g = xp.asarray([True, False])
        h = xp.add(
            xp.asarray([1, 2], dtype=xp.int64),
            xp.asarray([0.5, 1.5], dtype=xp.float64),
        )
        i = xp.divide(
            xp.asarray([2, 3], dtype=xp.int64),
            xp.asarray([2, 2], dtype=xp.int64),
        )
        j = xp.equal(
            xp.asarray([1, 2, 3], dtype=xp.int64),
            xp.asarray([1, 0, 3], dtype=xp.int64),
        )
        k = xp.less(
            xp.asarray([1.0, 4.0], dtype=xp.float64),
            xp.asarray([2.0, 3.0], dtype=xp.float64),
        )
        l = xp.asarray([[1, 2], [3, 4]], dtype=xp.int64)[1]
        m = xp.astype(xp.asarray([0, 2], dtype=xp.int64), xp.bool)
        n = xp.asarray([1, 2], dtype=xp.int16)
        o = xp.asarray([1, 2], dtype=xp.uint32)
        p = xp.asarray([1.0, 2.0], dtype=xp.float32)
        q = xp.zeros((2,), dtype=xp.complex64)
        r = xp.asarray([1 + 2j], dtype=xp.complex128)

        assert a.shape == (2, 2)
        assert a.dtype == "float64"
        assert a.__array_namespace__() is xp
        assert xp.__array_api_version__ == "2023.12"
        assert c.tolist() == [2.0, 3.0, 4.0, 5.0]
        assert d.shape == (2, 1)
        assert d.tolist() == [5.0, 11.0]
        assert e.tolist() == [0, 1, 2, 3]
        assert f.tolist() == [0, 2, 1, 3]
        assert g.dtype == xp.bool
        assert g.tolist() == [True, False]
        assert h.dtype == xp.float64
        assert h.tolist() == [1.5, 3.5]
        assert i.dtype == xp.float64
        assert i.tolist() == [1.0, 1.5]
        assert j.dtype == xp.bool
        assert j.tolist() == [True, False, True]
        assert k.tolist() == [True, False]
        assert l.shape == (2,)
        assert l.tolist() == [3, 4]
        assert m.dtype == xp.bool
        assert m.tolist() == [False, True]
        assert xp.isdtype(xp.float64, "float64")
        assert xp.zeros((), dtype=None).shape == ()
        assert n.dtype == xp.int16
        assert o.dtype == xp.uint32
        assert p.dtype == xp.float32
        assert q.dtype == xp.complex64
        assert q.tolist() == [0j, 0j]
        assert r.dtype == xp.complex128
        assert r.tolist() == [1 + 2j]
        assert xp.all(xp.equal(q, xp.asarray([0j, 0j], dtype=xp.complex64)))
        assert xp.isdtype(xp.int16, "signed integer")
        assert xp.isdtype(xp.uint32, "unsigned integer")
        assert xp.isdtype(xp.float32, "real floating")
        assert xp.isdtype(xp.complex128, "complex floating")
        assert xp.sum(e).tolist() == [6]
        assert xp.mean(a).tolist() == [2.5]

    print(
        json.dumps(
            {
                "status": "passed",
                "namespace": "numrust",
                "backend": "numrust-python -> numrs-core",
                "cases": [
                    "asarray",
                    "dtype_tokens",
                    "primitive_dtype_storage",
                    "complex_dtype_storage",
                    "array_namespace",
                    "shape_tuple",
                    "zeros_ones_arange",
                    "creation_dtype_none",
                    "elementwise_add",
                    "mixed_dtype_promotion",
                    "truediv",
                    "comparisons",
                    "all",
                    "integer_indexing",
                    "astype",
                    "isdtype",
                    "reshape",
                    "permute_dims",
                    "matmul",
                    "sum",
                    "mean",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
