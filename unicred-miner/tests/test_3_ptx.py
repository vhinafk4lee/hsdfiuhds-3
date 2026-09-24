"""Test 3: the embedded PTX is the NVRTC build of kernel.cu and passes ptxas without spills.

ptxas / NVRTC are taken from tools/.cache (python tools/build_ptx.py --download),
$PTXAS / $NVRTC_LIB or PATH; tests that need them are skipped otherwise.
"""
import os
import subprocess
import tempfile
import unittest

from tests.helpers import ROOT, load_worker

import importlib.util

_spec = importlib.util.spec_from_file_location("build_ptx", str(ROOT / "tools" / "build_ptx.py"))
build_ptx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_ptx)

SMS = ["52", "61", "70", "75", "80", "86", "89", "90"]


class PtxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ptx = (ROOT / "worker" / "kernel.ptx").read_text()
        cls.w = load_worker()

    def test_embedded_equals_kernel_ptx(self):
        self.assertEqual(self.w.embedded_ptx(), self.ptx)

    def test_header(self):
        self.assertRegex(self.ptx, r"(?m)^\.version 7\.8$")
        self.assertRegex(self.ptx, r"(?m)^\.target sm_52$")
        self.assertIn(".visible .entry unicred_search(", self.ptx)
        self.assertIn(".visible .entry unicred_hash(", self.ptx)
        self.assertIn("shf.l.wrap.b32", self.ptx)  # __funnelshift_l rotations
        self.assertIn("prmt.b32", self.ptx)        # byte swaps

    def _ptxas(self):
        p = build_ptx.find_ptxas()
        if not p or not os.path.exists(p):
            self.skipTest("ptxas not found (python tools/build_ptx.py --download)")
        return p

    def test_ptxas_no_spill(self):
        ptxas = self._ptxas()
        rows = build_ptx.run_ptxas(ptxas, ROOT / "worker" / "kernel.ptx", SMS)
        for sm, regs, st, ld, stack in rows:
            self.assertEqual((st, ld, stack), (0, 0, 0), "sm_%s spills" % sm)
            self.assertLessEqual(regs, 128, "sm_%s uses %d registers" % (sm, regs))

    def test_lower_isa_fallback_assembles(self):
        """Old drivers: the worker retries with .version 7.4 / 7.0 / 6.4."""
        ptxas = self._ptxas()
        for v in ("7.4", "7.0", "6.4"):
            text = self.w.lower_ptx_version(self.ptx, v)
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "k.ptx")
                with open(src, "w") as fh:
                    fh.write(text)
                res = subprocess.run([ptxas, "-arch=sm_61", src, "-o", os.path.join(tmp, "k.cubin")],
                                     capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, res.stderr)

    def test_ptx_is_up_to_date(self):
        nvrtc = build_ptx.find_nvrtc(None)
        if not os.path.exists(nvrtc):
            self.skipTest("NVRTC not found (python tools/build_ptx.py --download)")
        fresh = build_ptx.compile_ptx(nvrtc, "compute_52")
        self.assertEqual(fresh, self.ptx, "kernel.ptx is stale: run tools/build_ptx.py")


if __name__ == "__main__":
    unittest.main()
