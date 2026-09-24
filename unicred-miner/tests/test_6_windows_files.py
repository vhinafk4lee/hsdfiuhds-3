"""Files written on Windows: Notepad (UTF-8 BOM) and PowerShell 5.1 `>` (UTF-16 LE)."""
import json
import tempfile
import unittest
from pathlib import Path

from eth_account import Account

from unicred.config import load_config
from unicred.servers import load_servers
from unicred.signer import Wallet

LINES = ("ssh -p 31618 root@151.237.25.16 -L 8080:localhost:8080\r\n"
         "ssh -p 40017 root@14.234.172.15 -L 8080:localhost:8080\r\n")
ENCODINGS = {"utf8": lambda t: t.encode("utf-8"),
             "utf8-bom": lambda t: b"\xef\xbb\xbf" + t.encode("utf-8"),
             "utf16-le": lambda t: t.encode("utf-16")}


class WindowsFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_servers_txt(self):
        for name, enc in ENCODINGS.items():
            f = self.dir / ("servers-%s.txt" % name)
            f.write_bytes(enc(LINES))
            specs = load_servers(f)
            self.assertEqual([s.name for s in specs], ["151.237.25.16:31618", "14.234.172.15:40017"], name)
            self.assertEqual(specs[0].user, "root")

    def test_config_json(self):
        for name, enc in ENCODINGS.items():
            f = self.dir / ("config-%s.json" % name)
            f.write_bytes(enc(json.dumps({"max_mints": 7})))
            self.assertEqual(load_config(str(f)).max_mints, 7, name)

    def test_wallet_key(self):
        acct = Account.create()
        key = "0x" + acct.key.hex().replace("0x", "")
        for name, enc in ENCODINGS.items():
            f = self.dir / ("wallet-%s.key" % name)
            f.write_bytes(enc(key + "\r\n"))
            self.assertEqual(Wallet(f).address, acct.address, name)


class EncryptedSshKeyTest(unittest.TestCase):
    """A passphrase-protected SSH key: the passphrase is asked once and passed to paramiko."""

    def test_passphrase_prompt(self):
        import paramiko
        from unicred.servers import parse_server_line, prepare_ssh_passphrase
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "id_test"
            paramiko.RSAKey.generate(2048).write_private_key_file(str(key), password="s3cret")
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(json.dumps({"ssh_key": str(key)}))
            cfg = load_config(str(cfg_path))
            specs = [parse_server_line("ssh -p 31618 root@151.237.25.16 -L 8080:localhost:8080")]
            answers = iter(["wrong", "s3cret"])
            prepare_ssh_passphrase(cfg, specs, ask=lambda prompt: next(answers))
            self.assertEqual(cfg["ssh_key_passphrase"], "s3cret")
            self.assertEqual(cfg.public_view()["ssh_key_passphrase"], "***")
            # unencrypted key: no prompt
            key.unlink()
            paramiko.RSAKey.generate(2048).write_private_key_file(str(key))
            cfg2 = load_config(str(cfg_path))
            prepare_ssh_passphrase(cfg2, specs, ask=lambda prompt: self.fail("asked"))
            self.assertEqual(cfg2["ssh_key_passphrase"], "")


class BatchFilesTest(unittest.TestCase):
    """Double-click launchers: CRLF, ASCII, run from their own folder, call existing commands."""

    def test_launchers(self):
        root = Path(__file__).resolve().parent.parent
        expected = {"setup.bat": "unicred.py check", "check.bat": "unicred.py check",
                    "servers.bat": "unicred.py servers", "dry-run.bat": "unicred.py run --dry-run",
                    "run.bat": "unicred.py run\r\n"}
        for name, cmd in expected.items():
            data = (root / name).read_bytes()
            data.decode("ascii")
            self.assertNotIn(b"\n", data.replace(b"\r\n", b""), name + " must use CRLF")
            self.assertIn(b'cd /d "%~dp0"', data, name)
            self.assertIn(cmd.encode(), data, name)
            self.assertIn(b":nopython", data, name)


if __name__ == "__main__":
    unittest.main()
