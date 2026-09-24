"""GPU servers: servers.txt parsing, SSH (paramiko) / local transports, worker connections."""
import os
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER_FILES = [ROOT / "worker" / "worker.py", ROOT / "worker" / "kernel.cu"]

# ssh options that take an argument
_SSH_ARG_OPTS = set("BbcDEeFIiJLlmOoPpQRSWw")


class ServerSpec(object):
    def __init__(self, line, host=None, port=22, user="root", key_file=None, local=False,
                 worker_args=None):
        self.line = line
        self.host = host
        self.port = port
        self.user = user
        self.key_file = key_file
        self.local = local
        self.worker_args = worker_args or []
        self.name = "local" if local else "%s:%d" % (host, port)

    def __repr__(self):
        return "ServerSpec(%s)" % self.name


def parse_server_line(line):
    """Parse a line from servers.txt.

    Accepted forms:
      ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080   (as copied from vast.ai)
      ssh root@1.2.3.4 -p 22 -i ~/.ssh/vast_key
      root@1.2.3.4:22
      local [worker args...]         run the worker on this machine (tests / local GPU)
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    tokens = shlex.split(line, posix=True)
    if tokens[0] == "local":
        return ServerSpec(line, local=True, worker_args=tokens[1:])
    if tokens[0] == "ssh":
        tokens = tokens[1:]
    port, user, key, dest = None, None, None, None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-") and len(tok) >= 2:
            opt = tok[1]
            if opt in _SSH_ARG_OPTS:
                val = tok[2:] if len(tok) > 2 else (tokens[i + 1] if i + 1 < len(tokens) else "")
                i += 1 if len(tok) > 2 else 2
                if opt == "p":
                    port = int(val)
                elif opt == "l":
                    user = val
                elif opt == "i":
                    key = os.path.expanduser(val)
                continue
            i += 1  # flag without argument (-A, -N, -T, ...)
            continue
        if dest is None:
            dest = tok
        i += 1
    if dest is None:
        raise ValueError("не найден хост в строке: %s" % line)
    if dest.startswith("ssh://"):
        dest = dest[6:]
    if "@" in dest:
        u, dest = dest.split("@", 1)
        user = user or u
    if dest.count(":") == 1 and port is None:
        dest, p = dest.split(":")
        port = int(p)
    return ServerSpec(line, host=dest, port=port or 22, user=user or "root", key_file=key)


def load_servers(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit("нет файла %s: вставьте туда SSH-строки из vast.ai (см. servers.example.txt)" % path)
    specs, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            spec = parse_server_line(line)
        except ValueError as exc:
            raise SystemExit("%s:%d: %s" % (path, n, exc))
        if spec and spec.name not in seen:
            seen.add(spec.name)
            specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------
class Stream(object):
    """A running remote/local process with line-oriented stdin/stdout."""

    def write(self, data):
        raise NotImplementedError

    def lines(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class SSHTransport(object):
    def __init__(self, spec, cfg):
        self.spec = spec
        self.cfg = cfg
        self.client = None

    def connect(self):
        import paramiko
        client = paramiko.SSHClient()
        known = self.cfg.runtime / "known_hosts"
        try:
            if known.exists():
                client.load_host_keys(str(known))
        except Exception:
            pass
        # vast.ai hosts are rented and re-created all the time: accept new host keys.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key = self.spec.key_file or (str(self.cfg.path("ssh_key")) if self.cfg.ssh_key else None)
        client.connect(
            self.spec.host, port=self.spec.port, username=self.spec.user,
            key_filename=key, password=self.cfg.ssh_password or None,
            look_for_keys=True, allow_agent=True,
            timeout=15, banner_timeout=30, auth_timeout=30)
        try:
            client.save_host_keys(str(known))
        except Exception:
            pass
        client.get_transport().set_keepalive(10)
        self.client = client

    def upload(self, files, remote_dir):
        try:
            sftp = self.client.open_sftp()
            try:
                sftp.mkdir(remote_dir)
            except IOError:
                pass
            for f in files:
                sftp.put(str(f), "%s/%s" % (remote_dir, Path(f).name))
            sftp.close()
        except Exception:
            # no SFTP subsystem: fall back to cat over exec
            self.run("mkdir -p $HOME/%s" % remote_dir, timeout=20)
            for f in files:
                data = Path(f).read_bytes()
                chan = self.client.get_transport().open_session()
                chan.exec_command("cat > $HOME/%s/%s" % (remote_dir, Path(f).name))
                chan.sendall(data)
                chan.shutdown_write()
                chan.recv_exit_status()
                chan.close()

    def run(self, cmd, timeout=60):
        """Run a command, return (exit code, combined output)."""
        chan = self.client.get_transport().open_session()
        chan.set_combine_stderr(True)
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        out = b""
        deadline = time.time() + timeout
        try:
            while True:
                if time.time() > deadline:
                    raise TimeoutError("timeout: %s" % cmd)
                chunk = chan.recv(65536)
                if not chunk:
                    break
                out += chunk
            code = chan.recv_exit_status()
        finally:
            chan.close()
        return code, out.decode(errors="replace")

    def stream(self, cmd):
        chan = self.client.get_transport().open_session()
        chan.exec_command(cmd)
        return _SSHStream(chan)

    def remote_python(self, remote_dir):
        return "cd $HOME/%s && P=$(command -v python3 || command -v python) && $P" % remote_dir

    def worker_path(self, remote_dir):
        return "$HOME/%s/worker.py" % remote_dir

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None


class _SSHStream(Stream):
    def __init__(self, chan):
        self.chan = chan
        self.lock = threading.Lock()
        self.stdout = chan.makefile("r")
        self.stderr = chan.makefile_stderr("r")

    def write(self, data):
        with self.lock:
            self.chan.sendall(data.encode())

    def lines(self):
        for line in self.stdout:
            yield line if isinstance(line, str) else line.decode(errors="replace")

    def err_lines(self):
        for line in self.stderr:
            yield line if isinstance(line, str) else line.decode(errors="replace")

    def close(self):
        try:
            self.chan.shutdown_write()
        except Exception:
            pass
        try:
            self.chan.close()
        except Exception:
            pass


class LocalTransport(object):
    """Runs the worker on this machine (tests, or a controller box with a GPU)."""

    def __init__(self, spec, cfg):
        self.spec = spec
        self.cfg = cfg
        self.dir = cfg.runtime / "local_worker"

    def connect(self):
        self.dir.mkdir(parents=True, exist_ok=True)

    def upload(self, files, remote_dir):
        for f in files:
            shutil.copy(str(f), str(self.dir / Path(f).name))

    def _argv(self, args):
        return [sys.executable, "-u", str(self.dir / "worker.py")] + list(args) + self.spec.worker_args

    def run_worker(self, args, timeout=600):
        res = subprocess.run(self._argv(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=timeout)
        return res.returncode, res.stdout.decode(errors="replace")

    def run(self, cmd, timeout=60):
        return 127, "local: shell commands not supported"

    def stream_worker(self, args):
        proc = subprocess.Popen(self._argv(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        return _LocalStream(proc)

    def close(self):
        pass


class _LocalStream(Stream):
    def __init__(self, proc):
        self.proc = proc
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.proc.stdin.write(data.encode())
            self.proc.stdin.flush()

    def lines(self):
        for line in iter(self.proc.stdout.readline, b""):
            yield line.decode(errors="replace")

    def err_lines(self):
        for line in iter(self.proc.stderr.readline, b""):
            yield line.decode(errors="replace")

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def make_transport(spec, cfg):
    return LocalTransport(spec, cfg) if spec.local else SSHTransport(spec, cfg)


def kill_pattern(remote_dir):
    """pkill -f pattern that does not match the shell running pkill itself."""
    target = "%s/worker.py" % remote_dir.strip("/")
    return "[%s]%s" % (target[0], target[1:])


def remote_worker_cmd(tr, cfg, args):
    return "%s -u %s %s" % (tr.remote_python(cfg.remote_dir), tr.worker_path(cfg.remote_dir),
                            " ".join(shlex.quote(a) for a in args))


def ensure_python(tr):
    """Images like nvidia/cuda:*-runtime have no Python: install python3 via apt once."""
    code, _ = tr.run("command -v python3 || command -v python", timeout=20)
    if code == 0:
        return True
    code, out = tr.run("(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                       "python3-minimal) >/dev/null 2>&1; command -v python3", timeout=300)
    return code == 0


def run_worker_once(tr, cfg, args, timeout):
    """Run worker.py with args (selftest / bench), return (code, output)."""
    if isinstance(tr, LocalTransport):
        return tr.run_worker(args, timeout=timeout)
    return tr.run("pkill -f '%s'; sleep 0.5; %s" % (kill_pattern(cfg.remote_dir),
                                                  remote_worker_cmd(tr, cfg, args)), timeout=timeout)


# ---------------------------------------------------------------------------
# Worker connection used by `run`
# ---------------------------------------------------------------------------
class WorkerConn(threading.Thread):
    """Keeps one server connected and its worker running; reconnects forever."""

    def __init__(self, spec, cfg, on_found, on_event, on_ready):
        threading.Thread.__init__(self, name="conn-%s" % spec.name)
        self.daemon = True
        self.spec = spec
        self.cfg = cfg
        self.on_found = on_found
        self.on_event = on_event
        self.on_ready = on_ready
        self.status = "ожидание"
        self.gpus = []
        self.hr = 0.0
        self.hr_gpu = []
        self.hr_at = 0.0
        self.ping_ms = None
        self.found = 0
        self.errors = 0
        self.last_error = ""
        self.ready = False
        self.stream = None
        self.tr = None
        self._stop = threading.Event()
        self._ping_sent = {}
        self.current_job = None

    # --- control -----------------------------------------------------------
    def stop(self):
        self._stop.set()
        s = self.stream
        if s:
            try:
                s.write("QUIT\n")
            except Exception:
                pass
            s.close()
        if self.tr:
            self.tr.close()

    def send(self, line):
        s = self.stream
        if not s or not self.ready:
            return False
        try:
            s.write(line + "\n")
            return True
        except Exception as exc:
            self._fail("запись: %s" % exc)
            return False

    def send_job(self, job):
        """JOB <id> <prefix128hex> <sender20hex> <target32hex> <noncePrefix16hex>"""
        self.current_job = job
        if job is None:
            return self.send("IDLE")
        nonce_prefix = os.urandom(16).hex()
        return self.send("JOB %s %s %s %064x %s" % (job.id, job.prefix.hex(), job.sender.hex(),
                                                    job.search_target, nonce_prefix))

    def hashrate(self):
        # stale HR -> 0
        return self.hr if time.time() - self.hr_at < 5 else 0.0

    # --- lifecycle -----------------------------------------------------------
    def _fail(self, msg):
        repeated = msg == self.last_error
        self.last_error = msg
        self.errors += 1
        self.ready = False
        self.status = "ошибка"
        self.on_event("log" if repeated else "warn", "%s: %s" % (self.spec.name, msg))
        s = self.stream
        self.stream = None
        if s:
            try:
                s.close()
            except Exception:
                pass

    def run(self):
        backoff = 3
        while not self._stop.is_set():
            started = time.time()
            try:
                self._session()
            except Exception as exc:
                if not self._stop.is_set():
                    self._fail(str(exc).split("\n")[0][:200])
            self.ready = False
            self.hr = 0.0
            if self._stop.is_set():
                break
            if time.time() - started > 60:
                backoff = 3
            self.status = "повтор через %ds" % backoff
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 60)
        if self.tr:
            self.tr.close()

    def _session(self):
        self.status = "подключение"
        self.tr = tr = make_transport(self.spec, self.cfg)
        tr.connect()
        self.status = "загрузка worker.py"
        tr.upload(WORKER_FILES, self.cfg.remote_dir)
        args = ["--idle-timeout", str(int(self.cfg.worker_idle_timeout))]
        if isinstance(tr, LocalTransport):
            stream = tr.stream_worker(args)
        else:
            if not ensure_python(tr):
                raise RuntimeError("на сервере нет python3 и apt-get не смог его поставить")
            tr.run("pkill -f '%s'; sleep 0.3" % kill_pattern(self.cfg.remote_dir), timeout=20)
            stream = tr.stream(remote_worker_cmd(tr, self.cfg, args).replace("&& $P", "&& exec $P", 1))
        self.stream = stream
        self.status = "selftest + JIT"
        threading.Thread(target=self._stderr_reader, args=(stream,), daemon=True).start()
        threading.Thread(target=self._pinger, args=(stream,), daemon=True).start()
        for raw in stream.lines():
            if self._stop.is_set():
                break
            self._handle(raw.strip())
        if not self._stop.is_set():
            raise RuntimeError("воркер завершился (%s)" % (self.last_error or "EOF"))

    def _pinger(self, stream):
        while not self._stop.is_set() and self.stream is stream:
            token = "%d" % random.getrandbits(40)
            self._ping_sent[token] = time.time()
            try:
                stream.write("PING %s\n" % token)
            except Exception:
                return
            if len(self._ping_sent) > 20:
                self._ping_sent.clear()
            self._stop.wait(2.0)

    def _stderr_reader(self, stream):
        try:
            for line in stream.err_lines():
                line = line.strip()
                if line:
                    self.last_error = line[:200]
                    self.on_event("log", "%s stderr: %s" % (self.spec.name, line[:200]))
        except Exception:
            pass

    def _handle(self, line):
        if not line:
            return
        parts = line.split()
        kind = parts[0]
        if kind == "HR" and len(parts) >= 2:
            self.hr = float(parts[1])
            self.hr_gpu = [float(x) for x in parts[2].split(",")] if len(parts) > 2 else []
            self.hr_at = time.time()
        elif kind == "FOUND" and len(parts) == 4:
            self.found += 1
            self.on_found(self, parts[1], int(parts[2], 16), parts[3])
        elif kind == "PONG" and len(parts) >= 2:
            t = self._ping_sent.pop(parts[1], None)
            if t:
                self.ping_ms = (time.time() - t) * 1000
        elif kind == "READY":
            import json
            try:
                self.gpus = json.loads(line.split(None, 2)[2]) if len(parts) > 2 else []
            except ValueError:
                self.gpus = []
            self.ready = True
            self.status = "работает"
            self.on_event("ok", "%s: готов, %d GPU (%s)" % (
                self.spec.name, len(self.gpus), ", ".join(sorted(set(self.gpus)))))
            self.on_ready(self)
        elif kind == "ERR":
            msg = line[4:]
            self.last_error = msg[:200]
            self.errors += 1
            self.on_event("warn", "%s: %s" % (self.spec.name, msg[:200]))
        elif kind == "LOG":
            self.on_event("log", "%s: %s" % (self.spec.name, line[4:][:200]))
