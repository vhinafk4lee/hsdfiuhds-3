# UNICRED GPU miner (Unichain, vast.ai)

GPU-майнер для PoW-минта NFT **UNICRED** в сети Unichain (chainId 130).

* Контракт UNICRED: `0xf60de24F228dc7Ca6fF025958d2eE3A956ED88E5`
* $CRED: `0x0FBc2Fc1366D5BA517E6ca5A304c10359F554E0D`, Pool hook: `0x44AE6071D11476Ac6C53b7B715182b0386c320cC`
* Сайт: https://unicred.fun/#mine

## Как это устроено

```
 ваш ПК (Windows/Mac/Linux)                         серверы vast.ai (GPU)
 ┌───────────────────────────────┐   SSH (paramiko) ┌───────────────────────────┐
 │ unicred.py run  (контроллер)  │ ───JOB/IDLE────▶ │ worker.py (stdlib only)   │
 │  • опрос RPC каждые 250 мс    │ ◀──FOUND/HR───── │  • PTX вшит, JIT драйвером│
 │  • задания, кэш кандидатов    │                  │  • поток на каждую карту  │
 │  • подпись и отправка mint()  │                  │  • ключей НЕТ             │
 │  • лимиты, дашборд, логи      │                  └───────────────────────────┘
 │  • wallet.key (только здесь)  │
 └───────────────────────────────┘
```

* **Воркер** (`worker/worker.py`) — один файл, только стандартная библиотека Python. Ядро Keccak
  (`worker/kernel.cu`) вшито как PTX и загружается через CUDA Driver API (`libcuda.so.1`, ctypes),
  поэтому `nvcc` на сервере не нужен. Нужен только драйвер NVIDIA ≥ 520. Для более старых драйверов
  есть автоматический fallback на PTX ISA 7.4/7.0/6.4, а если на сервере есть `nvcc`, воркер соберёт ядро им.
* **Контроллер** (`unicred.py`) работает у вас. Он заливает воркер на серверы, раздаёт задания,
  перепроверяет найденные nonce, подписывает и отправляет транзакции, следит за лимитами.
* **Приватный ключ** читается только из локального файла `wallet.key`. Он не печатается,
  не пишется в логи и не уходит на серверы: воркеры получают только адрес кошелька,
  потому что работа PoW привязана к `msg.sender`.

PoW (сверено с минтом #636):
```
inner  = keccak256(blockhash(anchorBlock) ‖ challenge)
digest = keccak256(abi.encode(TYPEHASH, 130, UNICRED, inner, msg.sender, nonce))
digest < target(msg.sender)    →  mint(anchorBlock, nonce, maxPrice), payable
```

---

## 1. Установка на Windows

1. Установите **Python 3.9+** с https://www.python.org/downloads/.
   В установщике отметьте **«Add python.exe to PATH»**.
2. Скачайте репозиторий (Code → Download ZIP) и распакуйте, например, в `C:\unicred-miner`.
3. Откройте **PowerShell** или **Windows Terminal** в этой папке и поставьте зависимости:
   ```powershell
   cd C:\unicred-miner
   py -m pip install -r requirements.txt
   ```
   Если команда `py` не находится, используйте `python`.

Дашборд лучше смотреть в **Windows Terminal**, он есть в Windows 11 по умолчанию. В старом `cmd.exe` тоже работает.

### SSH-ключ для vast.ai (один раз)

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub
```
Скопируйте выведенную строку в vast.ai: **Account → Keys → SSH Keys → Add**. Добавить ключ нужно
**до** аренды, иначе он не попадёт на уже созданные серверы.
Контроллер сам найдёт ключ в `C:\Users\<вы>\.ssh\`. Если ключ лежит в другом месте, укажите путь в `ssh_key` в `config.json`.

### Кошелёк

Создайте файл `wallet.key` рядом с `unicred.py` и запишите в него одну строку: приватный ключ в hex (`0x…`).
Лучше завести **отдельный кошелёк под майнинг** и держать на нём ровно столько ETH (сеть Unichain),
сколько готовы потратить.

### Конфиг

```powershell
copy config.example.json config.json
copy servers.example.txt servers.txt
```

Основные параметры `config.json`:

| параметр | по умолчанию | смысл |
|---|---|---|
| `rpc_url` | `https://mainnet.unichain.org` | RPC Unichain. Можно указать свой (Alchemy, QuickNode и т.п.) |
| `send_rpc_urls` | `[]` | дополнительные RPC для отправки транзакции, шлётся во все сразу |
| `private_key_file` | `wallet.key` | файл с ключом |
| `address` | `""` | адрес кошелька. Нужен только для `run --dry-run` без ключа |
| `servers_file` | `servers.txt` | список серверов |
| `ssh_key` | `""` | путь к SSH-ключу. Пусто = искать в `~/.ssh` и ssh-agent |
| `priority_fee_gwei` | `0.05` | priority fee. Минт возможен один на блок, поэтому при одновременной находке побеждает тот, кто платит больше |
| `max_fee_gwei` | `2.0` | потолок maxFeePerGas |
| `gas_limit` | `350000` | лимит газа (минт тратит около 200k) |
| `max_total_spend_eth` | `0.05` | **общий потолок трат** (цена + газ) за всё время, хранится в `runtime/state.json` |
| `max_price_eth` | `0.006` | не минтить дороже этой цены |
| `min_balance_eth` | `0.001` | неснижаемый остаток на кошельке |
| `max_mints` | `10` | максимум успешных минтов (за всё время, `state.json`) |
| `simulate_before_send` | `false` | делать `eth_call` перед отправкой. Это +~100 мс, а проигранная гонка стоит только газа |
| `poll_interval` | `0.25` | период опроса сети, с |
| `anchor_refresh_blocks` | `150` | через сколько блоков обновлять anchor |
| `candidate_cache_shift` | `2` | GPU ищут с target × 2^shift. «Почти-решения» кэшируются и отправляются, если target вырастет (heat остыл) |

Чтобы сбросить счётчики лимитов (`потрачено`, `минтов`), удалите `runtime/state.json`.

---

## 2. Как добавить сервер vast.ai

1. На vast.ai арендуйте инстанс. Подойдёт любой образ с драйвером NVIDIA ≥ 520, например шаблон
   **PyTorch** или **NVIDIA CUDA**. Если в образе нет `python3`, контроллер поставит его сам через apt.
2. В карточке инстанса нажмите **Connect** и скопируйте строку вида
   ```
   ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080
   ```
3. Вставьте её **как есть** отдельной строкой в `servers.txt`. Одна строка — один сервер, строки с `#` считаются комментариями.
   Можно указать отдельный ключ для сервера: `ssh -p 41234 root@ssh5.vast.ai -i C:\keys\vast`.

Файл можно сохранять в любой кодировке: UTF-8, UTF-8 с BOM (Блокнот) или UTF-16 (PowerShell `>`).
Перезапуск контроллера подхватывает изменения в `servers.txt`.

---

## 3. Запуск: check → servers → run --dry-run → run

### 3.1 `check`: онлайн-проверки сети и формулы

```powershell
py unicred.py check
```
Команда проверяет:
* chainId = 130;
* локальный digest совпадает с тест-вектором и с view-функцией контракта `0x3a39a703`
  на тест-векторе и 3 случайных входах;
* `eth_call mint()` с плохим nonce даёт revert `0x7ca55c77` (PoW не прошёл);
* выводит баланс, minted, цену, ваш и глобальный target, ожидаемое число хешей на минт
  и сколько минтов в час даст 1/10/100 GH/s и 1 TH/s.

### 3.2 `servers`: проверка железа

```powershell
py unicred.py servers
```
Для каждого сервера параллельно:
1. подключение и заливка `worker.py`;
2. `nvidia-smi`;
3. `worker.py --selftest`: каждая карта должна найти nonce из тест-вектора. Первый запуск
   включает JIT-компиляцию PTX драйвером и может занять до минуты, дальше берётся из кэша;
4. `worker.py --bench 10`: хешрейт по картам.

В конце выводится итоговая таблица и ожидаемое число минтов в час при текущем target.
Сервер, у которого selftest не прошёл, в работу не пускается: воркер сам исключает сломанные карты
и не сообщает `READY`, пока не найдёт хотя бы одну рабочую.

### 3.3 `run --dry-run`: майнинг без отправки транзакций

```powershell
py unicred.py run --dry-run
```
Всё работает как в боевом режиме, только транзакции **не отправляются**. Найденные решения
пишутся в ленту событий и в `runtime/txs.csv` со статусом `dry-run`. Так можно безопасно
проверить реальную частоту находок и сравнить её с оценкой на дашборде. Ключ для этого режима не обязателен,
достаточно `address` в конфиге.

### 3.4 `run`: боевой режим

```powershell
py unicred.py run
```
Перед стартом контроллер покажет конфиг (без ключа), адрес, баланс, текущую цену и лимиты
и попросит ввести `yes`. Флаг `--yes` пропускает вопрос, а `--no-dashboard` заменяет дашборд
построчным логом (удобно для запуска в фоне).

### Дашборд (обновляется раз в секунду)

```
СЕТЬ     блок, minted/4444, цена, задержка RPC
         мой target 2^x, global 2^x, хешей на минт, интервал между минтами, оценка хешрейта сети
МАЙНИНГ  мой хешрейт, моя доля, ожидаемое число минтов в час, текущее задание и возраст anchor
КОШЕЛЁК  баланс, потрачено / лимит, минты ok / revert / в пути, находки
СЕРВЕРЫ  статус, GPU, хешрейт (и по картам), пинг, сколько нашёл
СОБЫТИЯ  лента: находки, отправленные tx, результаты, ошибки серверов
```

### Как остановить

* **Ctrl+C** в окне контроллера. Контроллер закрывает SSH-каналы, воркеры получают EOF и сразу завершаются.
* Если контроллер упал или пропала сеть, воркер сам завершится через 30 с без команд
  (`worker_idle_timeout`), так что GPU не жгут устаревшую работу.
* **Не забудьте удалить или остановить инстансы на vast.ai**, иначе аренда продолжит списываться.

---

## 4. Логика контроллера

* **Опрос** каждые ~250 мс одним batch JSON-RPC: заголовок последнего блока (номер, `parentHash` =
  `blockhash(head-1)`, baseFee), `challenge`, `target(мой адрес)`, `minted`. Цена перечитывается только
  при изменении `minted`. Если RPC не умеет batch, контроллер переходит на последовательные запросы.
  Ответы отставшей ноды (старый блок или уже сменённый challenge) игнорируются.
* **Задание** = (challenge, anchor, target). Новое задание рассылается при смене challenge, при росте
  target и когда anchor старше 150 блоков. anchor = head−1.
* **Nonce** = 16 байт от контроллера (случайные, свои для каждого сервера и задания) + 8 байт
  (индекс GPU + random) + 8 байт счётчика.
* **Кандидат** (`FOUND`):
  1. отбрасывается, если challenge уже сменился;
  2. digest перепроверяется локально (pycryptodome);
  3. сразу подписывается и отправляется `mint(anchor, nonce, maxPrice = текущая цена)`,
     `value = цена` (излишек контракт возвращает), EIP-1559, gas limit 350k;
  4. на один challenge уходит одна отправка.
* Nonce транзакций считается локально и пересинхронизируется при ошибке.
* Квитанции проверяются в фоне. Для revert контроллер ищет в том же блоке чужой минт и пишет
  «гонка проиграна».

Протокол воркера (stdin/stdout через SSH):
```
контроллер → воркер:  JOB <id> <prefix128hex> <sender20hex> <target32hex> <noncePrefix16hex> | IDLE | PING <t>
воркер → контроллер:  READY <n> <gpu names> | HR <total> <per-gpu> | FOUND <id> <nonce32hex> <digest32hex> | PONG <t> | LOG … | ERR …
```

## 5. Логи

* `runtime/log.txt` — все события;
* `runtime/txs.csv` — транзакции: время, hash, статус, tokenId, nonce, anchor, challenge, цена, value, газ, итог;
* `runtime/state.json` — потрачено, число минтов и транзакции в пути (для лимитов между запусками);
* `runtime/known_hosts` — ключи хостов vast.

---

## 6. Для разработчика

### Структура

```
unicred.py              CLI: check / servers / run
unicred/pow.py          формула digest, calldata, коды ошибок, тест-вектор
unicred/chain.py        eth_call / опрос / логи / отправка
unicred/rpc.py          JSON-RPC (batch + fallback)
unicred/miner.py        задания, кандидаты, подпись, лимиты, квитанции
unicred/servers.py      servers.txt, SSH (paramiko), переподключение, протокол
unicred/signer.py       ключ и подпись EIP-1559
unicred/dashboard.py    консольный дашборд
worker/worker.py        воркер (stdlib, PTX вшит, CUDA Driver API через ctypes)
worker/kernel.cu        ядро (CUDA; собирается и как обычный C++ для тестов)
worker/kernel.ptx       PTX (NVRTC 11.8, compute_52, ISA 7.8)
tools/build_ptx.py      пересборка PTX и встраивание в worker.py, проверка ptxas
tests/                  тесты
```

### Ядро

* Сообщение занимает 192 байта, это 2 блока Keccak (rate 136). Midstate первого блока
  (TYPEHASH | chainid | contract | inner | 8 нулевых байт) воркер считает один раз на CPU. GPU получает
  `c_x = midstate ^ block2` и на каждый nonce делает `s[6] ^= bswap64(counter)` плюс одну keccak-f.
* Keccak полностью развёрнут, ротации сделаны через `__funnelshift_l`. В последнем раунде считаются только lane 0 и 1.
* GPU отбирает кандидата по верхним 128 битам digest, полную проверку делает воркер, а затем контроллер.
* ptxas: 78–80 регистров на sm_70…90 и 95 на sm_52/61, **spill 0**.
* Длина launch подстраивается автоматически под ~40 мс. Между launch'ами воркер проверяет, не пришло ли новое задание.

### Пересборка PTX (Linux или WSL)

```bash
python tools/build_ptx.py --download --check
```
Скрипт скачивает `nvidia-cuda-nvrtc-cu11==11.8.89` и `nvidia-cuda-nvcc-cu12` (ptxas) в `tools/.cache`,
компилирует `kernel.cu` через NVRTC (`--gpu-architecture=compute_52` → PTX ISA 7.8), кладёт
`worker/kernel.ptx`, встраивает PTX в `worker/worker.py` (zlib+base64) и проверяет ptxas для
sm_61/75/86/89/90 без spill.

### Тесты

```bash
python -m unittest discover -s tests -v
```
1. `test_1_pow` — digest в Python (pycryptodome и чистый Keccak воркера) совпадает с тест-вектором.
2. `test_2_kernel_cpu` — `kernel.cu`, собранный как C++, находит nonce тест-вектора и совпадает
   с pycryptodome на 1000 случайных входов.
3. `test_3_ptx` — PTX актуален (NVRTC), проходит ptxas для sm_52…90 без spill, fallback-версии ISA собираются.
4. `test_4_e2e` — сквозной тест: мок RPC на 127.0.0.1, настоящий `worker.py` на CPU-сборке ядра →
   задание → FOUND → подписанная транзакция с правильным calldata. Проверяются лимиты, dry-run,
   `check`, Ctrl+C. В реальную сеть ничего не отправляется.
5. `test_5_worker_cuda_path` — CUDA-путь воркера (ctypes Driver API) на фейковом `libcuda`, который
   исполняет ядро на CPU: selftest, bench, протокол, fallback PTX для старых драйверов.
6. `test_6_windows_files` — `servers.txt`, `config.json`, `wallet.key`, сохранённые Блокнотом (UTF-8 с BOM)
   или PowerShell 5.1 (`>` пишет UTF-16), читаются правильно.

Тестам 2, 3 и 5 нужны Linux и g++ (WSL подойдёт); на Windows они пропускаются.
Результат прогона лежит в `TEST_RESULTS.txt`.

## Частые проблемы

* **selftest FAIL / `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`** — слишком старый драйвер. Воркер пробует
  понизить версию PTX; если не помогло, возьмите другой инстанс (драйвер ≥ 520).
* **`Authentication failed`** — ключ не добавлен в vast до создания инстанса или не найден. Укажите
  `ssh_key` в конфиге.
* **Хешрейт 0 и статус «ПАУЗА»** — сработал лимит (причина видна в строке «МАЙНИНГ»), либо закончились
  токены (sold out).
* **RPC 429 / медленный RPC** — укажите свой `rpc_url`. Задержка RPC напрямую влияет на то, как быстро
  воркеры переключаются на новый challenge.
