"""UNICRED proof-of-work: digest, job prefix, calldata, error codes."""
from Crypto.Hash import keccak as _keccak

CHAIN_ID = 130
UNICRED = "0xf60de24F228dc7Ca6fF025958d2eE3A956ED88E5"
CRED = "0x0FBc2Fc1366D5BA517E6ca5A304c10359F554E0D"
POOL_HOOK = "0x44AE6071D11476Ac6C53b7B715182b0386c320cC"
TYPEHASH = bytes.fromhex("d38d03374ffe0ab0595a37e26339a924c9bf9d87e54292dabb07b3bddb3cc04e")
MAX_SUPPLY = 4444
ANCHOR_WINDOW = 250
MAX_UINT256 = (1 << 256) - 1

SEL_MINT = "106c9da1"            # mint(uint256 anchorBlock, uint256 nonce, uint256 maxPrice)
SEL_CHALLENGE = "a4da5da2"       # challenge()
SEL_MINTED = "a2309ff8"          # minted
SEL_TARGET_FOR = "16ccc8c0"      # target(address)
SEL_GLOBAL_TARGET = "c737d485"   # global target
SEL_PRICE = "b9186d7d"           # price(uint256 n)
SEL_LAST_MINT_BLOCK = "9cf5c3f5"  # block of the last mint
SEL_DIGEST = "3a39a703"          # digest(bytes32 blockhash, bytes32 challenge, address miner, uint256 nonce)
MINT_TOPIC = "0xfc7415cd24d41544776bc1f004b64ee4dc521f883a554618c30e37ce200be529"

ERRORS = {
    "6f312cbd": "майнинг не стартовал",
    "52df9fe5": "sold out",
    "440f8b45": "в этом блоке уже был минт",
    "6e84ebb5": "плохой anchor",
    "7ca55c77": "PoW не прошёл",
    "a89fb05f": "цена > maxPrice",
    "a5cc3e35": "msg.value < цены",
}

# Test vector: mint #636, tx 0xa516486426b59c718714e062d4c026ccde1deef8cf5343243c93f67b9b30f699
TV = {
    "anchor_block": 59492322,
    "blockhash": "0xcc06b669726f684d0d4e2ab050dbe9abcd592e41741cdd280eb8609956618656",
    "challenge": "0x0000000000062445aa544d049a58f68155e7c268a6334635ed9796e9f8cf4fdb",
    "miner": "0x67d09f31a4453b11f39007a982248bde00625902",
    "nonce": 0x877FACE3B45DDC27C83C163A1DC14E34E2FE5E2F92BF3C0032098B94396A2939,
    "digest": "0x000000000000edf23d93bdc90caa71781e8c9bd831b71e8d5b2ae17271bced8e",
}


def keccak256(data):
    return _keccak.new(digest_bits=256, data=bytes(data)).digest()


def b32(value):
    """0x-hex / bytes / int -> 32 bytes."""
    if isinstance(value, int):
        return value.to_bytes(32, "big")
    if isinstance(value, str):
        value = bytes.fromhex(value[2:] if value.startswith("0x") else value)
    if len(value) != 32:
        raise ValueError("expected 32 bytes")
    return bytes(value)


def addr20(value):
    if isinstance(value, str):
        value = bytes.fromhex(value[2:] if value.startswith("0x") else value)
    if len(value) != 20:
        raise ValueError("expected 20-byte address")
    return bytes(value)


def inner_hash(blockhash, challenge):
    return keccak256(b32(blockhash) + b32(challenge))


def job_prefix(blockhash, challenge, chain_id=CHAIN_ID, contract=UNICRED):
    """First 128 bytes of the hashed message (the part fixed for a job)."""
    return (TYPEHASH + chain_id.to_bytes(32, "big") + b"\x00" * 12 + addr20(contract)
            + inner_hash(blockhash, challenge))


def pow_message(blockhash, challenge, miner, nonce, chain_id=CHAIN_ID, contract=UNICRED):
    return (job_prefix(blockhash, challenge, chain_id, contract) + b"\x00" * 12 + addr20(miner)
            + b32(nonce))


def digest(blockhash, challenge, miner, nonce, chain_id=CHAIN_ID, contract=UNICRED):
    """keccak256(abi.encode(TYPEHASH, chainid, UNICRED, inner, miner, nonce)) as bytes."""
    return keccak256(pow_message(blockhash, challenge, miner, nonce, chain_id, contract))


def digest_int(*args, **kw):
    return int.from_bytes(digest(*args, **kw), "big")


def mint_calldata(anchor_block, nonce, max_price):
    return "0x" + SEL_MINT + "%064x%064x%064x" % (anchor_block, nonce, max_price)


def decode_mint_calldata(data):
    data = data[2:] if data.startswith("0x") else data
    if data[:8] != SEL_MINT or len(data) != 8 + 192:
        raise ValueError("not a mint() calldata")
    words = [int(data[8 + 64 * i:8 + 64 * (i + 1)], 16) for i in range(3)]
    return {"anchor_block": words[0], "nonce": words[1], "max_price": words[2]}


def log2_str(value):
    """Human form of a 256-bit target: 2^211.3"""
    import math
    if not value:
        return "0"
    return "2^%.1f" % math.log2(value)


def decode_error(data):
    """Revert data -> (selector, human text)."""
    if not data:
        return None, "revert без данных"
    if isinstance(data, bytes):
        data = data.hex()
    data = data[2:] if data.startswith("0x") else data
    sel = data[:8].lower()
    return sel, ERRORS.get(sel, "revert 0x" + sel)
