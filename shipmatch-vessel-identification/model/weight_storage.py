"""Standard-library-only, lossless byte-plane storage for large model files.

This is storage compression, not quantization. Restored file bytes must match
the original SHA256 before any model can be loaded.
"""
from pathlib import Path
import hashlib
import json
import struct
import time
import zlib

MAGIC = b'SHIPWGT1'
CHUNK = 16 * 1024 * 1024


def pack_stream(source, target):
    target.write(MAGIC)
    h = hashlib.sha256(); crc = 0; size = 0; announced = 0
    for raw in iter(lambda: source.read(CHUNK), b''):
        h.update(raw); crc = zlib.crc32(raw, crc); size += len(raw)
        shuffled = raw[0::2] + raw[1::2]
        packed = zlib.compress(shuffled, 2)
        target.write(struct.pack('<II', len(raw), len(packed)))
        target.write(packed)
        if size-announced>=512*1024*1024:
            print('Packed original weight bytes',size,flush=True);announced=size
    return {'bytes': size, 'sha256': h.hexdigest(), 'crc32': crc}


def unpack_stream(source, target=None):
    if source.read(len(MAGIC)) != MAGIC:
        raise ValueError('Unknown packed-weight format')
    h = hashlib.sha256(); crc = 0; size = 0
    while True:
        header = source.read(8)
        if not header:
            break
        if len(header) != 8:
            raise ValueError('Truncated packed-weight block header')
        raw_size, packed_size = struct.unpack('<II', header)
        if not 0 < raw_size <= CHUNK or not 0 < packed_size <= CHUNK + 65536:
            raise ValueError('Invalid packed-weight block size')
        packed = source.read(packed_size)
        if len(packed) != packed_size:
            raise ValueError('Truncated packed-weight block')
        decoder = zlib.decompressobj()
        shuffled = decoder.decompress(packed, raw_size + 1)
        if len(shuffled) != raw_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid packed-weight deflate payload')
        split = (raw_size + 1) // 2
        raw = bytearray(raw_size)
        raw[0::2] = shuffled[:split]; raw[1::2] = shuffled[split:]
        h.update(raw); crc = zlib.crc32(raw, crc); size += raw_size
        if target is not None:
            target.write(raw)
    return {'bytes': size, 'sha256': h.hexdigest(), 'crc32': crc}


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def ensure_local_weights(model_dir):
    root = Path(model_dir).resolve()
    manifest_path = root / 'packed_weights/manifest.json'
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'SHIPWGT1':
        raise ValueError('Unsupported packed-weight manifest')
    marker = root / 'packed_weights/restored.json'
    cache = json.loads(marker.read_text()) if marker.exists() else {}
    started = time.perf_counter(); restored = 0
    for record in manifest['weights']:
        target = (root / record['output']).resolve()
        packed = (root / record['packed']).resolve()
        if not target.is_relative_to(root) or not packed.is_relative_to(root):
            raise ValueError('Packed model path escaped model directory')
        previous = cache.get(record['output'], {})
        if target.exists():
            st = target.stat()
            if (st.st_size == record['bytes'] and previous.get('sha256') == record['sha256']
                    and previous.get('mtime_ns') == st.st_mtime_ns):
                continue
            if st.st_size != record['bytes'] or file_digest(target) != record['sha256']:
                raise ValueError('Existing base weight differs from packaged original: ' + record['output'])
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.restore-part')
            try:
                with packed.open('rb') as source, partial.open('wb') as destination:
                    actual = unpack_stream(source, destination)
                if any(actual[k] != record[k] for k in ['bytes', 'sha256', 'crc32']):
                    raise ValueError('Restored weight checksum mismatch: ' + record['output'])
                partial.replace(target); restored += 1
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        cache[record['output']] = {'sha256': record['sha256'], 'mtime_ns': target.stat().st_mtime_ns}
        temp = marker.with_suffix('.tmp')
        temp.write_text(json.dumps(cache, indent=2), encoding='utf-8'); temp.replace(marker)
    print('Packed base weights verified; restored %d files in %.2f seconds' %
          (restored, time.perf_counter() - started), flush=True)
