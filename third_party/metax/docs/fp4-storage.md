# Packed FP4 storage and transfers on MetaX

This contract applies to E2M1 data in ordinary Triton (`triton.language`).
It does not introduce a four-bit Triton scalar type. On C550, transfers use
ordinary integer memory operations; FP4 conversion and matrix arithmetic
are separate operations.

## Packing and address units

Two adjacent logical values occupy one `uint8` container:

```text
byte[i] = code[2*i] | (code[2*i + 1] << 4)
```

The first code occupies the low nibble. All 16 E2M1 codes are meaningful,
including positive zero (`0x0`) and negative zero (`0x8`). Copy their raw
bits without floating-point casts or numerical integer conversions.

- `N` logical values occupy `ceil(N / 2)` bytes, excluding scales and alignment.
- With packing along an axis, that axis has `ceil(size / 2)` containers.
- A `uint8*` offset, stride or mask counts bytes, each containing two values.
- An `int32*` offset, stride or mask counts four-byte words, each containing
  eight values. C550 uses little-endian byte order: logical code `j` in a word
  occupies bits `[4*j + 3 : 4*j]` for `0 <= j < 8`.
- Reinterpret pointers or use a storage view to access the same bytes as
  `int32`. `uint8.to(int32)` converts each integer value and does not pack
  four bytes into one word.
- PyTorch `float4_e2m1fn_x2` tensors can be copied by the runtime. Pass their
  `view(torch.uint8)` or `triton.reinterpret(x, tl.uint8)` to Triton kernels;
  direct x2 pointer type mapping is not provided.

## Alignment, masks and tails

Use naturally aligned addresses for `int32` access. Copy `nbytes // 4` complete
words, then copy `nbytes % 4` remaining bytes with masked `uint8` loads/stores.
Do not round a word mask upward and overwrite bytes outside the destination.
Unaligned byte addresses and byte strides can use the ordinary `uint8` path.

For an odd number of logical values, the last high nibble is padding. Encoding
may explicitly fill it with positive zero; `tl.fp_to_fp4` requires an even-sized
input tile, so pad and mask that tile explicitly. A raw copy preserves the
entire final byte, including any existing high-nibble bits. A byte mask cannot
select only one of its two values. Updating one nibble requires preserving the
other nibble and synchronizing any concurrent access to the same container.

## Transfer paths

| Path | Representation and behavior |
| --- | --- |
| Host/device and same-device runtime copies | Preserve packed `uint8`, `int32`, or PyTorch x2 storage bits |
| Global memory to/from registers | Ordinary `tl.load` / `tl.store`, with masks in container units |
| Registers through shared memory | Integer data remains packed; cross-warp gather is tested for `uint8` and `int32` |
| Tensor descriptor load/store/gather/scatter | Host-created and kernel-created descriptors use container shapes, strides, block shapes and offsets |

Descriptor bases must satisfy the existing 16-byte alignment contract; leading
strides count containers and must correspond to multiples of 16 bytes. The
last dimension is contiguous and descriptor blocks contain at least 16 bytes
in that dimension. Integer descriptors support zero padding, not NaN padding.
Out-of-bounds stores/scatters are masked. For example, a packed `[M, K/2]`
`uint8` tensor uses descriptor shape `[M, K/2]`, not `[M, K]`.

On C550, the MetaX descriptor implementation lowers these operations to
non-native synchronous pointer loads/stores. This is not a native FP4 TMA or
device asynchronous-copy interface. Runtime host/device copy APIs have their
own stream semantics and are not evidence of device TMA support.

Transposing packed bytes or words keeps the codes inside each container in
place. It does not transpose individual logical FP4 values or automatically
repack them along a different axis. E8M0 scales are separate byte tensors;
their transfer and grouping are the caller's responsibility.

## Directed checks

`python/test/unit/language/test_fp4.py` checks runtime copies, all 256 byte
patterns, low-nibble-first ordering for byte/word containers, strided and
reversed copies, guards, odd lengths, word tails and shared-memory gather.
`python/test/unit/language/test_tensor_descriptor.py` reuses the FP8 transfer
kernel for packed `uint8`/`int32`, checking load/store/gather/scatter and
roundtrips, negative offsets, zero padding, row gaps and guards.

```bash
python -m pytest third_party/metax/python/test/unit/language/test_fp4.py \
  third_party/metax/python/test/unit/language/test_tensor_descriptor.py \
  -k 'metax_' -q
```

These are bit-preserving transfer checks. They do not establish support for
independent four-bit loads/stores, FP4 atomics, native FP4 asynchronous/TMA
transfers, peer-device copies, or Gluon APIs.
