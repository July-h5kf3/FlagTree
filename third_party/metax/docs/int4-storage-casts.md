# INT4 storage and numerical conversions on MetaX

This contract applies to ordinary Triton (`triton.language`). The existing
[INT4 tutorial](../../../python/tutorials/mixed_precision/04-int4.py) already
defines the packed representation and provides a complete byte-pattern
roundtrip example. This document records its use on MetaX; the representation
and bitwise conversion recipe predate this PR.

## Types and storage

There is no public `tl.int4` or `tl.uint4` scalar or pointer type. An unpacked
signed INT4 value uses `tl.int8`, restricted to `[-8, 7]`, and occupies one byte.
An unpacked unsigned value uses `tl.uint8`, restricted to `[0, 15]`.

Two logical values occupy one packed `uint8` container:

```text
byte[j] = (value[2*j] & 15) | ((value[2*j + 1] & 15) << 4)
```

The first value occupies the low four bits. Signed values use four-bit two's
complement: codes `0..7` mean `0..7`, and codes `8..15` mean `-8..-1`. For
example, `0x8f` contains `[-1, -8]` in signed form or `[15, 8]` in unsigned form.
This encoding differs from FP4 E2M1 even though both can use byte containers.

`N` logical values occupy `ceil(N / 2)` bytes. A `uint8*` offset, stride or
load/store mask counts bytes, each containing two logical values. A mask cannot
select a single four-bit value. Updating one value requires preserving the
other four bits and coordinating concurrent accesses to the same byte.

For odd `N`, an encoder must explicitly zero the final high four bits, and
the consumer ignores that padding value. An ordinary byte copy preserves all
eight bits, including existing nonzero padding; it does not encode or unpack
the values. Shapes and strides must describe the chosen packing axis.

## Unpacking and widening

For a `uint8` tensor `packed`, the existing signed conversion recipe is:

```python
lo = (packed << 4).to(tl.int8, bitcast=True) >> 4
hi = packed.to(tl.int8, bitcast=True) >> 4
values = tl.interleave(lo, hi)
```

The arithmetic right shifts sign-extend both values into INT8. To decode
unsigned values, use `packed & 15` and `packed >> 4`, then interleave them.
Reinterpreting the whole byte as INT8 alone does not unpack its two values.

Once decoded, all signed and unsigned four-bit values convert exactly to
INT8/16/32/64 and FP16/BF16/FP32/FP64 with ordinary `.to(dtype)` casts. These are
GPU integer operations and ordinary casts, not a dedicated INT4 conversion
API. This PR additionally fixes direct UINT8-to-BF16 lowering; all 256 UINT8
values are checked, not just the unsigned four-bit subset.

## Packing and numerical policy

For an even-sized INT8 tile, packing uses:

```python
lo, hi = tl.split(tl.reshape(values, (BLOCK, 2)))
packed = (((hi & 15) << 4) | (lo & 15)).to(tl.uint8)
```

Mask loads for odd logical lengths and supply zero for the final missing
value. Integer encoding takes the low four bits, equivalent to modulo 16;
it does not clamp. Signed values in `[-8, 7]` roundtrip exactly. Out-of-range
examples are `8 -> 0x8 -> -8` and `-9 -> 0x7 -> 7` after signed decoding.

An explicit floating-point recipe may convert FP16/BF16/FP32/FP64 to INT8
first, using the ordinary cast's truncation toward zero, then pack the low
four bits. The truncated result must fit INT8. This gives, for example,
`7.75 -> 7`, `-0.75 -> 0`, and `8.75 -> 8 -> 0x8 -> -8` after signed decoding.
NaN, infinity and unrepresentable integer results have no guaranteed output
under this recipe and are excluded from its tests.

There is no INT4 quantization interface that implicitly selects RTNE,
saturation, a fractional bit count, scale or zero point. Callers must define
such policies explicitly before packing. Packed UINT8 storage alone does not
establish support for any particular quantization scheme.

## Verification and scope

`python/test/unit/language/test_int4.py` checks all 256 byte patterns, signed
and unsigned decoding, integer and standard floating-point conversions,
modulo packing, finite floating-point truncation, odd lengths, zero padding
and output guards. Roundtrips cover one and four warps. The UINT8-to-BF16
regression is in `python/test/unit/language/test_conversions.py`.

```bash
python -m pytest third_party/metax/python/test/unit/language/test_int4.py \
  third_party/metax/python/test/unit/language/test_conversions.py \
  -k 'metax_int4 or metax_uint8_to_bf16' -q
```

These checks establish the representation and explicit software conversion
recipes on C550. They do not establish native INT4 matrix instructions or an
independent four-bit load/store interface. Gluon is outside this contract.
