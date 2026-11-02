// RUN: triton-opt %s -split-input-file --tritonppu-tle-lower-pipe -verify-diagnostics
// REQUIRES: flagtree-backend-ppu

// Forms the PPU pipe lowering rejects before rewriting. Each diagnostic names
// what the caller wrote, so a pipe that cannot lower is not left to fail
// later in the conversion.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// A lifecycle op whose pipe has no create in the same function: PPU pipes must
// be created and used in one kernel, after inlining.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @missing_create(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    // expected-error @below {{has no matching pipe.create in this function}}
    tle.pipe.writer_acquire %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A reader naming a reader the pipe does not declare.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @undeclared_reader(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], readers = ["local"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    // expected-error @below {{names reader 'remote' that the pipe does not declare}}
    %closed = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A reader naming a reader on a pipe that declares none.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @named_reader_without_declaration(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    // expected-error @below {{names reader 'local' of a pipe without declared readers}}
    %closed = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A pipe with a writer but no reader: the lowering needs both endpoints.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @no_reader(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    // expected-error @below {{needs a writer and at least one reader in this kernel}}
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_acquire %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_commit %ring[%c0] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Two readers of one pipe placed in the same warp group: a hand-over barrier
// counts the writer and one reader group, so each reader needs its own group.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @two_readers_one_group(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    // expected-error @below {{run in the same warp group; one reader per group}}
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], readers = ["a", "b"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_acquire %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_commit %ring[%c0] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    %closed_a = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "a", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.reader_release %ring[%c0] {capacity = 4 : i32, field_names = ["data"], reader_name = "a", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    %closed_b = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "b", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.reader_release %ring[%c0] {capacity = 4 : i32, field_names = ["data"], reader_name = "b", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}
