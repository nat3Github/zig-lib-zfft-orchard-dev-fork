const std = @import("std");
const math = std.math;
const print = std.debug.print;
const Complex = std.math.Complex;

const add = @import("complex_math").add;
const mul = @import("complex_math").mul;
const sub = @import("complex_math").sub;

const cp_dit_bf = @import("butterflies").cp_dit_bf;
const cp_dit_bf_0 = @import("butterflies").cp_dit_bf_0;

const get_twiddle = @import("Twiddles").Std.get;

pub fn fft(comptime C: type, N: usize, w: [*]C, out: [*]C, in: [*]C) void {
    cp_dit_di(C, w, N, in, out);
}

pub fn cp_dit_di(comptime C: type, w: [*]C, N: usize, in: [*]C, out: [*]C) void {
    const log2_N: usize = math.log2(N);

    const r: usize = @bitSizeOf(usize) - log2_N;
    var p: usize = 0;
    var q: usize = 0;

    var h: usize = 0;
    var hn: usize = 0;

    while (h < N) : (h = hn) {

        // evalulate binary carry sequence
        hn = h + 2;
        const c: usize = @bitSizeOf(usize) - 2 - @clz(h ^ hn);

        //input indices
        const i_0: usize = math.shr(usize, (p -% q), r);
        const i_1: usize = i_0 ^ math.shr(usize, N, 1);

        if (c & 1 == 1) {
            out[h] = in[i_0];
            out[h + 1] = in[i_1];
        } else {
            const a: C = in[i_0];
            const b: C = in[i_1];
            out[h] = add(a, b);
            out[h + 1] = sub(a, b);
        }

        // higher stages
        var j: usize = 1 - (c & 1);
        while (j < c) : (j += 2) {
            const ss: usize = math.shl(usize, 1, j);
            const rr: usize = h + 2 - 4 * ss;
            const tt: usize = log2_N - j - 2;
            var bb: usize = 0;

            while (bb < ss) : (bb += 1) {
                cp_dit_bf(C, ss, out + rr + bb, get_twiddle(C, math.shl(usize, bb, tt), log2_N, w));
            }
        }

        const m2: usize = math.shr(usize, 0x2000_0000_0000_0000, c);
        const m1: usize = m2 - 1;
        const m: usize = p & m2;

        q = (q & m1) | m;
        p = (p & m1) | (math.shl(usize, (m ^ m2), 1));
    }
}
