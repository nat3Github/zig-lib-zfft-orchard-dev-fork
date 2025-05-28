const std = @import("std");
const math = std.math;
const print = std.debug.print;
const Complex = std.math.Complex;

const fft = @import("FFT").Sr_dit_bi_G_L.fft;

var Bench = @import("Bench").Bench().init();

const verify_only = @import("build_options").verify_only;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    try Bench.setName(allocator, @src());

    inline for (.{ f32, f64 }) |T| {
        print("\n", .{});
        inline for (Bench.m, 0..) |m, i_m| {
            const C: type = Complex(T);

            // input x, output y arrays
            const nfft: usize = std.math.pow(usize, 2, m);
            const x = try allocator.alloc(C, nfft);
            const y = try allocator.alloc(C, nfft);
            const x_ref = try allocator.alloc(C, nfft);
            const y_ref = try allocator.alloc(C, nfft);
            Bench.gen_data(x_ref, x);

            // fft under test ------------------------------------------------------------

            const twiddle_sr_init = @import("Twiddles").Std.sr_init;

            const sr_input_lut_init = @import("LUT").SR.input_lut_init;
            const sr_sched_lut_init = @import("LUT").SR.sched_lut_init;
            const jacobsthal = @import("LUT").SR.jacobsthal;

            var sr_input_lut: [*]usize = undefined;
            var sr_sched_off: [*]usize = undefined;
            var sr_sched_cnt: [*]usize = undefined;

            const w1 = try allocator.alloc(C, nfft / 4);
            const w3 = try allocator.alloc(C, nfft / 4);

            twiddle_sr_init(C, nfft / 4, nfft, w1.ptr, w3.ptr);

            const sr_input_lut_slice = try allocator.alloc(usize, nfft);
            sr_input_lut = sr_input_lut_slice.ptr;
            sr_input_lut_init(usize, nfft, sr_input_lut);

            const sr_sched_cnt_slice = try allocator.alloc(usize, m);
            sr_sched_cnt = sr_sched_cnt_slice.ptr;

            const sr_sched_off_slice = try allocator.alloc(usize, jacobsthal(math.log2(nfft)) + 1);
            sr_sched_off = sr_sched_off_slice.ptr;

            sr_sched_lut_init(usize, nfft, sr_sched_cnt, sr_sched_off);

            const in_place = false;

            const args = .{ .C = C, .nfft = nfft, .w1 = w1.ptr, .w3 = w3.ptr, .yp = y.ptr, .xp = x.ptr, .sr_input_lut = sr_input_lut, .sr_sched_off = sr_sched_off, .sr_sched_cnt = sr_sched_cnt };

            // ---------------------------------------------------------------------------

            if (verify_only) {
                try std.testing.expect(Bench.verify(C, in_place, nfft, y_ref, x_ref, fft, args) == true);
            } else {
                print("\t {s}\t", .{Bench.fft_name});
                try Bench.speedTest(C, in_place, i_m, x_ref, fft, args);
            }
        }
    }

    if (!verify_only) {
        try Bench.writeResult(allocator);
    }
}
