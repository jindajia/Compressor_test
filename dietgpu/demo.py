import torch
import torch.nn.functional as F

torch.ops.load_library("/ocean/projects/asc200010p/jjia1/TOOLS/dietgpu/build/lib/libdietgpu.so")
dev = torch.device("cuda:0")


def calc_comp_ratio(input_ts, out_sizes):
    total_input_size = 0
    total_comp_size = 0

    for t, s in zip(input_ts, out_sizes):
        total_input_size += t.numel() * t.element_size()
        total_comp_size += s

    return total_input_size, total_comp_size, total_comp_size / total_input_size


def get_float_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_float_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            True, ts, True, tempMem, comp, sizes
        )
        end.record()

        comp_size = 0

        torch.cuda.synchronize()
        if i > 0:
            comp_time += start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            True, comp_ts, out_ts, True, tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        if i > 0:
            decomp_time += start.elapsed_time(end)

        # validate
        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    comp_time /= num_runs
    decomp_time /= num_runs

    return comp_time, decomp_time, total_size, comp_size


def get_any_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_any_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            False, ts, True, tempMem, comp, sizes
        )
        end.record()

        comp_size = 0

        torch.cuda.synchronize()
        comp_time = start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            False, comp_ts, out_ts, True, tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        decomp_time = start.elapsed_time(end)

        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    return comp_time, decomp_time, total_size, comp_size

def read_data_from_bin(filepath, read_dtype = torch.float32, output_dtype = torch.bfloat16, device = dev):
    with open(filepath, 'rb') as f:
        data = f.read()
    ts = torch.tensor(list(data), dtype=read_dtype, device = dev)
    if ts.dtype is not output_dtype:
        ts = ts.to(output_dtype)
    return ts

def main_test():
    abspath = '/ocean/projects/asc200010p/jjia1/Compressor/data/'
    files = ['cesm-CLDHGH-3600x1800', 'exafel-59200x388', 'hurr-CLOUDf48-500x500x100']
    data_descs = ['3600*1800', '59200*388', '400*500*100']
    read_dtype = torch.float32
    k = 3
    batch_size = 128

    print('\nStart Test for Float codec\n')
    for dt in [torch.bfloat16, torch.float16, torch.float32]:
        for i in range(k):

            print('test case {} of {}, test data name: {}' .format(i+1, k, files[i]))

            #Non Batched
            ts = []
            ts.append(read_data_from_bin(abspath + files[i], read_dtype, dt))

            c, dc, total_size, comp_size = get_float_comp_timings(ts)
            ratio = comp_size / total_size
            c_bw = (total_size / 1e9) / (c * 1e-3)
            dc_bw = (total_size / 1e9) / (dc * 1e-3)

            print("Float codec non-batched perf {} {}" .format(data_descs[i], dt))
            print(
                "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                    c, c_bw, total_size, comp_size, ratio
                )
            )
            print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))

            #Batched
            ts = []
            data = read_data_from_bin(abspath + files[i], read_dtype, dt)
            sub_tensor_length = -(- data.shape[0] // batch_size) ## another way to ceil
            ts = torch.split(data, sub_tensor_length)

            c, dc, total_size, comp_size = get_float_comp_timings(ts)
            ratio = comp_size / total_size
            c_bw = (total_size / 1e9) / (c * 1e-3)
            dc_bw = (total_size / 1e9) / (dc * 1e-3)

            print("Float codec batched perf {} {}" .format(data_descs[i], dt))
            print(
                "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                    c, c_bw, total_size, comp_size, ratio
                )
            )
            print("decomp time {:.3f} ms B/W {:.1f} GB/s\n".format(dc, dc_bw))

    print('\nStart Test for Raw ANS\n')
    for dt in [torch.bfloat16, torch.float16, torch.float32]:
        for i in range(k):
            print('test case {} of {}, test data name: {}' .format(i+1, k, files[i]))

            # Non-batched
            ts = []
            ts.append(read_data_from_bin(abspath + files[i], read_dtype, dt))

            c, dc, total_size, comp_size = get_any_comp_timings(ts)
            ratio = comp_size / total_size
            c_bw = (total_size / 1e9) / (c * 1e-3)
            dc_bw = (total_size / 1e9) / (dc * 1e-3)

            print("Raw ANS byte-wise non-batched perf  {} {}" .format(data_descs[i], dt))
            print(
                "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                    c, c_bw, total_size, comp_size, ratio
                )
            )
            print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))

            # Batched
            ts = []
            data = read_data_from_bin(abspath + files[i], read_dtype, dt)
            sub_tensor_length = -(- data.shape[0] // batch_size) ## another way to ceil
            ts = torch.split(data, sub_tensor_length)


            c, dc, total_size, comp_size = get_any_comp_timings(ts)
            ratio = comp_size / total_size
            c_bw = (total_size / 1e9) / (c * 1e-3)
            dc_bw = (total_size / 1e9) / (dc * 1e-3)

            print("Raw ANS byte-wise batched perf {} {}" .format(data_descs[i], dt))
            print(
                "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                    c, c_bw, total_size, comp_size, ratio
                )
            )
            print("decomp time {:.3f} ms B/W {:.1f} GB/s\n".format(dc, dc_bw))


if __name__ == "__main__":
    main_test()

