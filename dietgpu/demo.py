import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
import io
import os

torch.ops.load_library("/ocean/projects/asc200010p/jjia1/TOOLS/dietgpu/build/lib/libdietgpu.so")
dev = torch.device("cuda:0")


def read_data_to_tensor(filepath: str, 
                       read_dtype: Union[np.dtype, torch.dtype, type] = np.float32, 
                       device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Read binary data from a file and convert it to a PyTorch tensor.

    Parameters:
    - filepath (str): Path to the binary file.
    - read_dtype (Union[np.dtype, torch.dtype, type]): Data type of the binary data in the file.
    - device (torch.device): Device to which the tensor should be moved.

    Returns:
    - torch.Tensor: Tensor containing the data from the file.
    """
    
    if read_dtype == torch.bfloat16:
        # Load bfloat16 data using torch
        data_tensor = torch.load(filepath)
    else:
        with open(filepath, 'rb') as f:
            if isinstance(read_dtype, torch.dtype):
                np_dtype = torch_to_numpy_dtype(read_dtype)
            else:
                np_dtype = read_dtype
            np_array = np.frombuffer(f.read(), dtype=np_dtype).copy()
        data_tensor = torch.from_numpy(np_array)
    
    ts = data_tensor.to(device)
    
    return ts

def convert_data(input_path, input_dtype, output_path, output_dtype):
    """
    Convert data from one dtype to another.

    Parameters:
    - input_path: Path to input binary file.
    - output_path: Path to save converted binary file.
    - input_dtype: Input data type (e.g., torch.float16, torch.float32, torch.bfloat16).
    - output_dtype: Output data type (e.g., torch.bfloat16, torch.float32).
    """

    data_tensor = read_data_to_tensor(input_path, input_dtype)

    # Convert tensor to target dtype
    data_converted_tensor = data_tensor.to(dtype=output_dtype)

    # Save converted tensor to disk
    if output_dtype == torch.bfloat16:
        torch.save(data_converted_tensor, output_path)
    else:
        data_converted_np = data_converted_tensor.numpy()
        data_converted_np.tofile(output_path)

def torch_to_numpy_dtype(torch_dtype):
    """Convert torch dtype to numpy dtype."""
    if torch_dtype == torch.float16:
        return np.float16
    elif torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    else:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")

def calc_comp_ratio(input_ts, out_sizes):
    total_input_size = 0
    total_comp_size = 0

    for t, s in zip(input_ts, out_sizes):
        total_input_size += t.numel() * t.element_size()
        total_comp_size += s

    return total_input_size, total_comp_size, total_comp_size / total_input_size


def get_float_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([500 * 500 * 100 * 6], dtype=torch.uint8, device=dev)

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
    tempMem = torch.empty([500 * 500 * 100 * 6], dtype=torch.uint8, device=dev)

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

def compress_tensor(ts, comp, sizes, save_path, outputname):
    compressed_tensors = []
    comp = comp.cpu() 
    for i in range(len(sizes)):
        compressed_tensor = comp[i, :sizes[i]]
        compressed_tensors.append(compressed_tensor)

    total_size = sum([tensor.numel() * tensor.element_size() for tensor in compressed_tensors])
    print(f"Estimated size of all compressed tensors: {total_size} bytes")

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(len(sizes)):
        with open(save_path+str(i)+'.pth', 'wb') as f:
            f.write(compressed_tensors[i].numpy().tobytes())
    print('save to {}' .format(save_path))

def decompress_tensor(load_path, type, dev):
    loaded_data = torch.load(load_path)
    
    original_shape = loaded_data["original_shape"]
    compressed_tensor = loaded_data["compressed_tensor"]
    ten_len = loaded_data["tensor_length"]
    ten_size = loaded_data["tensor_size"]
    ts = []
    for i in range(ten_len):
        ts.append(torch.empty(ten_size, dtype=type, device=dev))
    decompressed_tensor = compressed_tensor.view(original_shape)
    
    return decompressed_tensor, ts



def compress_to_disk_any(ts, save_path, outputname, batch_size):
    tempMem = torch.empty([500 * 500 * 100 * 6], dtype=torch.uint8, device=dev)

    #do batch
    print('ts.shape = ' + str(ts.shape) + ' ts.shape = ' + str(ts.shape[0]))
    sub_tensor_length = -(- ts.shape[0] // batch_size) ## another way to ceil
    print('sub_tensor_length = ' + str(sub_tensor_length))
    ts = list(torch.split(ts, sub_tensor_length))

    #any compress
    rows, cols = torch.ops.dietgpu.max_any_compressed_output_size(ts)
    print('rows = {} cols = {}' .format(rows, cols))
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)
    # print('len(ts[0]) = ', len(ts[0]))
    print('max_any_compressed_output_size:  comp.shape = {} , max_comp_size / ts_size = ratio = {}' .format(comp.shape, (comp.element_size() * comp.numel()) / (ts[0].element_size() * ts[0].numel() * len(ts))))
    comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
        False, ts, True, tempMem, comp, sizes)
    total_size, comp_size, comp_ratio = calc_comp_ratio(ts, sizes)
    print('ts[0].size = {}, ts[0].element_size() = {}' .format(ts[0].size(), ts[0].element_size()))
    print('any compression {} -> {} bytes ({:.4f}x)' .format(total_size, comp_size, comp_ratio))
    compress_tensor(ts, comp, sizes=sizes, save_path=save_path, outputname=outputname)
    return rows, cols, [len(ts), ts[0].shape[0]]

def decompress_from_disk_any(load_path, rows, cols, original_shape, dtype, device):
    tempMem = torch.empty([500 * 500 * 100 * 6], dtype=torch.uint8, device=device)

    files = sorted([f for f in os.listdir(load_path) if f.endswith('.pth')])
    print('files = ', files)
    decompressed_tensor = torch.zeros([rows, cols], dtype=torch.uint8, device=device)
    for idx, file in enumerate(files):
        with open(os.path.join(load_path, file), 'rb') as f:
            data = torch.tensor(bytearray(f.read()), dtype=torch.uint8, device=device)
            decompressed_tensor[idx, :data.shape[0]] = data
    print(decompressed_tensor.shape)

    out_ts = []
    comp_ts = [*decompressed_tensor]
    for i in range(original_shape[0]):
        out_ts.append(torch.empty(original_shape[1], dtype=dtype, device=device))
    out_status = torch.empty([original_shape[0]], dtype=torch.uint8, device=dev)
    out_sizes = torch.empty([original_shape[0]], dtype=torch.int32, device=dev)
    torch.ops.dietgpu.decompress_data(
        False, comp_ts, out_ts, True, tempMem, out_status, out_sizes
    )
    return torch.cat(out_ts, dim=0)

def main_test():
    abspath = '/ocean/projects/asc200010p/jjia1/Compressor/data_bfloat16/'
    files = ['cesm-CLDHGH-3600x1800', 'exafel-59200x388', 'hurr-CLOUDf48-500x500x100']
    data_descs = ['3600*1800', '59200*388', '500*500*100']
    dt = torch.bfloat16
    k = 3
    batch_size = 128

    print('\nStart Test for Float codec\n')
    for i in range(k):

        print('test case {} of {}, test data name: {}' .format(i+1, k, files[i]))

        #Non Batched
        ts = []
        ts.append(read_data_to_tensor(abspath + files[i], dt, device=dev))

        c, dc, total_size, comp_size = get_float_comp_timings(ts)
        ratio =  total_size / comp_size
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
        data = read_data_to_tensor(abspath + files[i], dt, device=dev)
        sub_tensor_length = -(- data.shape[0] // batch_size) ## another way to ceil
        ts = list(torch.split(data, sub_tensor_length))

        c, dc, total_size, comp_size = get_float_comp_timings(ts)
        ratio = total_size / comp_size
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
    for i in range(k):
        print('test case {} of {}, test data name: {}' .format(i+1, k, files[i]))

        # Non-batched
        ts = []
        ts.append(read_data_to_tensor(abspath + files[i], dt, device=dev))

        c, dc, total_size, comp_size = get_any_comp_timings(ts)
        ratio = total_size / comp_size
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
        data = read_data_to_tensor(abspath + files[i], dt, device=dev)
        sub_tensor_length = -(- data.shape[0] // batch_size) ## another way to ceil
        ts = list(torch.split(data, sub_tensor_length))


        c, dc, total_size, comp_size = get_any_comp_timings(ts)
        ratio = total_size / comp_size
        c_bw = (total_size / 1e9) / (c * 1e-3)
        dc_bw = (total_size / 1e9) / (dc * 1e-3)

        print("Raw ANS byte-wise batched perf {} {}" .format(data_descs[i], dt))
        print(
            "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                c, c_bw, total_size, comp_size, ratio
            )
        )
        print("decomp time {:.3f} ms B/W {:.1f} GB/s\n".format(dc, dc_bw))

def small_test():
    abspath = '/ocean/projects/asc200010p/jjia1/Compressor/data/'
    files = ['hurr-CLOUDf48-500x500x100']
    data_descs = ['500*500*100']
    read_dtype = torch.float16
    k = 1
    batch_size = 128

    print('\nStart Test for Raw ANS\n')
    for dt in [torch.float16, torch.float32]: #Choose from [torch.bfloat16, torch.float16, torch.float32]
        for i in range(k):
            print('test case {} of {}, test data name: {}' .format(i+1, k, files[i]))

            # Non-batched
            ts = []
            ts.append(read_data_to_tensor(abspath + files[i], read_dtype, dev))

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
            data = read_data_to_tensor(abspath + files[i], read_dtype, dt)
            sub_tensor_length = -(- data.shape[0] // batch_size) ## another way to ceil
            ts = list(torch.split(data, sub_tensor_length))


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

def test_comp(filepath, read_type):
    '''
    this function only test compression
    '''
    tempMem = torch.empty([500 * 500 * 100 * 6], dtype=torch.uint8, device=dev)
    ts = []
    dt = read_type
    ts.append(read_data_to_tensor(filepath, read_dtype=read_type, device = dev))
    print('dt = {} , ts[0].shape = {} ' .format(dt, ts[0].shape))
    # print('len(ts) = ', len(ts))

    #any compress
    rows, cols = torch.ops.dietgpu.max_any_compressed_output_size(ts)
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)
    # print('len(ts[0]) = ', len(ts[0]))
    print('max_any_compressed_output_size:  comp.shape = {} , max_comp_size / ts_size = ratio = {}' .format(comp.shape, (comp.element_size() * comp.numel()) / (ts[0].element_size() * ts[0].numel() * len(ts))))
    comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
        False, ts, True, tempMem, comp, sizes)
    total_size, comp_size, comp_ratio = calc_comp_ratio(ts, sizes)
    print('ts = {}, ts[0].numel = {}, ts[0].element_size() = {}' .format(ts, ts[0].numel(), ts[0].element_size()))
    print('any compression {} -> {} bytes ({:.4f}x)' .format(total_size, comp_size, comp_ratio))
    print()
    compress_tensor(comp, sizes=sizes, save_path='any_compress_bin')

    #float_compress
    rows, cols = torch.ops.dietgpu.max_float_compressed_output_size(ts)
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)
    print('max_float_compressed_output_size:  comp.shape = {} ,  max_comp_size / ts_size = ratio = {}' .format(comp.shape,  (comp.element_size() * comp.numel()) / (ts[0].element_size() * ts[0].numel())))
    # print('ts = ', ts)
    comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
        True, ts, True, tempMem, comp, sizes)
    total_size, comp_size, comp_ratio = calc_comp_ratio(ts, sizes)
    print('ts = {}, ts[0].numel = {}, ts[0].element_size() = {}' .format(ts, ts[0].numel(), ts[0].element_size()))
    print('float compression {} -> {} bytes ({:.4f}x)' .format(total_size, comp_size, comp_ratio))

def convert_datas():
    '''
    This is a utilities function, it help me to conver a set of data to another type
    '''
    relative_path = '../../../Compressor/data_float32/'
    ouput_relative_path = '../../../Compressor/data_float16/'
    files = ['cesm-CLDHGH-3600x1800', 'exafel-59200x388', 'hurr-CLOUDf48-500x500x100']
    for i in range(len(files)):
        convert_data(relative_path+files[i], torch.float32, ouput_relative_path+files[i], torch.float16)

def validation_test():
    '''
    this function is used to test if output data is the same with original data
    '''
    relative_path = '../../../Compressor/data_float16/'
    readtype = torch.float16
    files = ['cesm-CLDHGH-3600x1800', 'exafel-59200x388', 'hurr-CLOUDf48-500x500x100']
    for i in range(len(files)):
        ts = read_data_to_tensor(filepath=relative_path+files[i], read_dtype=readtype, device = dev)
        rows, cols, original_shape = compress_to_disk_any(ts=ts, save_path='output_data/'+ str(readtype).split('.')[-1]+'/' +files[i]+'/', outputname=files[i], batch_size=10)
        out_ts = decompress_from_disk_any('output_data/'+ str(readtype).split('.')[-1]+'/' +files[i]+'/', rows= rows, cols=cols, original_shape=original_shape, dtype=readtype, device=dev )
        #start validate
        are_equal = torch.equal(ts, out_ts)
        # Check if the compressed data matches the original data
        print("Compressed data same with original data\n" if are_equal else "Compressed data not same with original data\n")

if __name__ == "__main__":
    # test_comp('/ocean/projects/asc200010p/jjia1/Compressor/data_float32/hurr-CLOUDf48-500x500x100', torch.float32)
    # small_test()
    # main_test()
    # convert_datas()
    validation_test()

