import numpy as np
from pathlib import Path

class TensorRTInference:
    """TensorRT Inference Wrapper — TRT v10 compatible."""
    
    def __init__(self, engine_path: str):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine nicht gefunden: {self.engine_path}")
        import tensorrt as trt
        import pycuda.driver as cuda
        self.trt = trt
        self.cuda = cuda
        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(str(self.engine_path), 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._setup_io()
        print(f"[TRT] Engine: {self.engine_path.name}")

    def _setup_io(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = int(np.prod(shape))
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            info = {
                'name': name,
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'size': size,
            }
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.inputs.append(info)
            else:
                self.outputs.append(info)
            # TRT v10: set tensor address
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_tensor_address(name, int(device_mem))
            else:
                self.context.set_output_tensor_address(name, int(device_mem))

    def _execute(self):
        """TRT v10: execute_async_v3 with stream handle only."""
        self.context.execute_async_v3(stream_handle=self.stream.handle)

    def _transfer_input(self, tensor_data, input_idx=0):
        """Copy input data to GPU."""
        inp = self.inputs[input_idx]
        np.copyto(inp['host'], tensor_data.ravel())
        self.cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

    def _transfer_output(self):
        """Copy output data from GPU."""
        results = []
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        for out in self.outputs:
            results.append(out['host'].reshape(out['shape']).squeeze())
        return results[0] if len(results) == 1 else results

    def infer(self, left, right):
        """Stereo inference (two inputs)."""
        self._transfer_input(left, 0)
        self._transfer_input(right, 1)
        self._execute()
        return self._transfer_output()

    def infer_single(self, tensor):
        """Monocular inference (single input)."""
        self._transfer_input(tensor, 0)
        self._execute()
        return self._transfer_output()

    def __del__(self):
        try:
            self.cuda_ctx.pop()
        except Exception:
            pass
