import numpy as np
from pathlib import Path

class TensorRTInference:
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
            info = {'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape}
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.inputs.append(info)
            else:
                self.outputs.append(info)

    def infer(self, left, right):
        """Stereo inference (two inputs)."""
        np.copyto(self.inputs[0]['host'], left.ravel())
        np.copyto(self.inputs[1]['host'], right.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape']).squeeze()

    def infer_single(self, tensor):
        """Monocular inference (single input)."""
        np.copyto(self.inputs[0]['host'], tensor.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape']).squeeze()

    def __del__(self):
        try:
            self.cuda_ctx.pop()
        except Exception:
            pass
