import numpy as np
from pathlib import Path

class TensorRTInference:
    """TensorRT Inference Wrapper — auto-detects TRT v8/v9/v10 API."""
    
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
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = int(np.prod(shape))
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            info = {'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape}
            
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.inputs.append(info)
                self.input_names.append(name)
            else:
                self.outputs.append(info)
                self.output_names.append(name)
            
            # Try different API versions for setting tensor addresses
            self._set_tensor_address(name, int(device_mem))

    def _set_tensor_address(self, name, addr):
        """Auto-detect and use the correct tensor address API."""
        ctx = self.context
        # TRT v10
        for fn_name in ['setInputTensorAddress', 'set_input_tensor_address', 
                        'setOutputTensorAddress', 'set_output_tensor_address']:
            fn = getattr(ctx, fn_name, None)
            if fn:
                try:
                    fn(name, addr)
                    return
                except Exception:
                    pass
        # TRT v8/v9 — setTensorAddress
        for fn_name in ['setTensorAddress', 'set_tensor_address']:
            fn = getattr(ctx, fn_name, None)
            if fn:
                try:
                    fn(name, addr)
                    return
                except Exception:
                    pass
        # Fallback: no-op (execution might still work with old-style bindings)
        pass

    def _execute(self):
        """Auto-detect and use the correct execute API."""
        ctx = self.context
        # TRT v10
        fn = getattr(ctx, 'execute_async_v3', None)
        if fn:
            fn(stream_handle=self.stream.handle)
            return
        # TRT v8/v9
        fn = getattr(ctx, 'execute_async_v2', None)
        if fn:
            bindings = [int(inp['device']) for inp in self.inputs] + [int(out['device']) for out in self.outputs]
            fn(bindings=bindings, stream_handle=self.stream.handle)
            return
        raise RuntimeError("Weder execute_async_v3 noch execute_async_v2 verfügbar")

    def _transfer_input(self, tensor_data, input_idx=0):
        inp = self.inputs[input_idx]
        np.copyto(inp['host'], tensor_data.ravel())
        self.cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

    def _transfer_output(self):
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        results = [out['host'].copy().reshape(out['shape']).squeeze() for out in self.outputs]
        return results[0] if len(results) == 1 else results

    def infer(self, left, right):
        self._transfer_input(left, 0)
        self._transfer_input(right, 1)
        self._execute()
        return self._transfer_output()

    def infer_single(self, tensor):
        self._transfer_input(tensor, 0)
        self._execute()
        return self._transfer_output()

    def __del__(self):
        try:
            self.cuda_ctx.pop()
        except Exception:
            pass
