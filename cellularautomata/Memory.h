#pragma once

#include "cuda_runtime.h"
#include "helper_cuda.h"

template<class C> class Memory {
private:
	size_t _size;
	size_t _offset;
	C* _host;
	C* _device;

	cudaError_t cudaStatus;

	cudaError_t setDeviceMemory() {
		cudaStatus = cudaMalloc((void**)&_device, _size * sizeof(C));

		return cudaStatus;
	}

public:
	Memory(size_t size) : _size(size), _offset(0) {
		_host = new C[size];
		_device = 0;
		cudaStatus = cudaSuccess;
	}

	~Memory() {
		delete[] _host;

		if (_device != 0) {
			cudaFree(_device);
		}
	}

	C& operator [] (size_t i) const {
		if (i >= (_size - _offset)) {
			throw "Index of bound exception in Memory";
		}

		return _host[i + _offset];
	}

	C* host() const {
		return _host;
	}

	C* device() {
		if (_device == 0) {
			checkCudaErrors(setDeviceMemory());
		}

		return _device;
	}

	inline size_t size() const { return _size; }

	inline size_t offset() const { return _offset; }
	inline void offset(size_t value) { _offset = value; }

	void assignAll(C value) {
		for (int i = 0; i < _size; _host[i++] = value);
	}

	bool hostToDevice() {
		if (_device == 0) {
			checkCudaErrors(setDeviceMemory());
		}

		checkCudaErrors(cudaStatus = cudaMemcpy(_device, _host, _size * sizeof(C), cudaMemcpyHostToDevice));

		return success();
	}

	bool deviceToHost() {
		if (_device == 0) {
			checkCudaErrors(setDeviceMemory());
		}

		checkCudaErrors(cudaStatus = cudaMemcpy(_host, _device, _size * sizeof(C), cudaMemcpyDeviceToHost));

		return success();
	}

	bool success() {
		return cudaStatus == cudaSuccess;
	}
};

