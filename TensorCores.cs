/*
Script executes a PTX kernel that performs matrix multiplication and accumulation using Tensor Cores.  
It initializes two 16x16 matrices, A and B, with values of 1.0, and third 16x16 matrix C with values of 4.0. 
The kernel multiplies A and B, adds C to the result, and stores the final 16x16 matrix D in device memory. 
After execution, the result (matrix 16x16 with values of 20.0) is copied back to the host and displayed in Unity.
Tensor Cores are specialized hardware units in NVIDIA GPUs designed to accelerate matrix operations, 
particularly those used in deep learning and high-performance computing.
*/

using UnityEngine;
using System;
using System.Runtime.InteropServices;

public class TensorCores : MonoBehaviour
{
	[DllImport("nvcuda.dll")]
	static extern int cuInit(uint flags);

	[DllImport("nvcuda.dll")]
	static extern int cuDeviceGet(out IntPtr device, int ordinal);

	[DllImport("nvcuda.dll", EntryPoint="cuCtxCreate_v2")]
	static extern int cuCtxCreate(out IntPtr pctx, uint flags, IntPtr device);

	[DllImport("nvcuda.dll", EntryPoint = "cuMemAlloc_v2")]
	static extern int cuMemAlloc(out IntPtr dptr, uint bytesize);

	[DllImport("nvcuda.dll")]
	static extern int cuModuleLoadDataEx(out IntPtr module, IntPtr image, uint numOptions, uint options, uint optionValues);

	[DllImport("nvcuda.dll")]
	static extern int cuModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name);

	[DllImport("nvcuda.dll")]
	static extern int cuLaunchKernel(IntPtr f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shared, IntPtr stream, IntPtr[] args, IntPtr[] extra);

	[DllImport("nvcuda.dll", EntryPoint = "cuMemcpyDtoH_v2")]
	static extern int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, uint byteCount);

	[DllImport("nvcuda.dll", EntryPoint = "cuMemFree_v2")] 
	static extern int cuMemFree(IntPtr dptr);

	void Start()
	{
		int width = 16;
		int height = 16;
		IntPtr function, device;
		uint memory = (uint)(width * height * sizeof(float));
		cuInit(0);
		cuDeviceGet(out IntPtr cuDevice, 0);
		cuCtxCreate(out IntPtr context, 0, cuDevice);
		cuMemAlloc(out device, memory);
		byte[] source = System.Text.Encoding.ASCII.GetBytes(PTX);
		IntPtr moduleData = Marshal.AllocHGlobal(source.Length);
		Marshal.Copy(source, 0, moduleData, source.Length);
		cuModuleLoadDataEx(out IntPtr module, moduleData, 0, 0, 0);
		cuModuleGetFunction(out function, module, "_Z6KernelPf");
		GCHandle handle = GCHandle.Alloc(device, GCHandleType.Pinned);
		IntPtr[] addresses = new IntPtr[1] {handle.AddrOfPinnedObject()};
		IntPtr host = Marshal.AllocHGlobal((int)memory);
		cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, IntPtr.Zero, addresses, new IntPtr[1]);
		cuMemcpyDtoH(host, device, memory);
		float[] result = new float[16 * 16];
		Marshal.Copy(host, result, 0, result.Length);
		for (int i = 0; i < 16; i++)
		{
			string row = "";
			for (int j = 0; j < 16; j++)
			{
				row = row + result[i * 16 + j] + " ";
			}
			UnityEngine.Debug.Log(row);
		}
		handle.Free();
		cuMemFree(device);
		Marshal.FreeHGlobal(host);
		Marshal.FreeHGlobal(moduleData);
	}

	static string PTX = 
	@"
		.version 8.5
		.target sm_75
		.address_size 64
		.visible .entry _Z6KernelPf(.param .u64 _Z6KernelPf_param_0)
		{
			.reg .b16 	%rs<3>;
			.reg .f32 	%f<12>;
			.reg .b32 	%r<4>;
			.reg .b64 	%rd<3>;
			ld.param.u64 	%rd1, [_Z6KernelPf_param_0];
			mov.f32 	%f2, 0f3F800000;
			{  cvt.rn.f16.f32 %rs1, %f2;}
			mov.b32 	%r1, {%rs1, %rs1};
			cvta.to.global.u64 	%rd2, %rd1;
			{  cvt.rn.f16.f32 %rs2, %f2;}
			mov.b32 	%r2, {%rs2, %rs2};
			mov.f32 	%f3, 0f40800000;
			wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f4, %f5, %f6, %f7, %f8, %f9, %f10, %f11}, {%r1, %r1, %r1, %r1, %r1, %r1, %r1, %r1},
			{%r2, %r2, %r2, %r2, %r2, %r2, %r2, %r2}, {%f3, %f3, %f3, %f3, %f3, %f3, %f3, %f3};
			mov.u32 	%r3, 16;
			wmma.store.d.sync.aligned.row.m16n16k16.global.f32 	[%rd2], {%f4, %f5, %f6, %f7, %f8, %f9, %f10, %f11}, %r3;
			ret;
		}
	";
}

/*
// Save as tensorcores.cu and compile with NVCC, then copy content from file tensorcores.ptx to variable TensorCores.PTX:
// nvcc -ptx -arch=sm_75 tensorcores.cu
#include <mma.h>

__global__ void Kernel(float* result) 
{
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> d;
	nvcuda::wmma::fill_fragment(a, 1.0f);
	nvcuda::wmma::fill_fragment(b, 1.0f);
	nvcuda::wmma::fill_fragment(c, 4.0f);
	nvcuda::wmma::mma_sync(d, a, b, c);
	nvcuda::wmma::store_matrix_sync(result, d, 16, nvcuda::wmma::mem_row_major);
}

*/
