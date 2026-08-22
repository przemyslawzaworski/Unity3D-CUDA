using UnityEngine;
using UnityEngine.Rendering;
using System;
using System.Runtime.InteropServices;

// Required -force-gfx-direct, for editor or standalone build (otherwise crash)
// This script enables direct CUDA–Direct3D 11 interoperability in Unity, 
// allowing a CUDA kernel to generate image data on the GPU and display it as a Unity texture without copying through CPU memory.
// Add script to Main Camera. Play.
public class CudaD3D11 : MonoBehaviour
{
	[DllImport("nvcuda.dll")]
	static extern int cuInit(uint flags);

	[DllImport("nvcuda.dll", EntryPoint = "cuCtxCreate_v2")]
	static extern int cuCtxCreate(out IntPtr pctx, uint flags, int dev);

	[DllImport("nvcuda.dll", EntryPoint = "cuCtxDestroy_v2")]
	static extern int cuCtxDestroy(IntPtr ctx);

	[DllImport("nvcuda.dll")]
	static extern int cuCtxSetCurrent(IntPtr ctx);

	[DllImport("nvcuda.dll", EntryPoint = "cuMemAlloc_v2")]
	static extern int cuMemAlloc(out ulong dptr, uint bytesize);

	[DllImport("nvcuda.dll", EntryPoint = "cuMemFree_v2")]
	static extern int cuMemFree(ulong dptr);

	[DllImport("nvcuda.dll")]
	static extern int cuModuleLoadDataEx(out IntPtr module, IntPtr image, uint numOptions, IntPtr options, IntPtr optionValues);

	[DllImport("nvcuda.dll")]
	static extern int cuModuleUnload(IntPtr module);

	[DllImport("nvcuda.dll")]
	static extern int cuModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name);

	[DllImport("nvcuda.dll")]
	static extern int cuLaunchKernel(IntPtr f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint shared, IntPtr stream, [In] IntPtr[] kernelParams, IntPtr extra);

	[DllImport("nvcuda.dll")]
	static extern int cuD3D11GetDevices(out uint pCudaDeviceCount, [Out] int[] pCudaDevices, uint cudaDeviceCount, IntPtr pD3D11Device, uint deviceList);

	[DllImport("nvcuda.dll")]
	static extern int cuGraphicsD3D11RegisterResource(out IntPtr pCudaResource, IntPtr pD3DResource, uint flags);

	[DllImport("nvcuda.dll")]
	static extern int cuGraphicsMapResources(uint count, [In] IntPtr[] resources, IntPtr hStream);

	[DllImport("nvcuda.dll")]
	static extern int cuGraphicsUnmapResources(uint count, [In] IntPtr[] resources, IntPtr hStream);

	[DllImport("nvcuda.dll")]
	static extern int cuGraphicsSubResourceGetMappedArray(out IntPtr pArray, IntPtr resource, uint arrayIndex, uint mipLevel);

	[DllImport("nvcuda.dll")]
	static extern int cuGraphicsUnregisterResource(IntPtr resource);

	[DllImport("nvcuda.dll", EntryPoint = "cuMemcpy2D_v2")]
	static extern int cuMemcpy2D(ref CUDA_MEMCPY2D copyParams);

	[UnmanagedFunctionPointer(CallingConvention.StdCall)]
	delegate void ID3D11DeviceChildGetDevice(IntPtr self, out IntPtr device);

	[UnmanagedFunctionPointer(CallingConvention.StdCall)]
	delegate int ID3D11DeviceCreateTexture2D(IntPtr self, ref D3D11_TEXTURE2D_DESC desc, IntPtr initialData, out IntPtr texture2D);

	const uint CU_GRAPHICS_REGISTER_FLAGS_NONE = 0;
	const uint CU_D3D11_DEVICE_LIST_ALL = 1;
	const uint CU_MEMORYTYPE_DEVICE = 2;
	const uint CU_MEMORYTYPE_ARRAY = 3;
	const uint DXGI_FORMAT_R8G8B8A8_UNORM = 28;
	const uint D3D11_USAGE_DEFAULT = 0;
	const uint D3D11_BIND_SHADER_RESOURCE = 0x8;

	[StructLayout(LayoutKind.Sequential)]
	struct DXGI_SAMPLE_DESC
	{
		public uint Count;
		public uint Quality;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct D3D11_TEXTURE2D_DESC
	{
		public uint Width;
		public uint Height;
		public uint MipLevels;
		public uint ArraySize;
		public uint Format;
		public DXGI_SAMPLE_DESC SampleDesc;
		public uint Usage;
		public uint BindFlags;
		public uint CPUAccessFlags;
		public uint MiscFlags;
	}

	[StructLayout(LayoutKind.Sequential)]
	struct CUDA_MEMCPY2D
	{
		public UIntPtr srcXInBytes;
		public UIntPtr srcY;
		public uint srcMemoryType;
		public IntPtr srcHost;
		public ulong srcDevice;
		public IntPtr srcArray;
		public UIntPtr srcPitch;
		public UIntPtr dstXInBytes;
		public UIntPtr dstY;
		public uint dstMemoryType;
		public IntPtr dstHost;
		public ulong dstDevice;
		public IntPtr dstArray;
		public UIntPtr dstPitch;
		public UIntPtr WidthInBytes;
		public UIntPtr Height;
	}

	Texture2D _Texture;
	int _Resolution = 4096;
	int _Memory;
	IntPtr _Context;
	IntPtr _Module;
	IntPtr _Function;
	IntPtr _CudaGraphicsResource;
	IntPtr _D3DTexture;
	IntPtr[] _GraphicsResources;
	ulong _Device;
	float[] _TimeArgument;
	ulong[] _DeviceArgument;
	GCHandle[] _GCHandles;
	IntPtr[] _Params;

	void Start()
	{
		if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D11) { Debug.LogError("CUDA.cs requires Direct3D11. Current API: " + SystemInfo.graphicsDeviceType); enabled = false; return; }
		if (SystemInfo.renderingThreadingMode != RenderingThreadingMode.Direct) Debug.LogWarning("This script is intended for -force-gfx-direct. Current threading mode: " + SystemInfo.renderingThreadingMode);
		_Memory = _Resolution * _Resolution * 4;
		Check(cuInit(0), "cuInit");
		RenderTexture temporary = new RenderTexture(1, 1, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
		temporary.Create();
		IntPtr temporaryResource = temporary.GetNativeTexturePtr();
		if (temporaryResource == IntPtr.Zero) throw new Exception("Temporary RenderTexture.GetNativeTexturePtr() returned NULL.");
		IntPtr d3dDevice = GetD3D11Device(temporaryResource);
		_D3DTexture = CreateD3D11Texture2D(d3dDevice, _Resolution, _Resolution);
		int[] cudaDevices = new int[1];
		Check(cuD3D11GetDevices(out uint cudaDeviceCount, cudaDevices, 1, d3dDevice, CU_D3D11_DEVICE_LIST_ALL), "cuD3D11GetDevices");
		Marshal.Release(d3dDevice);
		temporary.Release();
		Destroy(temporary);
		if (cudaDeviceCount == 0) throw new Exception("The Direct3D11 device used by Unity has no corresponding CUDA device.");
		Check(cuCtxCreate(out _Context, 0, cudaDevices[0]), "cuCtxCreate");
		Check(cuCtxSetCurrent(_Context), "cuCtxSetCurrent");
		Check(cuMemAlloc(out _Device, (uint)_Memory), "cuMemAlloc");
		byte[] source = System.Text.Encoding.ASCII.GetBytes(PTX.Kernel + "\0");
		IntPtr moduleData = Marshal.AllocHGlobal(source.Length);
		Marshal.Copy(source, 0, moduleData, source.Length);
		try { Check(cuModuleLoadDataEx(out _Module, moduleData, 0, IntPtr.Zero, IntPtr.Zero), "cuModuleLoadDataEx"); } finally { Marshal.FreeHGlobal(moduleData); }
		Check(cuModuleGetFunction(out _Function, _Module, "mainImage"), "cuModuleGetFunction");
		Check(cuGraphicsD3D11RegisterResource(out _CudaGraphicsResource, _D3DTexture, CU_GRAPHICS_REGISTER_FLAGS_NONE), "cuGraphicsD3D11RegisterResource");
		_GraphicsResources = new IntPtr[1] { _CudaGraphicsResource };
		_DeviceArgument = new ulong[1] { _Device };
		_TimeArgument = new float[1];
		_GCHandles = new GCHandle[2] { GCHandle.Alloc(_DeviceArgument, GCHandleType.Pinned), GCHandle.Alloc(_TimeArgument, GCHandleType.Pinned) };
		_Params = new IntPtr[2] { _GCHandles[0].AddrOfPinnedObject(), _GCHandles[1].AddrOfPinnedObject() };
		_Texture = Texture2D.CreateExternalTexture(_Resolution, _Resolution, TextureFormat.RGBA32, false, true, _D3DTexture);
		_Texture.wrapMode = TextureWrapMode.Clamp;
		_Texture.filterMode = FilterMode.Point;
	}

	void Update()
	{
		if (_Context == IntPtr.Zero || _CudaGraphicsResource == IntPtr.Zero) return;
		Check(cuCtxSetCurrent(_Context), "cuCtxSetCurrent");
		_TimeArgument[0] = Time.time;
		Check(cuLaunchKernel(_Function, (uint)_Resolution / 8, (uint)_Resolution / 8, 1, 8, 8, 1, 0, IntPtr.Zero, _Params, IntPtr.Zero), "cuLaunchKernel");
		Check(cuGraphicsMapResources(1, _GraphicsResources, IntPtr.Zero), "cuGraphicsMapResources");
		try
		{
			Check(cuGraphicsSubResourceGetMappedArray(out IntPtr array, _CudaGraphicsResource, 0, 0), "cuGraphicsSubResourceGetMappedArray");
			CUDA_MEMCPY2D copy = new CUDA_MEMCPY2D { srcMemoryType = CU_MEMORYTYPE_DEVICE, srcDevice = _Device, srcPitch = (UIntPtr)(_Resolution * 4), dstMemoryType = CU_MEMORYTYPE_ARRAY, dstArray = array, WidthInBytes = (UIntPtr)(_Resolution * 4), Height = (UIntPtr)_Resolution };
			Check(cuMemcpy2D(ref copy), "cuMemcpy2D");
		}
		finally
		{
			Check(cuGraphicsUnmapResources(1, _GraphicsResources, IntPtr.Zero), "cuGraphicsUnmapResources");
		}
	}

	void OnGUI()
	{
		if (_Texture != null) GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), _Texture, ScaleMode.StretchToFill, false);
	}

	void OnDestroy()
	{
		if (_Context != IntPtr.Zero) cuCtxSetCurrent(_Context);
		if (_GCHandles != null) for (int i = 0; i < _GCHandles.Length; i++) if (_GCHandles[i].IsAllocated) _GCHandles[i].Free();
		if (_CudaGraphicsResource != IntPtr.Zero) { cuGraphicsUnregisterResource(_CudaGraphicsResource); _CudaGraphicsResource = IntPtr.Zero; }
		if (_Device != 0) { cuMemFree(_Device); _Device = 0; }
		if (_Module != IntPtr.Zero) { cuModuleUnload(_Module); _Module = IntPtr.Zero; }
		if (_Texture != null) { Destroy(_Texture); _Texture = null; }
		if (_D3DTexture != IntPtr.Zero) { Marshal.Release(_D3DTexture); _D3DTexture = IntPtr.Zero; }
		if (_Context != IntPtr.Zero) { cuCtxDestroy(_Context); _Context = IntPtr.Zero; }
	}

	static IntPtr GetD3D11Device(IntPtr d3dResource)
	{
		IntPtr vtable = Marshal.ReadIntPtr(d3dResource);
		IntPtr getDeviceAddress = Marshal.ReadIntPtr(vtable, IntPtr.Size * 3);
		ID3D11DeviceChildGetDevice getDevice = (ID3D11DeviceChildGetDevice)Marshal.GetDelegateForFunctionPointer(getDeviceAddress, typeof(ID3D11DeviceChildGetDevice));
		getDevice(d3dResource, out IntPtr device);
		if (device == IntPtr.Zero) throw new Exception("ID3D11DeviceChild::GetDevice returned NULL.");
		return device;
	}

	static IntPtr CreateD3D11Texture2D(IntPtr d3dDevice, int width, int height)
	{
		IntPtr vtable = Marshal.ReadIntPtr(d3dDevice);
		IntPtr createTexture2DAddress = Marshal.ReadIntPtr(vtable, IntPtr.Size * 5);
		ID3D11DeviceCreateTexture2D createTexture2D = (ID3D11DeviceCreateTexture2D)Marshal.GetDelegateForFunctionPointer(createTexture2DAddress, typeof(ID3D11DeviceCreateTexture2D));
		D3D11_TEXTURE2D_DESC desc = new D3D11_TEXTURE2D_DESC { Width = (uint)width, Height = (uint)height, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1 }, Usage = D3D11_USAGE_DEFAULT, BindFlags = D3D11_BIND_SHADER_RESOURCE };
		int result = createTexture2D(d3dDevice, ref desc, IntPtr.Zero, out IntPtr texture2D);
		if (result < 0 || texture2D == IntPtr.Zero) throw new Exception("ID3D11Device::CreateTexture2D failed with HRESULT 0x" + result.ToString("X8") + ".");
		return texture2D;
	}

	static void Check(int result, string function)
	{
		if (result != 0) throw new Exception(function + " failed with CUDA error " + result + ".");
	}
}

public class PTX
{	// Source code of CUDA PTX assembly language. Compiled program is executed on GPU.
	public static string Kernel =
	@"
		.version 8.0
		.target sm_52
		.address_size 64

		.visible .entry mainImage(.param .u64 output_param, .param .f32 time_param)
		{
			.reg .b16 %rs<5>;
			.reg .f32 %f<26>;
			.reg .b32 %r<12>;
			.reg .b64 %rd<5>;
			ld.param.u64 %rd1, [output_param];
			ld.param.f32 %f1, [time_param];
			cvta.to.global.u64 %rd2, %rd1;
			mov.u32 %r1, %ctaid.x;
			mov.u32 %r2, %ntid.x;
			mov.u32 %r3, %tid.x;
			mad.lo.u32 %r4, %r1, %r2, %r3;
			mov.u32 %r5, %ctaid.y;
			mov.u32 %r6, %ntid.y;
			mov.u32 %r7, %tid.y;
			mad.lo.u32 %r8, %r5, %r6, %r7;
			shl.b32 %r9, %r8, 12;
			add.u32 %r10, %r9, %r4;
			cvt.rn.f32.u32 %f2, %r4;
			cvt.rn.f32.u32 %f3, %r8;
			mul.f32 %f4, %f2, 0f3A000000;
			add.f32 %f5, %f4, 0fBF800000;
			mul.f32 %f6, %f3, 0f3A000000;
			add.f32 %f7, %f6, 0fBF800000;
			mul.f32 %f8, %f5, %f5;
			fma.rn.f32 %f9, %f7, %f7, %f8;
			sub.f32 %f10, 0f3E800000, %f9;
			abs.f32 %f11, %f10;
			sqrt.rn.f32 %f12, %f11;
			mul.f32 %f13, %f1, 0f3E4CCCCD;
			cos.approx.f32 %f14, %f13;
			sin.approx.f32 %f15, %f13;
			fma.rn.f32 %f16, %f12, %f12, %f9;
			sqrt.rn.f32 %f17, %f16;
			div.rn.f32 %f18, 0f3F800000, %f17;
			mul.f32 %f19, %f7, %f18;
			mul.f32 %f20, %f12, %f18;
			mul.f32 %f21, %f19, %f14;
			fma.rn.f32 %f22, %f20, %f15, %f21;
			max.f32 %f23, %f22, 0f00000000;
			min.f32 %f24, %f23, 0f3F800000;
			mul.f32 %f25, %f24, 0f437F0000;
			cvt.rzi.u32.f32 %r11, %f25;
			cvt.u16.u32 %rs1, %r11;
			cvt.u16.u32 %rs2, %r11;
			cvt.u16.u32 %rs3, %r11;
			mov.u16 %rs4, 255;
			mul.wide.u32 %rd3, %r10, 4;
			add.s64 %rd4, %rd2, %rd3;
			st.global.v4.u8 [%rd4], {%rs1, %rs2, %rs3, %rs4};
			ret;
		}
	";
}