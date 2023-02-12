using UnityEngine;
using System;
using System.Runtime.InteropServices;

public class CUDA : MonoBehaviour 
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

	Texture2D _Texture;
	int _Resolution = 1024;
	int _Memory;
	IntPtr _Function, _Host, _Device;
	GCHandle[] _GCHandles;
	IntPtr[] _Params;

	void Start()
	{
		_Memory = _Resolution * _Resolution * 4;
		cuInit(0);
		cuDeviceGet(out IntPtr cuDevice, 0);
		cuCtxCreate(out IntPtr context, 0, cuDevice);
		cuMemAlloc(out _Device, (uint)_Memory);
		byte[] source = System.Text.Encoding.ASCII.GetBytes(PTX.Kernel);
		IntPtr moduleData = Marshal.AllocHGlobal(source.Length);
		Marshal.Copy(source, 0, moduleData, source.Length);
		cuModuleLoadDataEx(out IntPtr module, moduleData, 0, 0, 0);
		cuModuleGetFunction(out _Function, module, "mainImage");
		_GCHandles = new GCHandle[2] {GCHandle.Alloc(_Device, GCHandleType.Pinned), GCHandle.Alloc(Time.time, GCHandleType.Pinned)};
		_Params = new IntPtr[2] {_GCHandles[0].AddrOfPinnedObject(), _GCHandles[1].AddrOfPinnedObject()};
		_Host = Marshal.AllocHGlobal(_Memory);
		_Texture = new Texture2D(_Resolution, _Resolution, TextureFormat.RGBA32, false);
	}

	void Update () 
	{
		_GCHandles[1] = GCHandle.Alloc(Time.time, GCHandleType.Pinned);
		_Params[1] = _GCHandles[1].AddrOfPinnedObject();
		cuLaunchKernel(_Function, (uint)_Resolution/8, (uint)_Resolution/8, 1, 8, 8, 1, 0, IntPtr.Zero, _Params, new IntPtr[1]);
		cuMemcpyDtoH(_Host, _Device, (uint)_Memory);
		_Texture.LoadRawTextureData(_Host, _Memory);
		_Texture.Apply();
	}

	void OnGUI()
	{
		GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), _Texture, ScaleMode.StretchToFill, true);
	}

	void OnDestroy()
	{
		for (int i = 0; i < _GCHandles.Length; i++) _GCHandles[i].Free();
		cuMemFree(_Device);
		Marshal.FreeHGlobal(_Host);
		Destroy(_Texture);
	}
}

public class PTX
{	// Source code of CUDA PTX assembly language, example program generates molecular movement. Compiled program is executed on GPU.
	public static string Kernel = 
	@"
		.version 8.0
		.target sm_52
		.address_size 64


		.visible .entry mainImage(.param .u64 _Z9mainImageP6uchar4f_param_0, .param .f32 _Z9mainImageP6uchar4f_param_1)
		{
			.reg .b16 	%rs<4>;
			.reg .f32 	%f<313>;
			.reg .b32 	%r<13>;
			.reg .f64 	%fd<25>;
			.reg .b64 	%rd<5>;
			ld.param.u64 	%rd1, [_Z9mainImageP6uchar4f_param_0];
			ld.param.f32 	%f1, [_Z9mainImageP6uchar4f_param_1];
			cvta.to.global.u64 	%rd2, %rd1;
			mov.u32 	%r1, %ctaid.x;
			mov.u32 	%r2, %ntid.x;
			mov.u32 	%r3, %tid.x;
			mad.lo.s32 	%r4, %r1, %r2, %r3;
			mov.u32 	%r5, %ctaid.y;
			mov.u32 	%r6, %ntid.y;
			mov.u32 	%r7, %tid.y;
			mad.lo.s32 	%r8, %r5, %r6, %r7;
			shl.b32 	%r9, %r8, 10;
			add.s32 	%r10, %r9, %r4;
			cvt.rn.f32.u32 	%f2, %r4;
			cvt.rn.f32.u32 	%f3, %r8;
			mul.f32 	%f4, %f2, 0f3A800000;
			cvt.f64.f32 	%fd1, %f4;
			mul.f64 	%fd2, %fd1, 0d401551EB851EB852;
			cvt.rn.f32.f64 	%f5, %fd2;
			mul.f32 	%f6, %f3, 0f3A800000;
			cvt.f64.f32 	%fd3, %f6;
			mul.f64 	%fd4, %fd3, 0d401551EB851EB852;
			cvt.rn.f32.f64 	%f7, %fd4;
			cvt.f64.f32 	%fd5, %f5;
			add.f64 	%fd6, %fd5, 0dBFE999999999999A;
			cvt.rn.f32.f64 	%f8, %fd6;
			cvt.f64.f32 	%fd7, %f7;
			add.f64 	%fd8, %fd7, 0dBFE3851EB851EB85;
			cvt.rn.f32.f64 	%f9, %fd8;
			add.f32 	%f10, %f1, 0fBFC90FDB;
			div.rn.f32 	%f11, %f10, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f12, %f11;
			sub.f32 	%f13, %f11, %f12;
			fma.rn.f32 	%f14, %f13, 0f40000000, 0fBF800000;
			abs.f32 	%f15, %f14;
			mul.f32 	%f16, %f15, %f15;
			add.f32 	%f17, %f15, %f15;
			mov.f32 	%f18, 0f40400000;
			sub.f32 	%f19, %f18, %f17;
			mul.f32 	%f20, %f16, %f19;
			fma.rn.f32 	%f21, %f20, 0f40000000, 0fBF800000;
			add.f32 	%f22, %f21, %f8;
			add.f32 	%f23, %f1, 0f3FC90FDB;
			add.f32 	%f24, %f23, 0fBFC90FDB;
			div.rn.f32 	%f25, %f24, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f26, %f25;
			sub.f32 	%f27, %f25, %f26;
			fma.rn.f32 	%f28, %f27, 0f40000000, 0fBF800000;
			abs.f32 	%f29, %f28;
			mul.f32 	%f30, %f29, %f29;
			add.f32 	%f31, %f29, %f29;
			sub.f32 	%f32, %f18, %f31;
			mul.f32 	%f33, %f30, %f32;
			fma.rn.f32 	%f34, %f33, 0f40000000, 0fBF800000;
			add.f32 	%f35, %f34, %f9;
			cvt.rmi.f32.f32 	%f36, %f22;
			sub.f32 	%f37, %f22, %f36;
			cvt.rmi.f32.f32 	%f38, %f35;
			sub.f32 	%f39, %f35, %f38;
			add.f32 	%f40, %f37, 0fBF000000;
			add.f32 	%f41, %f39, 0fBF000000;
			mul.f32 	%f42, %f41, %f41;
			fma.rn.f32 	%f43, %f40, %f40, %f42;
			mov.f32 	%f44, 0f3F000000;
			min.f32 	%f45, %f44, %f43;
			add.f64 	%fd9, %fd5, 0dBFD70A3D70A3D70A;
			cvt.rn.f32.f64 	%f46, %fd9;
			add.f64 	%fd10, %fd7, 0dBFC999999999999A;
			cvt.rn.f32.f64 	%f47, %fd10;
			add.f32 	%f48, %f1, 0fBF800000;
			add.f32 	%f49, %f48, 0fBFC90FDB;
			div.rn.f32 	%f50, %f49, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f51, %f50;
			sub.f32 	%f52, %f50, %f51;
			fma.rn.f32 	%f53, %f52, 0f40000000, 0fBF800000;
			abs.f32 	%f54, %f53;
			mul.f32 	%f55, %f54, %f54;
			add.f32 	%f56, %f54, %f54;
			sub.f32 	%f57, %f18, %f56;
			mul.f32 	%f58, %f55, %f57;
			fma.rn.f32 	%f59, %f58, 0f40000000, 0fBF800000;
			add.f32 	%f60, %f59, %f46;
			add.f32 	%f61, %f48, 0f3FC90FDB;
			add.f32 	%f62, %f61, 0fBFC90FDB;
			div.rn.f32 	%f63, %f62, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f64, %f63;
			sub.f32 	%f65, %f63, %f64;
			fma.rn.f32 	%f66, %f65, 0f40000000, 0fBF800000;
			abs.f32 	%f67, %f66;
			mul.f32 	%f68, %f67, %f67;
			add.f32 	%f69, %f67, %f67;
			sub.f32 	%f70, %f18, %f69;
			mul.f32 	%f71, %f68, %f70;
			fma.rn.f32 	%f72, %f71, 0f40000000, 0fBF800000;
			add.f32 	%f73, %f72, %f47;
			cvt.rmi.f32.f32 	%f74, %f60;
			sub.f32 	%f75, %f60, %f74;
			cvt.rmi.f32.f32 	%f76, %f73;
			sub.f32 	%f77, %f73, %f76;
			add.f32 	%f78, %f75, 0fBF000000;
			add.f32 	%f79, %f77, 0fBF000000;
			mul.f32 	%f80, %f79, %f79;
			fma.rn.f32 	%f81, %f78, %f78, %f80;
			min.f32 	%f82, %f45, %f81;
			add.f64 	%fd11, %fd5, 0dBFE3333333333333;
			cvt.rn.f32.f64 	%f83, %fd11;
			add.f64 	%fd12, %fd7, 0dBFCEB851EB851EB8;
			cvt.rn.f32.f64 	%f84, %fd12;
			add.f32 	%f85, %f1, 0fC0000000;
			add.f32 	%f86, %f85, 0fBFC90FDB;
			div.rn.f32 	%f87, %f86, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f88, %f87;
			sub.f32 	%f89, %f87, %f88;
			fma.rn.f32 	%f90, %f89, 0f40000000, 0fBF800000;
			abs.f32 	%f91, %f90;
			mul.f32 	%f92, %f91, %f91;
			add.f32 	%f93, %f91, %f91;
			sub.f32 	%f94, %f18, %f93;
			mul.f32 	%f95, %f92, %f94;
			fma.rn.f32 	%f96, %f95, 0f40000000, 0fBF800000;
			add.f32 	%f97, %f96, %f83;
			add.f32 	%f98, %f85, 0f3FC90FDB;
			add.f32 	%f99, %f98, 0fBFC90FDB;
			div.rn.f32 	%f100, %f99, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f101, %f100;
			sub.f32 	%f102, %f100, %f101;
			fma.rn.f32 	%f103, %f102, 0f40000000, 0fBF800000;
			abs.f32 	%f104, %f103;
			mul.f32 	%f105, %f104, %f104;
			add.f32 	%f106, %f104, %f104;
			sub.f32 	%f107, %f18, %f106;
			mul.f32 	%f108, %f105, %f107;
			fma.rn.f32 	%f109, %f108, 0f40000000, 0fBF800000;
			add.f32 	%f110, %f109, %f84;
			cvt.rmi.f32.f32 	%f111, %f97;
			sub.f32 	%f112, %f97, %f111;
			cvt.rmi.f32.f32 	%f113, %f110;
			sub.f32 	%f114, %f110, %f113;
			add.f32 	%f115, %f112, 0fBF000000;
			add.f32 	%f116, %f114, 0fBF000000;
			mul.f32 	%f117, %f116, %f116;
			fma.rn.f32 	%f118, %f115, %f115, %f117;
			min.f32 	%f119, %f82, %f118;
			add.f64 	%fd13, %fd5, 0dBFC70A3D70A3D70A;
			cvt.rn.f32.f64 	%f120, %fd13;
			add.f64 	%fd14, %fd7, 0dBFEA3D70A3D70A3D;
			cvt.rn.f32.f64 	%f121, %fd14;
			add.f32 	%f122, %f1, 0fC0400000;
			add.f32 	%f123, %f122, 0fBFC90FDB;
			div.rn.f32 	%f124, %f123, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f125, %f124;
			sub.f32 	%f126, %f124, %f125;
			fma.rn.f32 	%f127, %f126, 0f40000000, 0fBF800000;
			abs.f32 	%f128, %f127;
			mul.f32 	%f129, %f128, %f128;
			add.f32 	%f130, %f128, %f128;
			sub.f32 	%f131, %f18, %f130;
			mul.f32 	%f132, %f129, %f131;
			fma.rn.f32 	%f133, %f132, 0f40000000, 0fBF800000;
			add.f32 	%f134, %f133, %f120;
			add.f32 	%f135, %f122, 0f3FC90FDB;
			add.f32 	%f136, %f135, 0fBFC90FDB;
			div.rn.f32 	%f137, %f136, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f138, %f137;
			sub.f32 	%f139, %f137, %f138;
			fma.rn.f32 	%f140, %f139, 0f40000000, 0fBF800000;
			abs.f32 	%f141, %f140;
			mul.f32 	%f142, %f141, %f141;
			add.f32 	%f143, %f141, %f141;
			sub.f32 	%f144, %f18, %f143;
			mul.f32 	%f145, %f142, %f144;
			fma.rn.f32 	%f146, %f145, 0f40000000, 0fBF800000;
			add.f32 	%f147, %f146, %f121;
			cvt.rmi.f32.f32 	%f148, %f134;
			sub.f32 	%f149, %f134, %f148;
			cvt.rmi.f32.f32 	%f150, %f147;
			sub.f32 	%f151, %f147, %f150;
			add.f32 	%f152, %f149, 0fBF000000;
			add.f32 	%f153, %f151, 0fBF000000;
			mul.f32 	%f154, %f153, %f153;
			fma.rn.f32 	%f155, %f152, %f152, %f154;
			min.f32 	%f156, %f119, %f155;
			mul.f32 	%f157, %f5, 0f3FB50481;
			mul.f32 	%f158, %f7, 0f3FB50481;
			cvt.f64.f32 	%fd15, %f157;
			add.f64 	%fd16, %fd15, 0dBFDCCCCCCCCCCCCD;
			cvt.rn.f32.f64 	%f159, %fd16;
			cvt.f64.f32 	%fd17, %f158;
			add.f64 	%fd18, %fd17, 0dBFD3333333333333;
			cvt.rn.f32.f64 	%f160, %fd18;
			add.f32 	%f161, %f1, 0fC0800000;
			add.f32 	%f162, %f161, 0fBFC90FDB;
			div.rn.f32 	%f163, %f162, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f164, %f163;
			sub.f32 	%f165, %f163, %f164;
			fma.rn.f32 	%f166, %f165, 0f40000000, 0fBF800000;
			abs.f32 	%f167, %f166;
			mul.f32 	%f168, %f167, %f167;
			add.f32 	%f169, %f167, %f167;
			sub.f32 	%f170, %f18, %f169;
			mul.f32 	%f171, %f168, %f170;
			fma.rn.f32 	%f172, %f171, 0f40000000, 0fBF800000;
			add.f32 	%f173, %f172, %f159;
			add.f32 	%f174, %f161, 0f3FC90FDB;
			add.f32 	%f175, %f174, 0fBFC90FDB;
			div.rn.f32 	%f176, %f175, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f177, %f176;
			sub.f32 	%f178, %f176, %f177;
			fma.rn.f32 	%f179, %f178, 0f40000000, 0fBF800000;
			abs.f32 	%f180, %f179;
			mul.f32 	%f181, %f180, %f180;
			add.f32 	%f182, %f180, %f180;
			sub.f32 	%f183, %f18, %f182;
			mul.f32 	%f184, %f181, %f183;
			fma.rn.f32 	%f185, %f184, 0f40000000, 0fBF800000;
			add.f32 	%f186, %f185, %f160;
			cvt.rmi.f32.f32 	%f187, %f173;
			sub.f32 	%f188, %f173, %f187;
			cvt.rmi.f32.f32 	%f189, %f186;
			sub.f32 	%f190, %f186, %f189;
			add.f32 	%f191, %f188, 0fBF000000;
			add.f32 	%f192, %f190, 0fBF000000;
			mul.f32 	%f193, %f192, %f192;
			fma.rn.f32 	%f194, %f191, %f191, %f193;
			min.f32 	%f195, %f156, %f194;
			add.f64 	%fd19, %fd15, 0dBFA47AE147AE147B;
			cvt.rn.f32.f64 	%f196, %fd19;
			add.f64 	%fd20, %fd17, 0dBFEC28F5C28F5C29;
			cvt.rn.f32.f64 	%f197, %fd20;
			add.f32 	%f198, %f1, 0fC0A00000;
			add.f32 	%f199, %f198, 0fBFC90FDB;
			div.rn.f32 	%f200, %f199, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f201, %f200;
			sub.f32 	%f202, %f200, %f201;
			fma.rn.f32 	%f203, %f202, 0f40000000, 0fBF800000;
			abs.f32 	%f204, %f203;
			mul.f32 	%f205, %f204, %f204;
			add.f32 	%f206, %f204, %f204;
			sub.f32 	%f207, %f18, %f206;
			mul.f32 	%f208, %f205, %f207;
			fma.rn.f32 	%f209, %f208, 0f40000000, 0fBF800000;
			add.f32 	%f210, %f209, %f196;
			add.f32 	%f211, %f198, 0f3FC90FDB;
			add.f32 	%f212, %f211, 0fBFC90FDB;
			div.rn.f32 	%f213, %f212, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f214, %f213;
			sub.f32 	%f215, %f213, %f214;
			fma.rn.f32 	%f216, %f215, 0f40000000, 0fBF800000;
			abs.f32 	%f217, %f216;
			mul.f32 	%f218, %f217, %f217;
			add.f32 	%f219, %f217, %f217;
			sub.f32 	%f220, %f18, %f219;
			mul.f32 	%f221, %f218, %f220;
			fma.rn.f32 	%f222, %f221, 0f40000000, 0fBF800000;
			add.f32 	%f223, %f222, %f197;
			cvt.rmi.f32.f32 	%f224, %f210;
			sub.f32 	%f225, %f210, %f224;
			cvt.rmi.f32.f32 	%f226, %f223;
			sub.f32 	%f227, %f223, %f226;
			add.f32 	%f228, %f225, 0fBF000000;
			add.f32 	%f229, %f227, 0fBF000000;
			mul.f32 	%f230, %f229, %f229;
			fma.rn.f32 	%f231, %f228, %f228, %f230;
			min.f32 	%f232, %f195, %f231;
			add.f64 	%fd21, %fd15, 0dBFAEB851EB851EB8;
			cvt.rn.f32.f64 	%f233, %fd21;
			add.f64 	%fd22, %fd17, 0dBFE147AE147AE148;
			cvt.rn.f32.f64 	%f234, %fd22;
			add.f32 	%f235, %f1, 0fC0C00000;
			add.f32 	%f236, %f235, 0fBFC90FDB;
			div.rn.f32 	%f237, %f236, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f238, %f237;
			sub.f32 	%f239, %f237, %f238;
			fma.rn.f32 	%f240, %f239, 0f40000000, 0fBF800000;
			abs.f32 	%f241, %f240;
			mul.f32 	%f242, %f241, %f241;
			add.f32 	%f243, %f241, %f241;
			sub.f32 	%f244, %f18, %f243;
			mul.f32 	%f245, %f242, %f244;
			fma.rn.f32 	%f246, %f245, 0f40000000, 0fBF800000;
			add.f32 	%f247, %f246, %f233;
			add.f32 	%f248, %f235, 0f3FC90FDB;
			add.f32 	%f249, %f248, 0fBFC90FDB;
			div.rn.f32 	%f250, %f249, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f251, %f250;
			sub.f32 	%f252, %f250, %f251;
			fma.rn.f32 	%f253, %f252, 0f40000000, 0fBF800000;
			abs.f32 	%f254, %f253;
			mul.f32 	%f255, %f254, %f254;
			add.f32 	%f256, %f254, %f254;
			sub.f32 	%f257, %f18, %f256;
			mul.f32 	%f258, %f255, %f257;
			fma.rn.f32 	%f259, %f258, 0f40000000, 0fBF800000;
			add.f32 	%f260, %f259, %f234;
			cvt.rmi.f32.f32 	%f261, %f247;
			sub.f32 	%f262, %f247, %f261;
			cvt.rmi.f32.f32 	%f263, %f260;
			sub.f32 	%f264, %f260, %f263;
			add.f32 	%f265, %f262, 0fBF000000;
			add.f32 	%f266, %f264, 0fBF000000;
			mul.f32 	%f267, %f266, %f266;
			fma.rn.f32 	%f268, %f265, %f265, %f267;
			min.f32 	%f269, %f232, %f268;
			add.f64 	%fd23, %fd15, 0dBFE47AE147AE147B;
			cvt.rn.f32.f64 	%f270, %fd23;
			add.f64 	%fd24, %fd17, 0dBFBEB851EB851EB8;
			cvt.rn.f32.f64 	%f271, %fd24;
			add.f32 	%f272, %f1, 0fC0E00000;
			add.f32 	%f273, %f272, 0fBFC90FDB;
			div.rn.f32 	%f274, %f273, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f275, %f274;
			sub.f32 	%f276, %f274, %f275;
			fma.rn.f32 	%f277, %f276, 0f40000000, 0fBF800000;
			abs.f32 	%f278, %f277;
			mul.f32 	%f279, %f278, %f278;
			add.f32 	%f280, %f278, %f278;
			sub.f32 	%f281, %f18, %f280;
			mul.f32 	%f282, %f279, %f281;
			fma.rn.f32 	%f283, %f282, 0f40000000, 0fBF800000;
			add.f32 	%f284, %f283, %f270;
			add.f32 	%f285, %f272, 0f3FC90FDB;
			add.f32 	%f286, %f285, 0fBFC90FDB;
			div.rn.f32 	%f287, %f286, 0f40C90FDB;
			cvt.rmi.f32.f32 	%f288, %f287;
			sub.f32 	%f289, %f287, %f288;
			fma.rn.f32 	%f290, %f289, 0f40000000, 0fBF800000;
			abs.f32 	%f291, %f290;
			mul.f32 	%f292, %f291, %f291;
			add.f32 	%f293, %f291, %f291;
			sub.f32 	%f294, %f18, %f293;
			mul.f32 	%f295, %f292, %f294;
			fma.rn.f32 	%f296, %f295, 0f40000000, 0fBF800000;
			add.f32 	%f297, %f296, %f271;
			cvt.rmi.f32.f32 	%f298, %f284;
			sub.f32 	%f299, %f284, %f298;
			cvt.rmi.f32.f32 	%f300, %f297;
			sub.f32 	%f301, %f297, %f300;
			add.f32 	%f302, %f299, 0fBF000000;
			add.f32 	%f303, %f301, 0fBF000000;
			mul.f32 	%f304, %f303, %f303;
			fma.rn.f32 	%f305, %f302, %f302, %f304;
			min.f32 	%f306, %f269, %f305;
			mul.f32 	%f307, %f306, 0f40400000;
			sqrt.rn.f32 	%f308, %f307;
			mov.f32 	%f309, 0f3F800000;
			sub.f32 	%f310, %f309, %f308;
			mov.f32 	%f311, 0f00000000;
			cvt.rzi.u32.f32 	%r11, %f311;
			mul.f32 	%f312, %f310, 0f437F0000;
			cvt.rzi.u32.f32 	%r12, %f312;
			mul.wide.u32 	%rd3, %r10, 4;
			add.s64 	%rd4, %rd2, %rd3;
			cvt.u16.u32 	%rs1, %r12;
			cvt.u16.u32 	%rs2, %r11;
			mov.u16 	%rs3, 255;
			st.global.v4.u8 	[%rd4], {%rs2, %rs2, %rs1, %rs3};
			ret;
		}
	";
}

/*
Minimal example:

nvcc -ptx test.cu

__global__ void mainImage(uchar4 *fragColor, float iTime)
{
	int width = 1024;
	int height = 1024;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = x + width * y;
	float2 iResolution = make_float2((float)width, (float)height);
	float2 fragCoord = make_float2((float)x, (float)y);	
	float2 uv = make_float2(fragCoord.x / iResolution.x, fragCoord.y / iResolution.y);
	float4 color = make_float4(uv.x, uv.y, 0.0, 1.0);
	fragColor[i] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
}
*/