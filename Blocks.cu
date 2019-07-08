// nvcc -o Blocks.dll --shared Blocks.cu

__global__ void mainImage(uchar4 *fragColor, float iTime)
{	
	asm volatile(
		".reg .b16 	%rs<5>;"
		".reg .f32 	%f<92>;"
		".reg .b32 	%r<16>;"
		".reg .b64 	%rd<5>;"
		"ld.param.u64 	%rd1, [_Z9mainImageP6uchar4f_param_0];"
		"ld.param.f32 	%f1, [_Z9mainImageP6uchar4f_param_1];"
		"cvta.to.global.u64 	%rd2, %rd1;"
		"mov.u32 	%r1, %ctaid.x;"
		"mov.u32 	%r2, %ntid.x;"
		"mov.u32 	%r3, %tid.x;"
		"mad.lo.s32 	%r4, %r2, %r1, %r3;"
		"mov.u32 	%r5, %ctaid.y;"
		"mov.u32 	%r6, %ntid.y;"
		"mov.u32 	%r7, %tid.y;"
		"mad.lo.s32 	%r8, %r6, %r5, %r7;"
		"mov.u32 	%r9, 4195327;"
		"sub.s32 	%r10, %r9, %r8;"
		"shl.b32 	%r11, %r10, 10;"
		"add.s32 	%r12, %r11, %r4;"
		"cvt.rn.f32.u32	%f2, %r4;"
		"cvt.rn.f32.u32	%f3, %r8;"
		"mul.f32 	%f4, %f3, 0f3A800000;"
		"fma.rn.f32 	%f5, %f4, 0f40000000, %f1;"
		"add.f32 	%f6, %f5, 0fBFC90FDA;"
		"div.rn.f32 	%f7, %f6, 0f40C90FDA;"
		"cvt.rmi.f32.f32	%f8, %f7;"
		"sub.f32 	%f9, %f7, %f8;"
		"fma.rn.f32 	%f10, %f9, 0f40000000, 0fBF800000;"
		"abs.f32 	%f11, %f10;"
		"mul.f32 	%f12, %f11, %f11;"
		"add.f32 	%f13, %f11, %f11;"
		"mov.f32 	%f14, 0f40400000;"
		"sub.f32 	%f15, %f14, %f13;"
		"mul.f32 	%f16, %f12, %f15;"
		"fma.rn.f32 	%f17, %f16, 0f40000000, 0fBF800000;"
		"div.rn.f32 	%f18, %f17, 0f40A00000;"
		"fma.rn.f32 	%f19, %f2, 0f3A800000, %f18;"
		"fma.rn.f32 	%f20, %f19, 0f40000000, %f1;"
		"add.f32 	%f21, %f20, 0f3FC90FDB;"
		"add.f32 	%f22, %f21, 0fBFC90FDA;"
		"div.rn.f32 	%f23, %f22, 0f40C90FDA;"
		"cvt.rmi.f32.f32	%f24, %f23;"
		"sub.f32 	%f25, %f23, %f24;"
		"fma.rn.f32 	%f26, %f25, 0f40000000, 0fBF800000;"
		"abs.f32 	%f27, %f26;"
		"mul.f32 	%f28, %f27, %f27;"
		"add.f32 	%f29, %f27, %f27;"
		"sub.f32 	%f30, %f14, %f29;"
		"mul.f32 	%f31, %f28, %f30;"
		"fma.rn.f32 	%f32, %f31, 0f40000000, 0fBF800000;"
		"div.rn.f32 	%f33, %f32, 0f40A00000;"
		"add.f32 	%f34, %f4, %f33;"
		"mul.f32 	%f35, %f19, 0f41800000;"
		"cvt.rmi.f32.f32	%f36, %f35;"
		"mul.f32 	%f37, %f36, 0f3D800000;"
		"mul.f32 	%f38, %f34, 0f41800000;"
		"cvt.rmi.f32.f32	%f39, %f38;"
		"mul.f32 	%f40, %f39, 0f3D800000;"
		"mul.f32 	%f41, %f40, 0f439BD99A;"
		"fma.rn.f32 	%f42, %f37, 0f42FE3333, %f41;"
		"mul.f32 	%f43, %f40, 0f43374CCD;"
		"fma.rn.f32 	%f44, %f37, 0f4386C000, %f43;"
		"mul.f32 	%f45, %f40, 0f43B9F333;"
		"fma.rn.f32 	%f46, %f37, 0f43D1999A, %f45;"
		"add.f32 	%f47, %f42, 0fBFC90FDA;"
		"div.rn.f32 	%f48, %f47, 0f40C90FDA;"
		"cvt.rmi.f32.f32	%f49, %f48;"
		"sub.f32 	%f50, %f48, %f49;"
		"fma.rn.f32 	%f51, %f50, 0f40000000, 0fBF800000;"
		"abs.f32 	%f52, %f51;"
		"mul.f32 	%f53, %f52, %f52;"
		"add.f32 	%f54, %f52, %f52;"
		"sub.f32 	%f55, %f14, %f54;"
		"mul.f32 	%f56, %f53, %f55;"
		"fma.rn.f32 	%f57, %f56, 0f40000000, 0fBF800000;"
		"mul.f32 	%f58, %f57, 0f472AEE8C;"
		"add.f32 	%f59, %f44, 0fBFC90FDA;"
		"div.rn.f32 	%f60, %f59, 0f40C90FDA;"
		"cvt.rmi.f32.f32	%f61, %f60;"
		"sub.f32 	%f62, %f60, %f61;"
		"fma.rn.f32 	%f63, %f62, 0f40000000, 0fBF800000;"
		"abs.f32 	%f64, %f63;"
		"mul.f32 	%f65, %f64, %f64;"
		"add.f32 	%f66, %f64, %f64;"
		"sub.f32 	%f67, %f14, %f66;"
		"mul.f32 	%f68, %f65, %f67;"
		"fma.rn.f32 	%f69, %f68, 0f40000000, 0fBF800000;"
		"mul.f32 	%f70, %f69, 0f472AEE8C;"
		"add.f32 	%f71, %f46, 0fBFC90FDA;"
		"div.rn.f32 	%f72, %f71, 0f40C90FDA;"
		"cvt.rmi.f32.f32	%f73, %f72;"
		"sub.f32 	%f74, %f72, %f73;"
		"fma.rn.f32 	%f75, %f74, 0f40000000, 0fBF800000;"
		"abs.f32 	%f76, %f75;"
		"mul.f32 	%f77, %f76, %f76;"
		"add.f32 	%f78, %f76, %f76;"
		"sub.f32 	%f79, %f14, %f78;"
		"mul.f32 	%f80, %f77, %f79;"
		"fma.rn.f32 	%f81, %f80, 0f40000000, 0fBF800000;"
		"mul.f32 	%f82, %f81, 0f472AEE8C;"
		"cvt.rmi.f32.f32	%f83, %f58;"
		"sub.f32 	%f84, %f58, %f83;"
		"mul.f32 	%f85, %f84, 0f437F0000;"
		"cvt.rzi.u32.f32	%r13, %f85;"
		"cvt.rmi.f32.f32	%f86, %f70;"
		"sub.f32 	%f87, %f70, %f86;"
		"mul.f32 	%f88, %f87, 0f437F0000;"
		"cvt.rzi.u32.f32	%r14, %f88;"
		"cvt.rmi.f32.f32	%f89, %f82;"
		"sub.f32 	%f90, %f82, %f89;"
		"mul.f32 	%f91, %f90, 0f437F0000;"
		"cvt.rzi.u32.f32	%r15, %f91;"
		"mul.wide.u32 	%rd3, %r12, 4;"
		"add.s64 	%rd4, %rd2, %rd3;"
		"cvt.u16.u32	%rs1, %r15;"
		"cvt.u16.u32	%rs2, %r14;"
		"cvt.u16.u32	%rs3, %r13;"
		"mov.u16 	%rs4, 255;"
		"st.global.v4.u8 	[%rd4], {%rs3, %rs2, %rs1, %rs4};"
		"ret;"
	);
}

extern "C" 
{
	__declspec(dllexport) unsigned char* Render(float time);
	__declspec(dllexport) void Clear();

	unsigned char* host;
	uchar4 *device;

	unsigned char* Render(float time)   //render procedural image
	{
		host = (unsigned char*) malloc(1024*1024*sizeof(uchar4));
		cudaMalloc((void**)&device, 1024*1024*sizeof(uchar4));
		dim3 block(8, 8);
		dim3 grid(128, 128);
		mainImage<<<grid, block>>>(device, time);
		cudaDeviceSynchronize();
		cudaMemcpy(host, device, 1024 * 1024 * sizeof(uchar4), cudaMemcpyDeviceToHost);
		return host;
	}
	
	void Clear()
	{
		free(host);
		cudaFree(device);
	}
}